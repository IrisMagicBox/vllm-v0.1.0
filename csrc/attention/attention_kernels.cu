/*
 * 改编自 https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * 版权所有 (c) 2023, vLLM 团队。
 * 版权所有 (c) 2020-2023, NVIDIA CORPORATION。保留所有权利。
 *
 * 根据 Apache 许可证 2.0 版（"许可证"）进行许可；
 * 除了遵守许可证外，不得使用此文件。
 * 您可以在以下位置获得许可证副本：
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * 除非适用法律要求或书面同意，根据许可证分发的软件
 * 按"现状"分发，不附带任何明示或暗示的担保条件。
 * 有关许可证下权限和限制的具体语言，请参见许可证。
 */
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "attention_dtypes.h"
#include "attention_utils.cuh"

#include <algorithm>

#define WARP_SIZE 32
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace vllm {

// 注意力 softmax 的实用函数。
template<int NUM_WARPS>
inline __device__ float block_sum(float* red_smem, float sum) {
  // 将线程索引分解为 warp / lane。
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  // 计算每个 warp 的总和。
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  // warp 领导者将数据存储到共享内存中。
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // 确保数据在共享内存中。
  __syncthreads();

  // warps 计算最终的总和。
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }

  // warp 内的并行归约。
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  // 广播到其他线程。
  return __shfl_sync(uint32_t(-1), sum, 0);
}

// 网格: (num_heads, num_seqs)。
template<
  typename scalar_t,
  int HEAD_SIZE,
  int BLOCK_SIZE,
  int NUM_THREADS>
__global__ void single_query_cached_kv_attention_kernel(
  scalar_t* __restrict__ out,             // [num_seqs, num_heads, head_size]
  const scalar_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
  const scalar_t* __restrict__ k_cache,   // [num_blocks, num_heads, head_size/x, block_size, x]
  const scalar_t* __restrict__ v_cache,   // [num_blocks, num_heads, head_size, block_size]
  const float scale,
  const int* __restrict__ block_tables,   // [num_seqs, max_num_blocks_per_seq]
  const int* __restrict__ context_lens,   // [num_seqs]
  const int max_num_blocks_per_seq,
  const int q_stride) {
  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int seq_idx = blockIdx.y;

  // 用于存储键或查询的一部分的向量类型。
  // 向量大小配置为线程组中的线程
  // 每次获取或计算 16 字节。
  // 例如，如果线程组大小为 4 且数据类型为 half，
  // 那么向量大小为 16 / (4 * sizeof(half)) == 2。
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // 将查询加载到寄存器中。
  // 线程组中的每个线程都有查询的不同部分。
  // 例如，如果线程组大小为 4，则组中的第一个线程
  // 具有查询的第 0、4、8、... 个向量，第二个线程有第 1、5、9、... 
  // 个查询向量，依此类推。
  // 注意(woosuk)：因为 q 是从 qkv 张量中拆分的，所以它可能不是连续的。
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  Q_vec q_vecs[NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_VECS_PER_THREAD; i++) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[i] = *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }

  // 内存规划。
  extern __shared__ char shared_mem[];
  // 注意(woosuk)：我们使用 FP32 作为 softmax logits 以获得更好的准确性。
  float* logits = reinterpret_cast<float*>(shared_mem);
  // 归约的工作空间。
  __shared__ float red_smem[2 * NUM_WARPS];

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  // 每个线程组每次从键中获取 x 个元素。
  constexpr int x = 16 / sizeof(scalar_t);
  float qk_max = -FLT_MAX;

  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  const int context_len = context_lens[seq_idx];
  const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // 遍历键块。
  // 每个 warp 在每次迭代中获取一个键块。
  // warp 中的每个线程组从块中获取一个键，并计算
  // 与查询的点积。
  for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
    const int physical_block_number = block_table[block_idx];

    // 将键加载到寄存器中。
    // 线程组中的每个线程都有键的不同部分。
    // 例如，如果线程组大小为 4，则组中的第一个线程
    // 具有键的第 0、4、8、... 个向量，第二个线程有第 1、5、9、... 第
    // 个键向量，依此类推。
    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset = (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      K_vec k_vecs[NUM_VECS_PER_THREAD];

#pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        const scalar_t* k_ptr = k_cache + physical_block_number * num_heads * HEAD_SIZE * BLOCK_SIZE
                                        + head_idx * HEAD_SIZE * BLOCK_SIZE
                                        + physical_block_offset * x;
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        const int offset1 = (vec_idx * VEC_SIZE) / x;
        const int offset2 = (vec_idx * VEC_SIZE) % x;
        k_vecs[j] = *reinterpret_cast<const K_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2);
      }

      // 计算点积。
      // 这包括同一线程组中线程的归约。
      const float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs, k_vecs);
      const bool mask = token_idx >= context_len;
    
      if (thread_group_offset == 0) {
        // 将部分归约存储到共享内存中。
        // 注意(woosuk)：必须将掩码的 logits 置零。
        logits[token_idx] = mask ? 0.f : qk;
        // 更新最大值。
        qk_max = mask ? qk_max : fmaxf(qk_max, qk);
      }
    }
  }

  // 在同一线程 warp 中执行归约以获得
  // 每个"warp"的最大 qk 值（尚未跨线程块）。
  // 每个线程组的第 0 个线程已经具有其最大 qk 值。
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();

  // TODO(woosuk)：重构此部分。
  // 获取序列的最大 qk 值。
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }
  // 将最大 qk 值广播到所有线程。
  qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);

  // 获取 exp 值的总和。
  float exp_sum = 0.f;
  for (int i = thread_idx; i < context_len; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  // 计算 softmax。
  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < context_len; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();

  // 每个线程每次将从值缓存中获取 16 字节。
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD = (HEAD_SIZE + NUM_ROWS_PER_ITER - 1) / NUM_ROWS_PER_ITER;

  // 注意(woosuk)：我们使用 FP32 作为累加器以获得更好的准确性。
  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
    const int physical_block_number = block_table[block_idx];
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    L_vec logits_vec;
    from_float(logits_vec, *reinterpret_cast<Float_L_vec*>(logits + token_idx));

    const scalar_t* v_ptr = v_cache + physical_block_number * num_heads * HEAD_SIZE * BLOCK_SIZE
                                    + head_idx * HEAD_SIZE * BLOCK_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
        accs[i] += dot(logits_vec, v_vec);
      }
    }
  }

  // 在每个 warp 内执行归约。
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += __shfl_xor_sync(uint32_t(-1), acc, mask);
    }
    accs[i] = acc;
  }

  // 注意(woosuk)：需要一个屏障，因为 logits 的共享内存空间
  // 被重新用于输出。
  __syncthreads();

  // 在 warp 之间执行归约。
  float* out_smem = reinterpret_cast<float*>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    // 上层 warps 写入共享内存。
    if (warp_idx >= mid && warp_idx < i) {
      float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];
        }
      }
    }
    __syncthreads();

    // 下层 warps 更新输出。
    if (warp_idx < mid) {
      const float* src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];
        }
      }
    }
    __syncthreads();
  }

  // 写入最终输出。
  if (warp_idx == 0) {
    scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
      }
    }
  }
}

} // namespace vllm

#define LAUNCH_ATTENTION_KERNEL(T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS)                        \
  vllm::single_query_cached_kv_attention_kernel<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>        \
  <<<grid, block, shared_mem_size, stream>>>(                                                 \
    out_ptr,                                                                                  \
    query_ptr,                                                                                \
    key_cache_ptr,                                                                            \
    value_cache_ptr,                                                                          \
    scale,                                                                                    \
    block_tables_ptr,                                                                         \
    context_lens_ptr,                                                                         \
    max_num_blocks_per_seq,                                                                   \
    query_stride);

// TODO(woosuk)：调整 NUM_THREADS。
template<
  typename T,
  int BLOCK_SIZE,
  int NUM_THREADS = 128>
void single_query_cached_kv_attention_launcher(
  torch::Tensor& out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int max_context_len) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int query_stride = query.stride(0);

  int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  assert(head_size % thread_group_size == 0);

  T* out_ptr = reinterpret_cast<T*>(out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  T* key_cache_ptr = reinterpret_cast<T*>(key_cache.data_ptr());
  T* value_cache_ptr = reinterpret_cast<T*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* context_lens_ptr = context_lens.data_ptr<int>();

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_context_len = ((max_context_len + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
  int logits_size = padded_max_context_len * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
  int shared_mem_size = std::max(logits_size, outputs_size);

  dim3 grid(num_heads, num_seqs);
  dim3 block(NUM_THREADS);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  switch (head_size) {
    // 注意(woosuk)：为了减少编译时间，我们省略了头尺寸
    // 32, 160, 192, 256。
    // case 32:
    //   LAUNCH_ATTENTION_KERNEL(T, 32, BLOCK_SIZE, NUM_THREADS);
    //   break;
    case 64:
      LAUNCH_ATTENTION_KERNEL(T, 64, BLOCK_SIZE, NUM_THREADS);
      break;
    case 80:
      LAUNCH_ATTENTION_KERNEL(T, 80, BLOCK_SIZE, NUM_THREADS);
      break;
    case 96:
      LAUNCH_ATTENTION_KERNEL(T, 96, BLOCK_SIZE, NUM_THREADS);
      break;
    case 128:
      LAUNCH_ATTENTION_KERNEL(T, 128, BLOCK_SIZE, NUM_THREADS);
      break;
    // case 160:
    //   LAUNCH_ATTENTION_KERNEL(T, 160, BLOCK_SIZE, NUM_THREADS);
    //   break;
    // case 192:
    //   LAUNCH_ATTENTION_KERNEL(T, 192, BLOCK_SIZE, NUM_THREADS);
    //   break;
    // case 256:
    //   LAUNCH_ATTENTION_KERNEL(T, 256, BLOCK_SIZE, NUM_THREADS);
    //   break;
    default:
      TORCH_CHECK(false, "不支持的头尺寸: ", head_size);
      break;
  }
}

#define CALL_KERNEL_LAUNCHER(T, BLOCK_SIZE)                         \
  single_query_cached_kv_attention_launcher<T, BLOCK_SIZE>(         \
    out,                                                            \
    query,                                                          \
    key_cache,                                                      \
    value_cache,                                                    \
    scale,                                                          \
    block_tables,                                                   \
    context_lens,                                                   \
    max_context_len);

// 注意(woosuk)：为了减少编译时间，我们省略了块大小
// 1, 2, 4, 64, 128, 256。
#define CALL_KERNEL_LAUNCHER_BLOCK_SIZE(T)                          \
  switch (block_size) {                                             \
    /* case 1:                         */                           \
    /*   CALL_KERNEL_LAUNCHER(T, 1);   */                           \
    /*   break;                        */                           \
    /* case 2:                         */                           \
    /*   CALL_KERNEL_LAUNCHER(T, 2);   */                           \
    /*   break;                        */                           \
    /* case 4:                         */                           \
    /*   CALL_KERNEL_LAUNCHER(T, 4);   */                           \
    /*   break;                        */                           \
    case 8:                                                         \
      CALL_KERNEL_LAUNCHER(T, 8);                                   \
      break;                                                        \
    case 16:                                                        \
      CALL_KERNEL_LAUNCHER(T, 16);                                  \
      break;                                                        \
    case 32:                                                        \
      CALL_KERNEL_LAUNCHER(T, 32);                                  \
      break;                                                        \
    /* case 64:                        */                           \
    /*   CALL_KERNEL_LAUNCHER(T, 64);  */                           \
    /*   break;                        */                           \
    /* case 128:                       */                           \
    /*   CALL_KERNEL_LAUNCHER(T, 128); */                           \
    /*   break;                        */                           \
    /* case 256:                       */                           \
    /*   CALL_KERNEL_LAUNCHER(T, 256); */                           \
    /*   break;                        */                           \
    default:                                                        \
      TORCH_CHECK(false, "不支持的块大小: ", block_size);   \
      break;                                                        \
  }

void single_query_cached_kv_attention(
  torch::Tensor& out,             // [num_seqs, num_heads, head_size]
  torch::Tensor& query,           // [num_seqs, num_heads, head_size]
  torch::Tensor& key_cache,       // [num_blocks, num_heads, head_size/x, block_size, x]
  torch::Tensor& value_cache,     // [num_blocks, num_heads, head_size, block_size]
  float scale,
  torch::Tensor& block_tables,    // [num_seqs, max_num_blocks_per_seq]
  torch::Tensor& context_lens,    // [num_seqs]
  int block_size,
  int max_context_len) {
  if (query.dtype() == at::ScalarType::Float) {
    CALL_KERNEL_LAUNCHER_BLOCK_SIZE(float);
  } else if (query.dtype() == at::ScalarType::Half) {
    CALL_KERNEL_LAUNCHER_BLOCK_SIZE(uint16_t);
  } else if (query.dtype() == at::ScalarType::BFloat16) {
    CALL_KERNEL_LAUNCHER_BLOCK_SIZE(__nv_bfloat16);
  } else {
    TORCH_CHECK(false, "不支持的数据类型: ", query.dtype());
  }
}

#undef WARP_SIZE
#undef MAX
#undef MIN
