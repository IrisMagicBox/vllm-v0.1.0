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
#pragma once

#include "attention_dtypes.h"

#include <float.h>
#include <type_traits>

namespace vllm {

// Q*K^T 运算。
template<int THREAD_GROUP_SIZE, typename Vec, int N>
inline __device__ float qk_dot_(const Vec (&q)[N], const Vec (&k)[N]) {
  using A_vec = typename FloatVec<Vec>::Type;
  // 计算 Q*K^T 的并行乘积（分别处理向量通道）。
  A_vec qk_vec = mul<A_vec, Vec, Vec>(q[0], k[0]);
#pragma unroll
  for (int ii = 1; ii < N; ++ii) {
    qk_vec = fma(q[ii], k[ii], qk_vec);
  }

  // 完成跨通道的归约。
  float qk = sum(qk_vec);
#pragma unroll
  for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
    qk += __shfl_xor_sync(uint32_t(-1), qk, mask);
  }
  return qk;
}

template<typename T, int THREAD_GROUP_SIZE>
struct Qk_dot {
  template<typename Vec, int N>
  static inline __device__ float dot(const Vec (&q)[N], const Vec (&k)[N]) {
    return qk_dot_<THREAD_GROUP_SIZE>(q, k);
  }
};

} // namespace vllm
