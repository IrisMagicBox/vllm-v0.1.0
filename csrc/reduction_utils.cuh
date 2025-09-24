/*
 * 改编自 https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/reduce_kernel_utils.cuh
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

namespace vllm {

template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(0xffffffff, val, mask, 32);
  return val;
}

/* 计算块中所有元素的总和 */
template<typename T>
__inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum<T>(val);

  if (lane == 0)
    shared[wid] = val;

  __syncthreads();

  // 从 blockDim.x << 5 修改为 blockDim.x / 32，以防止
  // blockDim.x 不能被 32 整除
  val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : (T)(0.0f);
  val = warpReduceSum<T>(val);
  return val;
}

} // namespace vllm
