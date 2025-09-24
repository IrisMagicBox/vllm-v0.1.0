/*
 * 改编自 https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention_utils.h
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

#include <stdint.h>

namespace vllm {

// 用于存储 Q、K、V 元素的向量类型。
template<typename T, int VEC_SIZE>
struct Vec {};

// 用于存储 FP32 累加器的向量类型。
template<typename T>
struct FloatVec {};

// 模板向量操作。
template<typename Acc, typename A, typename B>
inline __device__ Acc mul(A a, B b);

template<typename T>
inline __device__ float sum(T v);

template<typename T>
inline __device__ float dot(T a, T b) {
  return sum(mul<T, T, T>(a, b));
}

template<typename A, typename T>
inline __device__ float dot(T a, T b) {
  return sum(mul<A, T, T>(a, b));
}

template<typename T>
inline __device__ void zero(T& dst) {
  constexpr int WORDS = sizeof(T) / 4;
  union {
    T raw;
    uint32_t words[WORDS];
  } tmp;

#pragma unroll
  for (int ii = 0; ii < WORDS; ++ii) {
    tmp.words[ii] = 0u;
  }
  dst = tmp.raw;
}

} // namespace vllm
