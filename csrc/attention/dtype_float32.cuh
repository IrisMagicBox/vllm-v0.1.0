/*
 * 改编自 https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * 和 https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention_utils.h
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

#include "attention_generic.cuh"

#include <stdint.h>

namespace vllm {

// 定义自定义 FP32 向量数据类型。
struct Float4_ {
  float2 x;
  float2 y;
};

struct Float8_ {
  float2 x;
  float2 y;
  float2 z;
  float2 w;
};

// 用于 Q, K, V 的 FP32 向量类型。
template<>
struct Vec<float, 1> {
  using Type = float;
};
template<>
struct Vec<float, 2> {
  using Type = float2;
};
template<>
struct Vec<float, 4> {
  using Type = float4;
};

// 对应 Vec 的 FP32 累加器向量类型。
template<>
struct FloatVec<float> {
  using Type = float;
};
template<>
struct FloatVec<float2> {
  using Type = float2;
};
template<>
struct FloatVec<float4> {
  using Type = float4;
};

// 向量加法。
inline __device__ float add(float a, float b) {
  return a + b;
}

inline __device__ float2 add(float2 a, float2 b) {
  float2 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  return c;
}

inline __device__ float4 add(float4 a, float4 b) {
  float4 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  c.z = add(a.z, b.z);
  c.w = add(a.w, b.w);
  return c;
}

// 向量乘法。
template<>
inline __device__ float mul<float, float>(float a, float b) {
  return a * b;
}

template<>
inline __device__ float2 mul(float2 a, float2 b) {
  float2 c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  return c;
}

template<>
inline __device__ float2 mul(float a, float2 b) {
  float2 c;
  c.x = a * b.x;
  c.y = a * b.y;
  return c;
}

template<>
inline __device__ float4 mul(float4 a, float4 b) {
  float4 c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  c.z = a.z * b.z;
  c.w = a.w * b.w;
  return c;
}

template<>
inline __device__ float4 mul(float a, float4 b) {
  float4 c;
  c.x = a * b.x;
  c.y = a * b.y;
  c.z = a * b.z;
  c.w = a * b.w;
  return c;
}

// 向量融合乘加运算。
inline __device__ float fma(float a, float b, float c) {
  return a * b + c;
}

inline __device__ float2 fma(float2 a, float2 b, float2 c) {
  float2 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  return d;
}

inline __device__ float2 fma(float a, float2 b, float2 c) {
  float2 d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  return d;
}

inline __device__ float4 fma(float4 a, float4 b, float4 c) {
  float4 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  d.z = fma(a.z, b.z, c.z);
  d.w = fma(a.w, b.w, c.w);
  return d;
}

inline __device__ float4 fma(float a, float4 b, float4 c) {
  float4 d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  d.z = fma(a, b.z, c.z);
  d.w = fma(a, b.w, c.w);
  return d;
}

inline __device__ Float4_ fma(float a, Float4_ b, Float4_ c) {
  Float4_ d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  return d;
}

inline __device__ Float8_ fma(float a, Float8_ b, Float8_ c) {
  Float8_ d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  d.z = fma(a, b.z, c.z);
  d.w = fma(a, b.w, c.w);
  return d;
}

// 向量求和。
template<>
inline __device__ float sum(float v) {
  return v;
}

template<>
inline __device__ float sum(float2 v) {
  return v.x + v.y;
}

template<>
inline __device__ float sum(float4 v) {
  return v.x + v.y + v.z + v.w;
}

template<>
inline __device__ float sum(Float4_ v) {
  return v.x.x + v.x.y + v.y.x + v.y.y;
}

template<>
inline __device__ float sum(Float8_ v) {
  return v.x.x + v.x.y + v.y.x + v.y.y + v.z.x + v.z.y + v.w.x + v.w.y;
}

// 向量点积。
inline __device__ float dot(float a, float b) {
  return a * b;
}

inline __device__ float dot(float2 a, float2 b) {
  float2 c = mul<float2, float2, float2>(a, b);
  return c.x + c.y;
}

inline __device__ float dot(Float4_ a, Float4_ b) {
  float2 acc = mul<float2, float2, float2>(a.x, b.x);
  acc = fma(a.y, b.y, acc);
  return acc.x + acc.y;
}

inline __device__ float dot(Float8_ a, Float8_ b) {
  float2 acc = mul<float2, float2, float2>(a.x, b.x);
  acc = fma(a.y, b.y, acc);
  acc = fma(a.z, b.z, acc);
  acc = fma(a.w, b.w, acc);
  return acc.x + acc.y;
}

// 从 float 到 float。
inline __device__ void from_float(float& dst, float src) {
  dst = src;
}

inline __device__ void from_float(float2& dst, float2 src) {
  dst = src;
}

inline __device__ void from_float(float4& dst, float4 src) {
  dst = src;
}

// 从 float 到 float。
inline __device__ float to_float(float u) {
  return u;
}

inline __device__ float2 to_float(float2 u) {
  return u;
}

inline __device__ float4 to_float(float4 u) {
  return u;
}

inline __device__ Float4_ to_float(Float4_ u) {
  return u;
}

inline __device__ Float8_ to_float(Float8_ u) {
  return u;
}

} // namespace vllm
