"""多头注意力。"""
from typing import Optional

import torch
import torch.nn as nn
from xformers import ops as xops

from vllm import attention_ops
from vllm import cache_ops
from vllm import pos_encoding_ops
from vllm.model_executor.input_metadata import InputMetadata

_SUPPORTED_HEAD_SIZES = [64, 80, 96, 128]


class PagedAttention(nn.Module):
    """GPT风格的多头分页注意力。

    该类将扁平化的1D查询、键和值张量作为输入。输入的1D张量可以分为三个部分：
    提示标记、生成标记和填充。

    |<------------------------------------- num_valid_tokens ------------------------------------->|
    |<--------------- num_prompt_tokens -------------->|<------- num_generation_tokens (M) ------->|
    |<--prompt_0-->|<--prompt_1-->|...|<--prompt_N-1-->|<--generation_0-->|...|<--generation_M-1-->|<--padding-->|

    提示可能有不同的长度，而生成标记的长度始终为1。添加填充是为了使输入长度成为8的倍数，
    这对Tensor Cores是有利的。

    该类执行以下操作：
    1. 对提示执行multi_query_kv_attention。此操作不使用KV缓存。
    2. 等待缓存操作（例如，交换、复制）完成。缓存操作由缓存引擎在执行模型的前向传递之前
       发出，并异步执行。
    3. 重塑并将输入的键和值张量存储在KV缓存中。
    4. 对生成标记执行single_query_cached_kv_attention。
       此操作从KV缓存中读取之前的键和值张量。
    5. 输出扁平化的1D张量。
    """

    def __init__(self, num_heads: int, head_size: int, scale: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.attn_op = xops.fmha.cutlass.FwOp()

        if self.head_size not in _SUPPORTED_HEAD_SIZES:
            raise ValueError(f"不支持的头尺寸 ({self.head_size})。 "
                             f"支持的头尺寸：{_SUPPORTED_HEAD_SIZES}。")

    def multi_query_kv_attention(
        self,
        output: torch.Tensor,                   # [num_prompt_tokens, num_heads, head_size]
        query: torch.Tensor,                    # [num_prompt_tokens, num_heads, head_size]
        key: torch.Tensor,                      # [num_prompt_tokens, num_heads, head_size]
        value: torch.Tensor,                    # [num_prompt_tokens, num_heads, head_size]
        attn_bias: xops.AttentionBias,
    ) -> torch.Tensor:
        # TODO(woosuk): unsqueeze操作可能会带来一些CPU开销。优化。
        out = xops.memory_efficient_attention_forward(
            query.unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
            attn_bias=attn_bias,
            p=0.0,
            scale=self.scale,
            op=self.attn_op,
        )
        # TODO(woosuk): 无需复制。优化。
        output.copy_(out.squeeze(0))
        return output

    def single_query_cached_kv_attention(
        self,
        output: torch.Tensor,           # [num_generation_tokens, num_heads, head_size]
        query: torch.Tensor,            # [num_generation_tokens, num_heads, head_size]
        key_cache: torch.Tensor,        # [num_blocks, num_heads, head_size/x, block_size, x]
        value_cache: torch.Tensor,      # [num_blocks, num_heads, head_size, block_size]
        input_metadata: InputMetadata,
    ) -> None:
        block_size = value_cache.shape[3]
        attention_ops.single_query_cached_kv_attention(
            output,
            query,
            key_cache,
            value_cache,
            self.scale,
            input_metadata.block_tables,
            input_metadata.context_lens,
            block_size,
            input_metadata.max_context_len,
        )

    def forward(
        self,
        query: torch.Tensor,                    # [num_tokens, num_heads * head_size]
        key: torch.Tensor,                      # [num_tokens, num_heads * head_size]
        value: torch.Tensor,                    # [num_tokens, num_heads * head_size]
        key_cache: Optional[torch.Tensor],      # [num_blocks, num_heads, head_size/x, block_size, x]
        value_cache: Optional[torch.Tensor],    # [num_blocks, num_heads, head_size, block_size]
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:                          # [num_tokens, num_heads * head_size]
        # 注意：查询、键和值张量必须从形状为[num_tokens, 3 * num_heads * head_size]的qkv张量中切片。

        # 重塑查询、键和值张量。
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_heads, self.head_size)
        value = value.view(-1, self.num_heads, self.head_size)

        # 预分配输出张量。
        output = torch.empty_like(query)

        # 计算提示的注意力操作。
        num_prompt_tokens = input_metadata.num_prompt_tokens
        if num_prompt_tokens > 0:
            self.multi_query_kv_attention(
                output[:num_prompt_tokens],
                query[:num_prompt_tokens],
                key[:num_prompt_tokens],
                value[:num_prompt_tokens],
                input_metadata.attn_bias,
            )

        # 等待缓存操作完成。
        if cache_event is not None:
            cache_event.wait()

        # 重塑键和值并将它们存储在缓存中。
        # 当未提供key_cache和value_cache时，新的键和值向量不会被缓存。
        num_valid_tokens = input_metadata.num_valid_tokens
        if (num_valid_tokens > 0 and key_cache is not None
            and value_cache is not None):
            # 步长为3，因为键和值是从qkv切片的。
            cache_ops.reshape_and_cache(
                key[:num_valid_tokens],
                value[:num_valid_tokens],
                key_cache,
                value_cache,
                input_metadata.slot_mapping,
            )

        if input_metadata.num_generation_tokens > 0:
            assert key_cache is not None and value_cache is not None, (
                "生成标记时必须提供key_cache和value_cache。"
            )
            # 计算生成标记的注意力操作。
            self.single_query_cached_kv_attention(
                output[num_prompt_tokens:num_valid_tokens],
                query[num_prompt_tokens:num_valid_tokens],
                key_cache,
                value_cache,
                input_metadata)

        # 重塑输出张量。
        # 注意(woosuk)：输出张量可能包含填充。
        return output.view(-1, self.num_heads * self.head_size)


class PagedAttentionWithRoPE(PagedAttention):
    """带有GPT-NeoX风格旋转嵌入的分页注意力。"""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        rotary_dim: int,
        max_position: int = 8192,
        base: int = 10000,
    ) -> None:
        super().__init__(num_heads, head_size, scale)

        # 创建cos和sin缓存。
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))
        t = torch.arange(max_position).float()
        freqs = torch.einsum('i,j -> ij', t, inv_freq.float())
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)

        # FIXME(woosuk)：这假设我们在初始化模型时配置默认的dtype。
        # 使其更加健壮。
        torch_dtype = torch.get_default_dtype()
        cache = cache.to(torch_dtype)
        # 嵌入大小：[max_position, rotary_dim]
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,                # [num_tokens]
        query: torch.Tensor,                    # [num_tokens, num_heads * head_size]
        key: torch.Tensor,                      # [num_tokens, num_heads * head_size]
        value: torch.Tensor,                    # [num_tokens, num_heads * head_size]
        key_cache: torch.Tensor,                # [num_blocks, num_heads, head_size/x, block_size, x]
        value_cache: torch.Tensor,              # [num_blocks, num_heads, head_size, block_size]
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:                          # [num_tokens, num_heads * head_size]
        # 在将查询和键传递给注意力操作之前应用旋转嵌入。
        pos_encoding_ops.rotary_embedding_neox(
            positions,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
        )
        return super().forward(
            query,
            key,
            value,
            key_cache,
            value_cache,
            input_metadata,
            cache_event,
        )
