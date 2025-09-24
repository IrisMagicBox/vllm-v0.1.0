# Copyright 2023 The vLLM team.
# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/utils.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch
from typing import List, Sequence

def ensure_divisibility(numerator, denominator):
    """确保分子能被分母整除。"""
    assert numerator % denominator == 0, "{} 不能被 {} 整除".format(
        numerator, denominator
    )


def divide(numerator, denominator):
    """确保分子能被分母整除并返回
    除法值。"""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    """沿最后一个维度分割张量。

        参数:
            tensor: 输入张量。
            num_partitions: 分割张量的分区数
            contiguous_split_chunks: 如果为 True，在内存中使每个块连续。

        返回:
            张量列表
    """
    # 获取大小和维度。
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # 分割。
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # 注意：torch.split 默认不会创建连续张量。
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class VocabUtility:
    """将词汇表分割成 `world_size` 块，并返回属于 `rank` 分区的
        词汇表的第一个和最后一个索引：注意索引在 [first, last) 中

    """

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size: int, rank, world_size: int
    ) -> Sequence[int]:
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size: int, rank: int, world_size: int) -> Sequence[int]:
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank, world_size
        )
