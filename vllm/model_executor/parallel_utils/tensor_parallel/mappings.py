# Copyright 2023 The vLLM team.
# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/mappings.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch

from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
)
from .utils import split_tensor_along_last_dim


def _reduce(input_):
    """在模型并行组中对输入张量进行全规约操作。"""

    # 如果只使用1个GPU，则跳过此函数。
    if get_tensor_model_parallel_world_size()==1:
        return input_

    # 全规约。
    torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())

    return input_


def _split_along_last_dim(input_):
    """沿最后一个维度拆分张量并保留相应切片。"""

    world_size = get_tensor_model_parallel_world_size()
    # 如果只使用1个GPU，则跳过此函数。
    if world_size == 1:
        return input_

    # 沿最后一个维度拆分。
    input_list = split_tensor_along_last_dim(input_, world_size)

    # 注意：torch.split默认不创建连续的张量。
    rank = get_tensor_model_parallel_rank()
    output = input_list[rank].contiguous()

    return output


def _split_along_first_dim(input_):
    """沿第一个维度拆分张量并保留相应切片。"""

    world_size = get_tensor_model_parallel_world_size()
    # 如果只使用1个GPU，则跳过此函数。
    if world_size == 1:
        return input_

    # 沿第一个维度拆分。
    dim_size = input_.size()[0]
    assert dim_size % world_size == 0, \
        "张量的第一个维度应能被张量并行大小整除"
    local_dim_size = dim_size // world_size
    rank = get_tensor_model_parallel_rank()
    dim_offset = rank * local_dim_size

    output = input_[dim_offset:dim_offset+local_dim_size].contiguous()

    return output


def _gather_along_last_dim(input_):
    """沿最后一个维度收集张量并进行拼接。"""

    world_size = get_tensor_model_parallel_world_size()
    # 如果只使用1个GPU，则跳过此函数。
    if world_size == 1:
        return input_

    # 大小和维度。
    last_dim = input_.dim() - 1
    rank = get_tensor_model_parallel_rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=get_tensor_model_parallel_group())

    # 注意：torch.cat已经创建了一个连续的张量。
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output


def _gather_along_first_dim(input_):
    """沿第一个维度收集张量并进行拼接。"""

    world_size = get_tensor_model_parallel_world_size()
    # 如果只使用1个GPU，则跳过此函数。
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype,
                         device=torch.cuda.current_device())
    torch.distributed._all_gather_base(output, input_.contiguous(),
                                       group=get_tensor_model_parallel_group())

    return output

def _reduce_scatter_along_first_dim(input_):
    """在模型并行组中对输入张量进行规约-分散操作。"""
    world_size = get_tensor_model_parallel_world_size()
    # 如果只使用1个GPU，则跳过此函数。
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    assert dim_size[0] % world_size == 0, \
        "张量的第一个维度应能被张量并行大小整除"

    dim_size[0] = dim_size[0] // world_size

    output = torch.empty(dim_size, dtype=input_.dtype,
                         device=torch.cuda.current_device())
    torch.distributed._reduce_scatter_base(output, input_.contiguous(),
                                           group=get_tensor_model_parallel_group())
    return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    """将输入传递到模型并行区域。"""

    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """对来自模型并行区域的输入进行全规约操作。"""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """拆分输入并只保留对应rank的数据块。"""

    @staticmethod
    def symbolic(graph, input_):
        return _split_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_last_dim(grad_output)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """从模型并行区域收集输入并进行拼接。"""

    @staticmethod
    def symbolic(graph, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split_along_last_dim(grad_output)


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """拆分输入并只保留对应rank的数据块。"""

    @staticmethod
    def symbolic(graph, input_):
        return _split_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """从序列并行区域收集输入并进行拼接。"""

    @staticmethod
    def symbolic(graph, input_, tensor_parallel_output_grad=True):
        return _gather_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_, tensor_parallel_output_grad=True):
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        return _gather_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        tensor_parallel_output_grad = ctx.tensor_parallel_output_grad

        # 如果收集操作后的计算图处于张量并行模式，
        # 输出梯度需要进行规约-分散操作；
        # 而如果计算是重复的，则输出梯度需要被分散。
        if tensor_parallel_output_grad:
            return _reduce_scatter_along_first_dim(grad_output), None
        else:
            return _split_along_first_dim(grad_output), None


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """对来自模型并行区域的输入进行规约-分散操作。"""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce_scatter_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce_scatter_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)


# -----------------
# 辅助函数。
# -----------------

def copy_to_tensor_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_tensor_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_tensor_model_parallel_region(input_):
    return _ScatterToModelParallelRegion.apply(input_)


def gather_from_tensor_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)


def scatter_to_sequence_parallel_region(input_):
    return _ScatterToSequenceParallelRegion.apply(input_)


def gather_from_sequence_parallel_region(input_, tensor_parallel_output_grad=True):
    return _GatherFromSequenceParallelRegion.apply(input_, tensor_parallel_output_grad)


def reduce_scatter_to_sequence_parallel_region(input_):
    return _ReduceScatterToSequenceParallelRegion.apply(input_)

