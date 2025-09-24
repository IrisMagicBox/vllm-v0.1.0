# Copyright 2023 The vLLM team.
# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

# 代码的部分内容改编自 PyTorch
# repo: https://github.com/pytorch/pytorch


import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_all_reduce_launcher,
)
from .mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)

from .random import get_cuda_rng_tracker
from .utils import (
    divide,
    VocabUtility,
)

_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {'tensor_model_parallel': False,
                                      'partition_dim': -1,
                                      'partition_stride': 1}

def param_is_not_tensor_parallel_duplicate(param):
    return (hasattr(param, 'tensor_model_parallel') and
            param.tensor_model_parallel) or (
                get_tensor_model_parallel_rank() == 0)


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, 'tensor_model_parallel', is_parallel)
    setattr(tensor, 'partition_dim', dim)
    setattr(tensor, 'partition_stride', stride)


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):
    def maybe_set(attribute, value):
        if not hasattr(tensor, attribute):
            setattr(tensor, attribute, value)
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])


def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
    def maybe_copy(attribute):
        if hasattr(source_tensor, attribute):
            setattr(destination_tensor, attribute,
                    getattr(source_tensor, attribute))
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_copy(attribute)


def _initialize_affine_weight_gpu(weight, init_method,
                                  partition_dim, stride=1):
    """在 GPU 上为模型并行初始化仿射权重。"""

    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=True,
                                         dim=partition_dim,
                                         stride=stride)

    with get_cuda_rng_tracker().fork():
        init_method(weight)


def _initialize_affine_weight_cpu(weight, output_size, input_size,
                                  per_partition_size, partition_dim,
                                  init_method, stride=1,
                                  return_master_weight=False,
                                  *, params_dtype=None):
    """为模型并行初始化仿射权重。

    在所有进程中构建主权重并分发相关块。"""

    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=True,
                                         dim=partition_dim,
                                         stride=stride)

    if params_dtype is None:
        params_dtype = torch.get_default_dtype()

    # 初始化主权重
    master_weight = torch.empty(output_size, input_size,
                                dtype=torch.float,
                                requires_grad=False)
    init_method(master_weight)
    master_weight = master_weight.to(dtype=params_dtype)

    # 分割并复制
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size,
                              dim=partition_dim)
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


class VocabParallelEmbedding(torch.nn.Module):
    """在词汇维度上并行化的嵌入层。

    这主要改编自 torch.nn.Embedding 并保留了所有默认值。
    参数：
        num_embeddings: 词汇表大小。
        embedding_dim: 隐藏状态的大小。

    关键字参数：
        init_method: 初始化权重的方法。
        params_dtype
        use_cpu_initialization
        perform_initialization
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, *,
                 init_method=init.xavier_normal_,
                 params_dtype: torch.dtype=None,
                 use_cpu_initialization: bool=False,
                 perform_initialization: bool=True):
        super(VocabParallelEmbedding, self).__init__()
        # 保留输入维度。
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        # 为兼容性设置默认值。
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        # 沿词汇维度分割权重矩阵。
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, get_tensor_model_parallel_rank(),
                self.tensor_model_parallel_size)
        self.num_embeddings_per_partition = self.vocab_end_index - \
            self.vocab_start_index

        # 分配权重并初始化。
        if use_cpu_initialization:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                dtype=params_dtype))
            if perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight, self.num_embeddings, self.embedding_dim,
                    self.num_embeddings_per_partition, 0, init_method,
                    params_dtype=params_dtype)
        else:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                device=torch.cuda.current_device(), dtype=params_dtype))
            if perform_initialization:
                _initialize_affine_weight_gpu(self.weight, init_method,
                                              partition_dim=0, stride=1)

    def forward(self, input_):
        if self.tensor_model_parallel_size > 1:
            # 构建掩码。
            input_mask = (input_ < self.vocab_start_index) | \
                         (input_ >= self.vocab_end_index)
            # 掩码输入。
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # 获取嵌入。
        output_parallel = F.embedding(masked_input, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        # 掩码输出嵌入。
        if self.tensor_model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # 在所有模型并行 GPU 上进行规约。
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        return output


class ColumnParallelLinear(torch.nn.Module):
    """具有列并行性的线性层。

    线性层定义为 Y = XA + b。A 沿其第二个维度并行化为 A = [A_1, ..., A_p]。

    参数：
        input_size: 矩阵 A 的第一个维度。
        output_size: 矩阵 A 的第二个维度。

    关键字参数
        bias: 如果为 True，则添加偏置
        gather_output: 如果为 True，则对输出调用 all-gather 并使 Y 对所有 GPU 可用，
                       否则，每个 GPU 都将拥有其输出，即 Y_i = XA_i
        init_method: 初始化权重的方法。注意偏置始终设置为零。
        stride: 用于步长线性层。
        keep_master_weight_for_test: 这是为了测试而添加的，应设置为 False。
                                     它返回用于初始化的主权重。
        skip_bias_add: 这是为了启用性能优化而添加的，其中偏置可以与其他元素级操作融合。
                       我们跳过添加偏置，而是返回它。
        params_dtype:
        use_cpu_initialization:
    """

    def __init__(self, input_size, output_size, *,
                 bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False,
                 params_dtype=None,
                 use_cpu_initialization=False,
                 perform_initialization=True,
                 ):
        super(ColumnParallelLinear, self).__init__()

        # 保留输入参数
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # 沿最后一个维度分割权重矩阵。
        world_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        # 参数。
        # 注意：torch.nn.functional.linear 执行 XA^T + b，因此我们分配转置。
        # 初始化权重。
        if use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size_per_partition,
                                                self.input_size,
                                                dtype=params_dtype))
            if perform_initialization:
                self.master_weight = _initialize_affine_weight_cpu(
                    self.weight, self.output_size, self.input_size,
                    self.output_size_per_partition, 0, init_method,
                    stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size_per_partition, self.input_size,
                device=torch.cuda.current_device(), dtype=params_dtype))
            if perform_initialization:
                _initialize_affine_weight_gpu(self.weight, init_method,
                                              partition_dim=0, stride=stride)

        if bias:
            if use_cpu_initialization:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition, dtype=params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=params_dtype))
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            # 始终将偏置初始化为零。
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)


    def forward(self, input_):
        """ColumnParallelLinear 的前向传播

        参数：
            input_: 3D 张量，其维度顺序为 [sequence, batch, hidden]

        返回：
            - output
            - bias
        """
        bias = self.bias if not self.skip_bias_add else None

        input_parallel = input_
        # 矩阵乘法。
        output_parallel = F.linear(input_parallel, self.weight, bias)
        if self.gather_output:
            # 在分区之间进行 all-gather。
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class RowParallelLinear(torch.nn.Module):
    """具有行并行性的线性层。

    线性层定义为 Y = XA + b。A 沿其第一个维度并行化，X 沿其第二个维度并行化如下：
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    参数：
        input_size: 矩阵 A 的第一个维度。
        output_size: 矩阵 A 的第二个维度。

    关键字参数：
        bias: 如果为 True，则添加偏置。注意偏置不并行化。
        input_is_parallel: 如果为 True，我们假设输入已经跨 GPU 分割，
                           我们不再分割。
        init_method: 初始化权重的方法。注意偏置始终设置为零。
        stride: 用于步长线性层。
        keep_master_weight_for_test: 这是为了测试而添加的，应设置为 False。
                                     它返回用于初始化的主权重。
        skip_bias_add: 这是为了启用性能优化而添加的，其中偏置可以与其他元素级操作融合。
                       我们跳过添加偏置，而是返回它。
        params_dtype:
        use_cpu_initialization:
        perform_initialization:
    """

    def __init__(self, input_size, output_size, *,
                 bias=True, input_is_parallel=False,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False,
                 params_dtype=None,
                 use_cpu_initialization=False,
                 perform_initialization=True,
                 ):
        super(RowParallelLinear, self).__init__()

        # 保留输入参数
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        # 沿最后一个维度分割权重矩阵。
        world_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add

        # 参数。
        # 注意：torch.nn.functional.linear 执行 XA^T + b，因此我们分配转置。
        # 初始化权重。
        if use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size,
                                                self.input_size_per_partition,
                                                dtype=params_dtype))
            if perform_initialization:
                self.master_weight = _initialize_affine_weight_cpu(
                    self.weight, self.output_size, self.input_size,
                    self.input_size_per_partition, 1, init_method,
                    stride=stride, return_master_weight=keep_master_weight_for_test,
                    params_dtype=params_dtype)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size, self.input_size_per_partition,
                device=torch.cuda.current_device(), dtype=params_dtype))
            if perform_initialization:
                _initialize_affine_weight_gpu(self.weight, init_method,
                                              partition_dim=1, stride=stride)
        if bias:
            if use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size,
                                                  dtype=params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size, device=torch.cuda.current_device(),
                    dtype=params_dtype))

            # 始终将偏置初始化为零。
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)
        self.weight_t = self.weight.t()

    def forward(self, input_):
        """RowParallelLinear 的前向传播

        参数：
            input_: 3D 张量，其维度顺序为 [sequence, batch, hidden]

        返回：
            - output
            - bias
        """
        # 设置反向传播 all-reduce。
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        if get_tensor_model_parallel_world_size() == 1:
            # 矩阵乘法。
            output_ = F.linear(input_parallel, self.weight)
        else:
            # 矩阵乘法。
            all_reduce_launcher = get_all_reduce_launcher()
            num_tokens = input_parallel.shape[0]
            output_buffer = all_reduce_launcher.buffer[:num_tokens]
            torch.matmul(input_parallel, self.weight_t, out=output_buffer)
            # 在所有分区之间进行 all-reduce。
            output_ = all_reduce_launcher.launch(output_buffer)

        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias
