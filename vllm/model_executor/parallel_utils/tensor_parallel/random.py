# Copyright 2023 The vLLM team.
# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/random.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

# 部分代码改编自 PyTorch
# repo: https://github.com/pytorch/pytorch

import contextlib

import torch
from torch import _C
from torch.cuda import _lazy_call, device as device_ctx_manager

from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank,
)

# 模型并行随机数生成器跟踪器的默认名称。
_MODEL_PARALLEL_RNG_TRACKER_NAME = 'model-parallel-rng'


def _set_cuda_rng_state(new_state, device=-1):
    """设置当前 GPU 的随机数生成器状态。

    参数:
        new_state (torch.ByteTensor): 期望的状态
    此函数改编自 PyTorch 仓库 (torch.cuda.set_rng_state)
    有一个变化：输入状态不被克隆。克隆在 +4 GPU 情况下导致
    严重的性能问题。
    """
    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):
        # 较老的 PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)
    else:
        # 较新的 PyTorch
        if device == -1:
            device = torch.device('cuda')
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device('cuda', device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.cuda.default_generators[idx]
            default_generator.set_state(new_state)

    _lazy_call(cb)



class CudaRNGStatesTracker:
    """cuda RNG 状态的跟踪器。

    使用 `add` 方法，基于输入的 `seed` 初始化 cuda rng 状态
    并将其分配给 `name`。之后，通过 fork rng 状态，
    我们可以执行操作并返回到我们开始的 cuda 状态。
    """

    def __init__(self):
        # 从字符串名称到 cuda rng 状态的映射。
        self.states_ = {}
        # 种子仅用于记录，确保没有种子被设置两次。
        self.seeds_ = set()

    def reset(self):
        """设置为初始状态（无跟踪器）。"""
        self.states_ = {}
        self.seeds_ = set()

    def get_states(self):
        """获取 rng 状态。复制字典，以便我们有直接
        指向状态的指针，而不仅仅是字典的指针。"""
        states = {}
        for name in self.states_:
            states[name] = self.states_[name]
        return states

    def set_states(self, states):
        """设置 rng 状态。出于效率目的，我们不检查
        种子大小的兼容性。"""
        self.states_ = states

    def add(self, name, seed):
        """跟踪 rng 状态。"""
        # 检查种子是否已被使用。
        if seed in self.seeds_:
            raise Exception('种子 {} 已存在'.format(seed))
        self.seeds_.add(seed)
        # 检查状态是否已定义。
        if name in self.states_:
            raise Exception('cuda rng 状态 {} 已存在'.format(name))
        # 获取当前 rng 状态。
        orig_rng_state = torch.cuda.get_rng_state()
        # 设置新状态并存储它。
        torch.cuda.manual_seed(seed)
        self.states_[name] = torch.cuda.get_rng_state()
        # 将 rng 状态重置为之前的状态。
        _set_cuda_rng_state(orig_rng_state)

    @contextlib.contextmanager
    def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
        """Fork cuda rng 状态，执行操作，并以
        原始状态退出。"""
        # 检查我们是否已添加状态
        if name not in self.states_:
            raise Exception('cuda rng 状态 {} 未添加'.format(name))
        # 存储当前 rng 状态。
        orig_cuda_rng_state = torch.cuda.get_rng_state()
        # 将 rng 状态设置为所需的
        _set_cuda_rng_state(self.states_[name])
        # 执行我们想做的操作。
        try:
            yield
        finally:
            # 更新当前 rng 状态以供后续使用。
            self.states_[name] = torch.cuda.get_rng_state()
            # 并将状态设置为我们开始时的原始状态。
            _set_cuda_rng_state(orig_cuda_rng_state)


# RNG 跟踪器对象。
_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()


def get_cuda_rng_tracker():
    """获取 cuda rng 跟踪器。"""
    return _CUDA_RNG_STATE_TRACKER


def model_parallel_cuda_manual_seed(seed):
    """初始化模型并行 cuda 种子。

    此函数应在模型并行初始化后调用。
    此外，此函数之后不应调用 torch.cuda.manual_seed。
    基本上，这是该函数的替代品。
    跟踪两组 RNG 状态：
        默认状态：这是用于数据并行的，在一组模型并行 GPU 中相同，
                  但在不同的模型并行组之间不同。这用于
                  例如非张量模型并行区域中的 dropout。
        张量模型并行状态：此状态在一组模型并行 GPU 中不同，
                          但在数据并行组中相同。
                          这用于例如模型并行区域中的 dropout。
    """
    # 2718 只是为了好玩，任何正数值都可以。
    offset = seed + 2718
    tensor_model_parallel_seed = offset + get_tensor_model_parallel_rank()
    # 数据并行获取原始种子。
    data_parallel_seed = seed

    _CUDA_RNG_STATE_TRACKER.reset()
    # 设置默认状态。
    torch.cuda.manual_seed(data_parallel_seed)
    # 和模型并行状态。
    _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME,
                                tensor_model_parallel_seed)
