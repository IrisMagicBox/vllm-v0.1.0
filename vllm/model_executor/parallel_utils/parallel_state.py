# Copyright 2023 The vLLM team.
# Adapted from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""模型和数据并行组。"""

import torch
from typing import Optional

# 当前排名所属的层内模型并行组。
_TENSOR_MODEL_PARALLEL_GROUP = None
# 当前排名所属的层间模型并行组。
_PIPELINE_MODEL_PARALLEL_GROUP = None
# 当前排名所属的模型并行组（包括层内和流水线）。
_MODEL_PARALLEL_GROUP = None
# 嵌入组。
_EMBEDDING_GROUP = None
# 位置嵌入组。
_POSITION_EMBEDDING_GROUP = None
# 当前排名所属的数据并行组。
_DATA_PARALLEL_GROUP = None

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_PIPELINE_MODEL_PARALLEL_SPLIT_RANK = None

# 这些值使我们能够动态更改 mpu 大小。
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None

# 拥有嵌入副本的排名列表。
_EMBEDDING_GLOBAL_RANKS = None

# 拥有位置嵌入副本的排名列表。
_POSITION_EMBEDDING_GLOBAL_RANKS = None

# 每个流水线组的全局排名列表，便于在从第一个或最后一个流水线阶段广播时计算源排名。
_PIPELINE_GLOBAL_RANKS = None

# 每个数据并行组的全局排名列表，便于在从源到所有其他数据并行排名广播权重时计算源排名。
_DATA_PARALLEL_GLOBAL_RANKS = None

_ALL_REDUCE_LAUNCHER: Optional['GraphAllReduce'] = None

def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
) -> None:
    """
    初始化模型数据并行组。

    参数:
        tensor_model_parallel_size: 用于张量模型并行的 GPU 数量。
        pipeline_model_parallel_size: 用于流水线模型并行的 GPU 数量。
        virtual_pipeline_model_parallel_size: 虚拟阶段数（交错流水线）。
        pipeline_model_parallel_split_rank: 对于同时具有编码器和解码器的模型，
                                            流水线中分割点的排名。

    假设我们总共有16个GPU，用g0...g15表示，我们使用2个GPU来并行化模型张量，
    使用4个GPU来并行化模型流水线。此函数将创建8个张量模型并行组，
    4个流水线模型并行组和8个数据并行组：
        8个数据并行组：
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8个张量模型并行组：
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4个流水线模型并行组：
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    为提高效率，请注意调用者应确保相邻排名位于同一DGX服务器上。
    例如，如果我们使用2个DGX-1服务器，总共16个GPU，排名0到7属于第一个服务器，
    排名8到15属于第二个服务器。
    """
    # 获取世界大小和排名。确保一些一致性。
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    if world_size % (tensor_model_parallel_size * pipeline_model_parallel_size) != 0:
        raise RuntimeError(
            f"world_size ({world_size}) 不能被 tensor_model_parallel_size "
            f"({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size}) 整除"
        )

    data_parallel_size: int = world_size // (tensor_model_parallel_size *
                                             pipeline_model_parallel_size)

    num_tensor_model_parallel_groups: int  = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    num_data_parallel_groups: int = world_size // data_parallel_size

    if virtual_pipeline_model_parallel_size is not None:
        if not pipeline_model_parallel_size > 2:
            raise RuntimeError("流水线模型并行大小在交错调度时应大于2")
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

    if pipeline_model_parallel_split_rank is not None:
        global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
        _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

    rank = torch.distributed.get_rank()

    # 构建数据并行组。
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GLOBAL_RANKS
    assert _DATA_PARALLEL_GROUP is None, '数据并行组已初始化'
    all_data_parallel_group_ranks = []
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups
        for j in range(tensor_model_parallel_size):
            ranks = range(start_rank + j, end_rank, tensor_model_parallel_size)
            all_data_parallel_group_ranks.append(list(ranks))
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _DATA_PARALLEL_GROUP = group
                _DATA_PARALLEL_GLOBAL_RANKS = ranks

    # 构建模型并行组。
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, '模型并行组已初始化'
    for i in range(data_parallel_size):
        ranks = [data_parallel_group_ranks[i]
                 for data_parallel_group_ranks in all_data_parallel_group_ranks]
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group

    # 构建张量模型并行组。
    global _TENSOR_MODEL_PARALLEL_GROUP
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, \
        '张量模型并行组已初始化'
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size,
                      (i + 1) * tensor_model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group

    # 构建流水线模型并行组和嵌入组
    # （每个流水线模型并行组的第一个和最后一个排名）。
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert _PIPELINE_MODEL_PARALLEL_GROUP is None, \
        '流水线模型并行组已初始化'
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    assert _EMBEDDING_GROUP is None, '嵌入组已初始化'
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    assert _POSITION_EMBEDDING_GROUP is None, \
        '位置嵌入组已初始化'
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PIPELINE_GLOBAL_RANKS = ranks
        # 设置嵌入组（在第一阶段和最后阶段之间交换梯度）。
        if len(ranks) > 1:
            embedding_ranks = [ranks[0], ranks[-1]]
            position_embedding_ranks = [ranks[0]]
            if pipeline_model_parallel_split_rank is not None:
                if ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                    embedding_ranks = [ranks[0],
                                       ranks[pipeline_model_parallel_split_rank],
                                       ranks[-1]]
                if ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                    position_embedding_ranks = [ranks[0],
                                       ranks[pipeline_model_parallel_split_rank]]
        else:
            embedding_ranks = ranks
            position_embedding_ranks = ranks

        group = torch.distributed.new_group(embedding_ranks)
        if rank in embedding_ranks:
            _EMBEDDING_GROUP = group
        if rank in ranks:
            _EMBEDDING_GLOBAL_RANKS = embedding_ranks

        group = torch.distributed.new_group(position_embedding_ranks)
        if rank in position_embedding_ranks:
            _POSITION_EMBEDDING_GROUP = group
        if rank in ranks:
            _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks

def initialize_all_reduce_launcher(
    max_num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    disable_graph: bool = False,
) -> None:
    global _ALL_REDUCE_LAUNCHER
    _ALL_REDUCE_LAUNCHER = GraphAllReduce(
        max_num_tokens=max_num_tokens,
        hidden_size=hidden_size,
        dtype=dtype,
        disable_graph=disable_graph,
    )

def model_parallel_is_initialized():
    """检查模型和数据并行组是否已初始化。"""
    if _TENSOR_MODEL_PARALLEL_GROUP is None or \
        _PIPELINE_MODEL_PARALLEL_GROUP is None or \
        _DATA_PARALLEL_GROUP is None:
        return False
    return True


def get_model_parallel_group():
    """获取调用者排名所属的模型并行组。"""
    assert _MODEL_PARALLEL_GROUP is not None, \
        '模型并行组未初始化'
    return _MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_group():
    """获取调用者排名所属的张量模型并行组。"""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, \
        '层内模型并行组未初始化'
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group():
    """获取调用者排名所属的流水线模型并行组。"""
    assert _PIPELINE_MODEL_PARALLEL_GROUP is not None, \
        '流水线模型并行组未初始化'
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_data_parallel_group():
    """获取调用者排名所属的数据并行组。"""
    assert _DATA_PARALLEL_GROUP is not None, \
        '数据并行组未初始化'
    return _DATA_PARALLEL_GROUP


def get_embedding_group():
    """获取调用者排名所属的嵌入组。"""
    assert _EMBEDDING_GROUP is not None, \
        '嵌入组未初始化'
    return _EMBEDDING_GROUP


def get_position_embedding_group():
    """获取调用者排名所属的位置嵌入组。"""
    assert _POSITION_EMBEDDING_GROUP is not None, \
        '位置嵌入组未初始化'
    return _POSITION_EMBEDDING_GROUP


def set_tensor_model_parallel_world_size(world_size):
    """设置张量模型并行大小"""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_pipeline_model_parallel_world_size(world_size):
    """设置流水线模型并行大小"""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_tensor_model_parallel_world_size():
    """返回张量模型并行组的世界大小。"""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_world_size():
    """返回流水线模型并行组的世界大小。"""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_pipeline_model_parallel_group())


def set_tensor_model_parallel_rank(rank):
    """设置张量模型并行排名。"""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_rank(rank):
    """设置流水线模型并行排名。"""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_split_rank(rank):
    """设置流水线模型并行分割排名。"""
    global _MPU_PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_SPLIT_RANK = rank


def get_tensor_model_parallel_rank():
    """返回张量模型并行组的我的排名。"""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_rank():
    """返回流水线模型并行组的我的排名。"""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    if _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())


def is_pipeline_first_stage(ignore_virtual=False):
    """如果在第一个流水线模型并行阶段，则返回True，否则返回False。"""
    if not ignore_virtual:
        if get_virtual_pipeline_model_parallel_world_size() is not None and \
            get_virtual_pipeline_model_parallel_rank() != 0:
            return False
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_last_stage(ignore_virtual=False):
    """如果在最后一个流水线模型并行阶段，则返回True，否则返回False。"""
    if not ignore_virtual:
        virtual_pipeline_model_parallel_world_size = \
            get_virtual_pipeline_model_parallel_world_size()
        if virtual_pipeline_model_parallel_world_size is not None and \
            get_virtual_pipeline_model_parallel_rank() != (
                virtual_pipeline_model_parallel_world_size - 1):
            return False
    return get_pipeline_model_parallel_rank() == (
        get_pipeline_model_parallel_world_size() - 1)


def is_rank_in_embedding_group(ignore_virtual=False):
    """如果当前排名在嵌入组中，则返回true，否则返回False。"""
    rank = torch.distributed.get_rank()
    global _EMBEDDING_GLOBAL_RANKS
    if ignore_virtual:
        return rank in _EMBEDDING_GLOBAL_RANKS
    if rank in _EMBEDDING_GLOBAL_RANKS:
        if rank == _EMBEDDING_GLOBAL_RANKS[0]:
            return is_pipeline_first_stage(ignore_virtual=False)
        elif rank == _EMBEDDING_GLOBAL_RANKS[-1]:
            return is_pipeline_last_stage(ignore_virtual=False)
        else:
            return True
    return False


def is_rank_in_position_embedding_group():
    """如果当前排名在位置嵌入组中，则返回true，否则返回False。"""
    rank = torch.distributed.get_rank()
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    return rank in _POSITION_EMBEDDING_GLOBAL_RANKS


def is_pipeline_stage_before_split(rank=None):
    """如果流水线阶段执行编码器块（对于同时具有编码器和解码器的模型），
    则返回True。"""
    if get_pipeline_model_parallel_world_size() == 1:
        return True
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank < _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False


def is_pipeline_stage_after_split(rank=None):
    """如果流水线阶段执行解码器块（对于同时具有编码器和解码器的模型），
    则返回True。"""
    if get_pipeline_model_parallel_world_size() == 1:
        return True
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank >= _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False


def is_pipeline_stage_at_split():
    """如果流水线阶段执行解码器块且下一阶段执行编码器块（对于同时具有编码器和
    解码器的模型），则返回true。"""
    rank = get_pipeline_model_parallel_rank()
    return is_pipeline_stage_before_split(rank) and \
            is_pipeline_stage_after_split(rank+1)


def get_virtual_pipeline_model_parallel_rank():
    """返回虚拟流水线并行排名。"""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK


def set_virtual_pipeline_model_parallel_rank(rank):
    """设置虚拟流水线并行排名。"""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_virtual_pipeline_model_parallel_world_size():
    """返回虚拟流水线并行世界大小。"""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE


def get_tensor_model_parallel_src_rank():
    """计算张量模型并行组中第一个本地排名对应的全局排名。"""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_data_parallel_src_rank():
    """计算数据并行组中第一个本地排名对应的全局排名。"""
    assert _DATA_PARALLEL_GLOBAL_RANKS is not None, \
        "数据并行组未初始化"
    return _DATA_PARALLEL_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_first_rank():
    """返回当前张量并行组在流水线中的第一个进程的全局排名"""
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "流水线并行组未初始化"
    return _PIPELINE_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_last_rank():
    """返回当前张量并行组在流水线中的最后一个进程的全局排名"""
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "流水线并行组未初始化"
    last_rank_local = get_pipeline_model_parallel_world_size() - 1
    return _PIPELINE_GLOBAL_RANKS[last_rank_local]

def get_pipeline_model_parallel_next_rank():
    """返回流水线中跟随调用者的全局排名"""
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "流水线并行组未初始化"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def get_pipeline_model_parallel_prev_rank():
    """返回流水线中在调用者之前的全局排名"""
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "流水线并行组未初始化"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def get_data_parallel_world_size():
    """返回数据并行组的世界大小。"""
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank():
    """返回数据并行组的我的排名。"""
    return torch.distributed.get_rank(group=get_data_parallel_group())

def get_all_reduce_launcher() -> 'GraphAllReduce':
    assert _ALL_REDUCE_LAUNCHER is not None, 'all reduce启动器未初始化'
    return _ALL_REDUCE_LAUNCHER

def destroy_model_parallel():
    """将组设置为none。"""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _EMBEDDING_GROUP
    _EMBEDDING_GROUP = None
    global _POSITION_EMBEDDING_GROUP
    _POSITION_EMBEDDING_GROUP = None
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = None
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = None


class GraphAllReduce:

    def __init__(
        self,
        max_num_tokens: int,
        hidden_size: int,
        dtype: torch.dtype,
        disable_graph: bool = False,
    ) -> None:
        self.max_num_tokens = max_num_tokens
        self.hidden_size = hidden_size
        self.disable_graph = disable_graph

        tp_world_size = get_tensor_model_parallel_world_size()
        if tp_world_size == 1:
            return

        self.group = get_tensor_model_parallel_group()
        self.buffer = torch.empty(
            size=(max_num_tokens, hidden_size),
            dtype=dtype,
            device='cuda',
        )

        # 为不同数量的token构建图。
        if not self.disable_graph:
            self.graphs = {}
            for num_tokens in range(8, max_num_tokens + 1, 8):
                self.graphs[num_tokens] = self._build_graph(num_tokens)

    def _build_graph(self, num_tokens: int) -> torch.cuda.CUDAGraph:
        # 预热。
        torch.distributed.all_reduce(self.buffer[:num_tokens], group=self.group)
        torch.cuda.synchronize()

        # 构建图。
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            torch.distributed.all_reduce(
                self.buffer[:num_tokens], group=self.group)
        torch.cuda.synchronize()
        return graph

    def launch(self, x: torch.Tensor) -> torch.Tensor:
        # 注意：x必须是self.buffer的切片。
        num_tokens = x.shape[0]
        if self.disable_graph:
            torch.distributed.all_reduce(x, group=self.group)
        else:
            self.graphs[num_tokens].replay()
        return x