import argparse
import dataclasses
from dataclasses import dataclass
from typing import Optional, Tuple

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)


@dataclass
class EngineArgs:
    """vLLM 引擎的参数。"""
    model: str
    download_dir: Optional[str] = None
    use_np_weights: bool = False
    use_dummy_weights: bool = False
    dtype: str = "auto"
    seed: int = 0
    worker_use_ray: bool = False
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    # 默认情况下，每个块可以存储 16 个连续的 token
    block_size: int = 16
    swap_space: int = 4  # GiB
    gpu_memory_utilization: float = 0.90
    max_num_batched_tokens: int = 2560
    max_num_seqs: int = 256
    disable_log_stats: bool = False

    def __post_init__(self):
        self.max_num_seqs = min(self.max_num_seqs, self.max_num_batched_tokens)

    @staticmethod
    def add_cli_args(
        parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        """vLLM 引擎的共享 CLI 参数。"""
        # 模型参数
        parser.add_argument('--model', type=str, default='facebook/opt-125m',
                            help='要使用的 HuggingFace 模型的名称或路径')
        parser.add_argument('--download-dir', type=str,
                            default=EngineArgs.download_dir,
                            help='下载和加载权重的目录，'
                                 '默认为 HuggingFace 的默认缓存目录')
        parser.add_argument('--use-np-weights', action='store_true',
                            help='保存模型权重的 numpy 副本以加快加载速度。这可能会使磁盘使用量增加高达 2 倍。')
        parser.add_argument('--use-dummy-weights', action='store_true',
                            help='对模型权重使用虚拟值')
        # TODO(woosuk): 支持 FP32。
        parser.add_argument('--dtype', type=str, default=EngineArgs.dtype,
                            choices=['auto', 'half', 'bfloat16', 'float'],
                            help='模型权重和激活的数据类型。'
                                 '“auto”选项将对 FP32 和 FP16 模型使用 FP16 精度，对 BF16 模型使用 BF16 精度。')
        # 并行参数
        parser.add_argument('--worker-use-ray', action='store_true',
                            help='使用 Ray 进行分布式服务，当使用超过 1 个 GPU 时会自动设置')
        parser.add_argument('--pipeline-parallel-size', '-pp', type=int,
                            default=EngineArgs.pipeline_parallel_size,
                            help='流水线阶段数')
        parser.add_argument('--tensor-parallel-size', '-tp', type=int,
                            default=EngineArgs.tensor_parallel_size,
                            help='张量并行副本数')
        # KV 缓存参数
        parser.add_argument('--block-size', type=int,
                            default=EngineArgs.block_size,
                            choices=[8, 16, 32],
                            help='令牌块大小')
        # TODO(woosuk): 支持细粒度种子（例如，每个请求的种子）。
        parser.add_argument('--seed', type=int, default=EngineArgs.seed,
                            help='随机种子')
        parser.add_argument('--swap-space', type=int,
                            default=EngineArgs.swap_space,
                            help='每个 GPU 的 CPU 交换空间大小（GiB）')
        parser.add_argument('--gpu-memory-utilization', type=float,
                            default=EngineArgs.gpu_memory_utilization,
                            help='分配给模型执行器的 GPU 内存百分比')
        parser.add_argument('--max-num-batched-tokens', type=int,
                            default=EngineArgs.max_num_batched_tokens,
                            help='每次迭代的最大批处理令牌数')
        parser.add_argument('--max-num-seqs', type=int,
                            default=EngineArgs.max_num_seqs,
                            help='每次迭代的最大序列数')
        parser.add_argument('--disable-log-stats', action='store_true',
                            help='禁用日志统计')
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "EngineArgs":
        # 获取此数据类的属性列表。
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        # 从解析的参数设置属性。
        engine_args = cls(**{attr: getattr(args, attr) for attr in attrs})
        return engine_args

    def create_engine_configs(
        self,
    ) -> Tuple[ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig]:
        # 初始化配置。
        model_config = ModelConfig(
            self.model, self.download_dir, self.use_np_weights,
            self.use_dummy_weights, self.dtype, self.seed)
        cache_config = CacheConfig(self.block_size, self.gpu_memory_utilization,
                                   self.swap_space)
        parallel_config = ParallelConfig(self.pipeline_parallel_size,
                                         self.tensor_parallel_size,
                                         self.worker_use_ray)
        scheduler_config = SchedulerConfig(self.max_num_batched_tokens,
                                           self.max_num_seqs)
        return model_config, cache_config, parallel_config, scheduler_config


@dataclass
class AsyncEngineArgs(EngineArgs):
    """异步 vLLM 引擎的参数。"""
    engine_use_ray: bool = False
    disable_log_requests: bool = False

    @staticmethod
    def add_cli_args(
        parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:
        parser = EngineArgs.add_cli_args(parser)
        parser.add_argument('--engine-use-ray', action='store_true',
                            help='使用 Ray 在服务器进程之外的单独进程中启动 LLM 引擎。')
        parser.add_argument('--disable-log-requests', action='store_true',
                            help='禁用请求日志')
        return parser
