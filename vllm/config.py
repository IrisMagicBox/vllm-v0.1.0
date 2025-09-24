from typing import Optional

import torch
from transformers import AutoConfig, PretrainedConfig

from vllm.logger import init_logger
from vllm.utils import get_cpu_memory

logger = init_logger(__name__)

_GiB = 1 << 30


class ModelConfig:
    """模型配置。

    Args:
        model: 要使用的huggingface模型的名称或路径。
        download_dir: 下载和加载权重的目录，默认为
            huggingface的默认缓存目录。
        use_np_weights: 保存模型权重的numpy副本以加快加载速度。
            这可能会使磁盘使用量增加最多2倍。
        use_dummy_weights: 使用虚拟值作为模型权重（用于分析）。
        dtype: 模型权重和激活的数据类型。"auto"选项
            将对FP32和FP16模型使用FP16精度，对BF16模型使用BF16精度。
        seed: 用于重现的随机种子。
    """

    def __init__(
        self,
        model: str,
        download_dir: Optional[str],
        use_np_weights: bool,
        use_dummy_weights: bool,
        dtype: str,
        seed: int,
    ) -> None:
        self.model = model
        self.download_dir = download_dir
        self.use_np_weights = use_np_weights
        self.use_dummy_weights = use_dummy_weights
        self.seed = seed

        self.hf_config: PretrainedConfig = AutoConfig.from_pretrained(model)
        self.dtype = _get_and_verify_dtype(self.hf_config, dtype)

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        total_num_attention_heads = self.hf_config.num_attention_heads
        tensor_parallel_size = parallel_config.tensor_parallel_size
        if total_num_attention_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"注意力头总数 ({total_num_attention_heads})"
                " 必须能被张量并行大小整除 "
                f"({tensor_parallel_size}).")

        total_num_hidden_layers = self.hf_config.num_hidden_layers
        pipeline_parallel_size = parallel_config.pipeline_parallel_size
        if total_num_hidden_layers % pipeline_parallel_size != 0:
            raise ValueError(
                f"隐藏层总数 ({total_num_hidden_layers}) "
                " 必须能被流水线并行大小整除 "
                f"({pipeline_parallel_size}).")

    def get_hidden_size(self) -> int:
        return self.hf_config.hidden_size

    def get_head_size(self) -> int:
        # FIXME(woosuk): 这可能不是所有模型都适用。
        return self.hf_config.hidden_size // self.hf_config.num_attention_heads

    def get_num_heads(self, parallel_config: "ParallelConfig") -> int:
        total_num_attention_heads = self.hf_config.num_attention_heads
        return total_num_attention_heads // parallel_config.tensor_parallel_size

    def get_num_layers(self, parallel_config: "ParallelConfig") -> int:
        total_num_hidden_layers = self.hf_config.num_hidden_layers
        return total_num_hidden_layers // parallel_config.pipeline_parallel_size


class CacheConfig:
    """KV缓存的配置。

    Args:
        block_size: 缓存块的大小（以token数量为单位）。
        gpu_memory_utilization: 用于vLLM执行的GPU内存比例。
        swap_space: 每个GPU的CPU交换空间大小（以GiB为单位）。
    """
    def __init__(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        swap_space: int,
    ) -> None:
        self.block_size = block_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.swap_space_bytes = swap_space * _GiB
        self._verify_args()

        # 将在分析后设置。
        self.num_gpu_blocks = None
        self.num_cpu_blocks = None

    def _verify_args(self) -> None:
        if self.gpu_memory_utilization > 1.0:
            raise ValueError(
                "GPU内存利用率必须小于1.0。得到 "
                f"{self.gpu_memory_utilization}.")

    def verify_with_parallel_config(
        self,
        parallel_config: "ParallelConfig",
    ) -> None:
        total_cpu_memory = get_cpu_memory()
        # FIXME(woosuk): 在这里，假设张量并行组中的GPU在同一个节点上。
        # 然而，GPU可能分布在多个节点上。
        num_gpus_per_node = parallel_config.tensor_parallel_size
        cpu_memory_usage = self.swap_space_bytes * num_gpus_per_node

        msg = (
            f"{cpu_memory_usage / _GiB:.2f} GiB 的 "
            f"总CPU内存 {total_cpu_memory / _GiB:.2f} GiB 中 "
            "被分配给交换空间。")
        if cpu_memory_usage > 0.7 * total_cpu_memory:
            raise ValueError("交换空间过大。 " + msg)
        elif cpu_memory_usage > 0.4 * total_cpu_memory:
            logger.warn("交换空间可能过大。 " + msg)


class ParallelConfig:
    """分布式执行的配置。

    Args:
        pipeline_parallel_size: 流水线并行组的数量。
        tensor_parallel_size: 张量并行组的数量。
        worker_use_ray: 是否为模型工作进程使用Ray。
            如果pipeline_parallel_size或tensor_parallel_size大于1，则将设置为True。
    """
    def __init__(
        self,
        pipeline_parallel_size: int,
        tensor_parallel_size: int,
        worker_use_ray: bool,
    ) -> None:
        self.pipeline_parallel_size = pipeline_parallel_size
        self.tensor_parallel_size = tensor_parallel_size
        self.worker_use_ray = worker_use_ray

        self.world_size = pipeline_parallel_size * tensor_parallel_size
        if self.world_size > 1:
            self.worker_use_ray = True
        self._verify_args()

    def _verify_args(self) -> None:
        if self.pipeline_parallel_size > 1:
            raise NotImplementedError(
                "流水线并行目前还不支持。")


class SchedulerConfig:
    """调度器配置。

    Args:
        max_num_batched_tokens: 单次迭代中要处理的最大token数量。
        max_num_seqs: 单次迭代中要处理的最大序列数量。
    """
    def __init__(
        self,
        max_num_batched_tokens: int,
        max_num_seqs: int,
    ) -> None:
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs


_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def _get_and_verify_dtype(
    config: PretrainedConfig,
    dtype: str,
) -> torch.dtype:
    # NOTE: getattr(config, "torch_dtype", torch.float32) 不正确
    # 因为config.torch_dtype可能为None。
    config_dtype = getattr(config, "torch_dtype", None)
    if config_dtype is None:
        config_dtype = torch.float32

    dtype = dtype.lower()
    if dtype == "auto":
        if config_dtype == torch.float32:
            # 按照常见做法，我们对float32模型使用float16。
            torch_dtype = torch.float16
        else:
            torch_dtype = config_dtype
    else:
        if dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
            raise ValueError(f"未知的dtype: {dtype}")
        torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]

    # 验证dtype。
    if torch_dtype != config_dtype:
        if torch_dtype == torch.float32:
            # 允许向上转换为float32。
            pass
        elif config_dtype == torch.float32:
            # 允许从float32向下转换为float16或bfloat16。
            pass
        else:
            # 允许在float16和bfloat16之间转换，但会发出警告。
            logger.warn(f"将 {config_dtype} 转换为 {torch_dtype}.")

    # 检查GPU是否支持该dtype。
    if torch_dtype == torch.bfloat16:
        compute_capability = torch.cuda.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16仅在计算能力至少为8.0的GPU上支持。"
                f"你的 {gpu_name} GPU的计算能力为 "
                f"{compute_capability[0]}.{compute_capability[1]}.")
    return torch_dtype
