import time
from typing import Any, List, Optional

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.core.scheduler import Scheduler
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.ray_utils import DeviceID, initialize_cluster, ray
from vllm.engine.tokenizer_utils import detokenize_incrementally, get_tokenizer
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Counter
from vllm.worker.worker import Worker

logger = init_logger(__name__)


class LLMEngine:
    """一个接收请求并生成文本的 LLM 引擎。

    这是 vLLM 引擎的主要类。它接收来自客户端的请求并从 LLM 生成文本。它包括一个分词器，
    一个语言模型（可能分布在多个 GPU 上），以及为中间状态（即 KV 缓存）分配的 GPU 内存空间。
    此类利用迭代级调度和高效的内存管理来最大化服务吞吐量。

    `LLM` 类包装此类以进行离线批处理推理，而 `AsyncLLMEngine` 类包装此类以进行在线服务。

    注意：配置参数来自 `EngineArgs` 类。有关参数的完整列表，请参见 `EngineArgs`。

    Args:
        model_config: 与 LLM 模型相关的配置。
        cache_config: 与 KV 缓存内存管理相关的配置。
        parallel_config: 与分布式执行相关的配置。
        scheduler_config: 与请求调度器相关的配置。
        distributed_init_method: 分布式执行的初始化方法。
            有关详细信息，请参见 `torch.distributed.init_process_group`。
        stage_devices: 每个阶段的设备列表。每个阶段都是一组 (rank, node_resource, device) 元组。
        log_stats: 是否记录统计信息。
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        distributed_init_method: str,
        stage_devices: List[List[DeviceID]],
        log_stats: bool,
    ) -> None:
        logger.info(
            "Initializing an LLM engine with config: "
            f"model={model_config.model!r}, "
            f"dtype={model_config.dtype}, "
            f"use_dummy_weights={model_config.use_dummy_weights}, "
            f"download_dir={model_config.download_dir!r}, "
            f"use_np_weights={model_config.use_np_weights}, "
            f"tensor_parallel_size={parallel_config.tensor_parallel_size}, "
            f"seed={model_config.seed})"
        )
        # TODO(woosuk): 在调试模式下打印更多配置。

        self.model_config = model_config
        self.cache_config = cache_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.log_stats = log_stats
        self._verify_args()

        #初始化llm参数时就已经使用transformer的AutoConfig.from_pretrained将分词器配置加载完成了
        self.tokenizer = get_tokenizer(model_config.model)
        self.seq_counter = Counter()

        # 创建并行 GPU 工作进程。
        self.workers: List[Worker] = []
        assert len(stage_devices) == 1, "目前只支持一个阶段。"
        for rank, node_resource, _ in stage_devices[0]:
            worker_cls = Worker
            if self.parallel_config.worker_use_ray:
                worker_cls = ray.remote(
                    num_cpus=0,
                    num_gpus=1,
                    resources={node_resource: 1e-5},
                )(worker_cls).remote

            worker = worker_cls(
                model_config,
                parallel_config,
                scheduler_config,
                rank,
                distributed_init_method,
            )
            self.workers.append(worker)
        # 分析内存使用情况并初始化缓存。
        self._init_cache()

        # 创建调度器。
        self.scheduler = Scheduler(scheduler_config, cache_config, log_stats)

    def _verify_args(self) -> None:
        self.model_config.verify_with_parallel_config(self.parallel_config)
        self.cache_config.verify_with_parallel_config(self.parallel_config)

    def _init_cache(self) -> None:
        """分析内存使用情况并初始化 KV 缓存。"""
        # 获取可以在 GPU 和 CPU 上分配的最大块数。
        num_blocks = self._run_workers(
            "profile_num_available_blocks",
            get_all_outputs=True,
            block_size=self.cache_config.block_size,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
            cpu_swap_space=self.cache_config.swap_space_bytes,
        )

        # 由于我们使用共享的中央控制器，我们取所有工作进程中的最小块数，
        # 以确保内存操作可以应用到所有工作进程。
        num_gpu_blocks = min(b[0] for b in num_blocks)
        num_cpu_blocks = min(b[1] for b in num_blocks)
        # FIXME(woosuk): 更改为调试日志。
        logger.info(f'# GPU 块: {num_gpu_blocks}, '
                    f'# CPU 块: {num_cpu_blocks}')
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        # 初始化缓存。
        self._run_workers("init_cache_engine", cache_config=self.cache_config)

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> "LLMEngine":
        """根据引擎参数创建 LLM 引擎。"""
        # 创建引擎配置。
        engine_configs = engine_args.create_engine_configs()
        # 选择了 分布式执行的配置
        parallel_config = engine_configs[2]
        # 初始化集群。
        distributed_init_method, devices = initialize_cluster(parallel_config)
        # 创建 LLM 引擎。
        engine = cls(*engine_configs, distributed_init_method, devices,
                     log_stats=not engine_args.disable_log_stats)
        return engine

    def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
    ) -> None:
        """将请求添加到引擎的请求池中。

        请求被添加到请求池中，并在调用 `engine.step()` 时由调度器处理。
        确切的调度策略由调度器确定。

        Args:
            request_id: 请求的唯一 ID。
            prompt: 提示字符串。如果提供了 prompt_token_ids，则可以为 None。
            sampling_params: 文本生成的采样参数。
            prompt_token_ids: 提示的令牌 ID。如果为 None，我们使用
                分词器将提示转换为令牌 ID。
            arrival_time: 请求到达时间。如果为 None，我们使用当前时间。
        """
        if arrival_time is None:
            arrival_time = time.time()
        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = self.tokenizer.encode(prompt)

        # 创建序列。
        block_size = self.cache_config.block_size
        seqs: List[Sequence] = []
        for _ in range(sampling_params.best_of):
            seq_id = next(self.seq_counter)
            seq = Sequence(seq_id, prompt, prompt_token_ids, block_size)
            seqs.append(seq)

        # 创建序列组。
        seq_group = SequenceGroup(request_id, seqs, sampling_params,
                                  arrival_time)

        # 将序列组添加到调度器。
        self.scheduler.add_seq_group(seq_group)

    def abort_request(self, request_id: str) -> None:
        """中止具有给定 ID 的请求。

        Args:
            request_id: 要中止的请求的 ID。
        """
        self.scheduler.abort_seq_group(request_id)

    def get_num_unfinished_requests(self) -> int:
        """获取未完成请求的数量。"""
        return self.scheduler.get_num_unfinished_seq_groups()

    def has_unfinished_requests(self) -> bool:
        """如果有未完成的请求，则返回 True。"""
        return self.scheduler.has_unfinished_seqs()

    def step(self) -> List[RequestOutput]:
        """执行一次解码迭代并返回新生成的结果。

        此函数执行引擎的一次解码迭代。它首先调度在下次迭代中要执行的序列和
        要交换进/出/复制的令牌块。然后，它执行模型并使用模型输出更新调度器。
        最后，它解码序列并返回新生成的结果。
        """
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
        if (not seq_group_metadata_list) and scheduler_outputs.is_empty():
            # 无事可做。
            return []

        # 执行模型。
        output = self._run_workers(
            "execute_model",
            seq_group_metadata_list=seq_group_metadata_list,
            blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
            blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
            blocks_to_copy=scheduler_outputs.blocks_to_copy,
        )
        # 使用模型输出更新调度器。
        seq_groups = self.scheduler.update(output)

        # 解码序列。
        self._decode_sequences(seq_groups)
        # 停止满足停止条件的序列。
        self._stop_sequences(seq_groups)
        # 释放已完成的序列组。
        self.scheduler.free_finished_seq_groups()

        # 创建输出。
        request_outputs: List[RequestOutput] = []
        for seq_group in seq_groups:
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)
        return request_outputs

    def _decode_sequences(self, seq_groups: List[SequenceGroup]) -> None:
        """解码序列输出。"""
        for seq_group in seq_groups:
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                new_token, new_output_text = detokenize_incrementally(
                    self.tokenizer,
                    seq.output_tokens,
                    seq.get_last_token_id(),
                    skip_special_tokens=True,
                )
                seq.output_tokens.append(new_token)
                seq.output_text = new_output_text

    def _stop_sequences(self, seq_groups: List[SequenceGroup]) -> None:
        """停止已完成的序列。"""
        for seq_group in seq_groups:
            sampling_params = seq_group.sampling_params
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                # 检查序列是否已生成停止字符串。
                stopped = False
                for stop_str in sampling_params.stop:
                    if seq.output_text.endswith(stop_str):
                        # 截断输出文本，使停止字符串不包含在输出中。
                        seq.output_text = seq.output_text[:-len(stop_str)]
                        self.scheduler.free_seq(seq,
                                                SequenceStatus.FINISHED_STOPPED)
                        stopped = True
                        break
                if stopped:
                    continue

                # 检查序列是否已达到 max_tokens。
                if seq.get_output_len() == sampling_params.max_tokens:
                    self.scheduler.free_seq(
                        seq, SequenceStatus.FINISHED_LENGTH_CAPPED)
                    continue
                # 检查序列是否已生成 EOS 令牌。
                if not sampling_params.ignore_eos:
                    if seq.get_last_token_id() == self.tokenizer.eos_token_id:
                        self.scheduler.free_seq(seq,
                                                SequenceStatus.FINISHED_STOPPED)
                        continue

    def _run_workers(
        self,
        method: str,
        get_all_outputs: bool = False,
        *args,
        **kwargs,
    ) -> Any:
        """在所有工作进程上运行给定方法。"""
        all_outputs = []
        for worker in self.workers:
            executor = getattr(worker, method)
            if self.parallel_config.worker_use_ray:
                executor = executor.remote

            output = executor(*args, **kwargs)
            all_outputs.append(output)

        if self.parallel_config.worker_use_ray:
            all_outputs = ray.get(all_outputs)

        if get_all_outputs:
            return all_outputs

        # 确保所有工作进程都有相同的结果。
        output = all_outputs[0]
        for other_output in all_outputs[1:]:
            assert output == other_output
        return output
