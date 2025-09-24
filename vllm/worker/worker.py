"""GPU工作类。"""
from typing import Dict, List, Tuple

import torch

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.model_executor import get_model, InputMetadata, set_random_seed
from vllm.model_executor.parallel_utils.parallel_state import (
    initialize_model_parallel, initialize_all_reduce_launcher)
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceData, SequenceGroupMetadata, SequenceOutputs
from vllm.worker.cache_engine import CacheEngine
from vllm.utils import get_gpu_memory


class Worker:
    """在GPU上执行（模型分区）的工作类。

    每个工作者与单个GPU相关联。工作者负责在GPU上维护KV缓存和执行模型。
    在分布式推理的情况下，每个工作者被分配模型的一个分区。
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        rank: int,
        distributed_init_method: str,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.rank = rank
        self.distributed_init_method = distributed_init_method

        # 初始化分布式环境。
        _init_distributed_environment(parallel_config, rank,
                                      distributed_init_method)

        # 初始化模型。
        set_random_seed(self.model_config.seed)
        self.model = get_model(model_config)
        initialize_all_reduce_launcher(
            self.scheduler_config.max_num_batched_tokens,
            self.model_config.get_hidden_size(),
            self.model_config.dtype,
        )

        # 未初始化的缓存引擎。将由
        # self.init_cache_engine() 初始化。
        self.cache_config = None
        self.block_size = None
        self.cache_engine = None
        self.cache_events = None
        self.gpu_cache = None

    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
    ) -> Tuple[int, int]:
        # 分析模型的内存使用情况并获得可以使用剩余空闲内存分配的
        # 最大缓存块数。
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # 分析具有max_num_sequences序列的内存使用情况且总token数
        # 等于max_num_batched_tokens。

        # 启用top-k采样以反映准确的内存使用情况。
        sampling_params = SamplingParams(top_p=0.99,
                                         top_k=self.model.config.vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs
        seqs = []
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            seq_data = SequenceData([0] * seq_len)
            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
            )
            seqs.append(seq)

        input_tokens, input_positions, input_metadata = self._prepare_inputs(seqs)

        # 执行模型。
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        self.model(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=[(None, None)] * num_layers,
            input_metadata=input_metadata,
            cache_events=None,
        )

        # 计算可以使用分析得出的峰值内存分配的块数。
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        total_gpu_memory = get_gpu_memory()
        cache_block_size = CacheEngine.get_cache_block_size(
            block_size, self.model_config, self.parallel_config)
        num_gpu_blocks = int((total_gpu_memory * gpu_memory_utilization
                              - peak_memory) // cache_block_size)
        num_cpu_blocks = int(cpu_swap_space // cache_block_size)
        torch.cuda.empty_cache()

        # 重置种子以确保随机状态不受模型初始化和分析的影响。
        set_random_seed(self.model_config.seed)
        return num_gpu_blocks, num_cpu_blocks

    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        self.cache_config = cache_config
        self.block_size = cache_config.block_size
        self.cache_engine = CacheEngine(
            self.cache_config, self.model_config, self.parallel_config)
        self.cache_events = self.cache_engine.events
        self.gpu_cache = self.cache_engine.gpu_cache

    def _prepare_inputs(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata]:
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []

        # 添加提示token。
        prompt_lens: List[int] = []
        for seq_group_metadata in seq_group_metadata_list:
            if not seq_group_metadata.is_prompt:
                continue

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            # 使用组中的任何序列。
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)

            input_tokens.extend(prompt_tokens)
            # 注意(woosuk): 这里我们假设提示中的第一个token
            # 始终是序列中的第一个token。
            input_positions.extend(range(len(prompt_tokens)))

            if seq_group_metadata.block_tables is None:
                # 在内存分析期间，块表尚未初始化。
                # 在这种情况下，我们只使用虚拟槽映射。
                slot_mapping.extend([0] * prompt_len)
                continue

            # 计算槽映射。
            block_table = seq_group_metadata.block_tables[seq_id]
            for i in range(prompt_len):
                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

        # 添加生成token。
        max_context_len = 0
        max_num_blocks_per_seq = 0
        context_lens: List[int] = []
        generation_block_tables: List[List[int]] = []
        for seq_group_metadata in seq_group_metadata_list:
            if seq_group_metadata.is_prompt:
                continue

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append(generation_token)

                context_len = seq_data.get_len()
                position = context_len - 1
                input_positions.append(position)

                block_table = seq_group_metadata.block_tables[seq_id]
                generation_block_tables.append(block_table)

                max_context_len = max(max_context_len, context_len)
                max_num_blocks_per_seq = max(
                    max_num_blocks_per_seq, len(block_table))
                context_lens.append(context_len)

                block_number = block_table[position // self.block_size]
                block_offset = position % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

        # 优化：将输入长度填充为8的倍数。
        # 这是利用NVIDIA GPU中的Tensor Core所必需的。
        input_tokens = _pad_to_alignment(input_tokens, multiple_of=8)
        input_positions = _pad_to_alignment(input_positions, multiple_of=8)

        # Convert to tensors.
        tokens_tensor = torch.cuda.LongTensor(input_tokens)
        positions_tensor = torch.cuda.LongTensor(input_positions)
        slot_mapping_tensor = torch.cuda.IntTensor(slot_mapping)
        context_lens_tensor = torch.cuda.IntTensor(context_lens)
        padded_block_tables = [
            _pad_to_max(block_table, max_num_blocks_per_seq)
            for block_table in generation_block_tables]
        block_tables_tensor = torch.cuda.IntTensor(padded_block_tables)

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        input_metadata = InputMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            slot_mapping=slot_mapping_tensor,
            context_lens=context_lens_tensor,
            max_context_len=max_context_len,
            block_tables=block_tables_tensor,
        )
        return tokens_tensor, positions_tensor, input_metadata

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> Dict[int, SequenceOutputs]:
        # 发出缓存操作。
        issued_cache_op = False
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            issued_cache_op = True

        if issued_cache_op:
            cache_events = self.cache_events
        else:
            cache_events = None

        # 如果没有输入，我们不需要执行模型。
        if not seq_group_metadata_list:
            if cache_events is not None:
                for event in cache_events:
                    event.wait()
            return {}

        # 准备输入张量。
        input_tokens, input_positions, input_metadata = self._prepare_inputs(
            seq_group_metadata_list)

        # 执行模型。
        output = self.model(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=self.gpu_cache,
            input_metadata=input_metadata,
            cache_events=cache_events,
        )
        return output


def _init_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: str,
) -> None:
    """初始化分布式环境。"""
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=parallel_config.world_size,
        rank=rank,
        init_method=distributed_init_method,
    )
    # 一个小的all_reduce用于预热。
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    initialize_model_parallel(parallel_config.tensor_parallel_size,
                              parallel_config.pipeline_parallel_size)


def _pad_to_alignment(x: List[int], multiple_of: int) -> List[int]:
    return x + [0] * ((-len(x)) % multiple_of)


def _pad_to_max(x: List[int], max_len: int) -> List[int]:
    return x + [0] * (max_len - len(x))
