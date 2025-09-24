import enum
import time
from typing import Dict, List, Optional, Tuple

from vllm.config import CacheConfig, SchedulerConfig
from vllm.core.block_manager import BlockSpaceManager
from vllm.core.policy import PolicyFactory
from vllm.logger import init_logger
from vllm.sequence import (Sequence, SequenceData, SequenceGroup,
                           SequenceGroupMetadata, SequenceOutputs,
                           SequenceStatus)

logger = init_logger(__name__)

_LOGGING_INTERVAL_SEC = 5


class PreemptionMode(enum.Enum):
    """抢占模式。

    1. 交换：将被抢占序列的块交换到 CPU 内存中，
    当序列恢复时再交换回来。
    2. 重新计算：丢弃被抢占序列的块，
    当序列恢复时重新计算它们，将序列视为新的提示。
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()


class SchedulerOutputs:

    def __init__(
        self,
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        self.blocks_to_swap_in = blocks_to_swap_in
        self.blocks_to_swap_out = blocks_to_swap_out
        self.blocks_to_copy = blocks_to_copy
        # 交换进和交换出不应同时发生。
        assert not (blocks_to_swap_in and blocks_to_swap_out)

    def is_empty(self) -> bool:
        return (not self.blocks_to_swap_in
                and not self.blocks_to_swap_out
                and not self.blocks_to_copy)


class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        log_stats: bool,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.log_stats = log_stats

        # 实例化调度策略。
        self.policy = PolicyFactory.get_policy(policy_name='fcfs')
        # 创建块空间管理器。
        self.block_manager = BlockSpaceManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=self.cache_config.num_gpu_blocks,
            num_cpu_blocks=self.cache_config.num_cpu_blocks,
        )

        # WAITING 状态的序列组。
        self.waiting: List[SequenceGroup] = []
        # RUNNING 状态的序列组。
        self.running: List[SequenceGroup] = []
        # SWAPPED 状态的序列组。
        self.swapped: List[SequenceGroup] = []

        self.last_logging_time: float = 0.0
        # List[时间戳, token数量]
        self.num_input_tokens: List[Tuple[float, int]] = []

    def add_seq_group(self, seq_group: SequenceGroup) -> None:
        # 将序列组添加到等待队列。
        self.waiting.append(seq_group)

    def abort_seq_group(self, request_id: str) -> None:
        for state_queue in [self.waiting, self.running, self.swapped]:
            for seq_group in state_queue:
                if seq_group.request_id == request_id:
                    # 从状态队列中移除序列组。
                    state_queue.remove(seq_group)
                    for seq in seq_group.seqs:
                        if seq.is_finished():
                            continue
                        self.free_seq(seq, SequenceStatus.FINISHED_ABORTED)
                    return

    def has_unfinished_seqs(self) -> bool:
        return self.waiting or self.running or self.swapped

    def get_num_unfinished_seq_groups(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def _schedule(self) -> Tuple[SchedulerOutputs, List[str]]:
        # 模型执行前需要交换或复制的块。
        blocks_to_swap_in: Dict[int, int] = {}
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}

        # 固定当前时间。
        now = time.time()

        # NOTE(woosuk): 我们优先考虑 RUNNING 状态的序列组，
        # 以最小化抢占开销。
        # 仅当没有可用槽位来保持所有序列组处于 RUNNING 状态时，
        # 才会发生抢占。
        # 在这种情况下，策略负责决定抢占哪些序列组。
        self.running = self.policy.sort_by_priority(now, self.running)

        # 为正在运行的序列组保留新的 token 槽位。
        running: List[SequenceGroup] = []
        preempted: List[SequenceGroup] = []
        while self.running:
            seq_group = self.running.pop(0)
            while not self.block_manager.can_append_slot(seq_group):
                if self.running:
                    # 抢占最低优先级的序列组。
                    victim_seq_group = self.running.pop(-1)
                    self._preempt(victim_seq_group, blocks_to_swap_out)
                    preempted.append(victim_seq_group)
                else:
                    # 没有其他序列组可以被抢占。
                    # 抢占当前序列组。
                    self._preempt(seq_group, blocks_to_swap_out)
                    preempted.append(seq_group)
                    break
            else:
                # 向序列组追加新槽位。
                self._append_slot(seq_group, blocks_to_copy)
                running.append(seq_group)
        self.running = running

        # 如果可能，交换 SWAPPED 状态的序列组。
        self.swapped = self.policy.sort_by_priority(now, self.swapped)
        while self.swapped and not blocks_to_swap_out:
            seq_group = self.swapped[0]
            # 如果序列组在此步骤中被抢占，则停止。
            if seq_group in preempted:
                break
            # 如果序列组无法交换进来，则停止。
            if not self.block_manager.can_swap_in(seq_group):
                break

            # RUNNING 状态的序列总数不应
            # 超过最大序列数。
            num_new_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
            num_curr_seqs = len(self.running)
            if num_curr_seqs + num_new_seqs > self.scheduler_config.max_num_seqs:
                break

            seq_group = self.swapped.pop(0)
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slot(seq_group, blocks_to_copy)
            self.running.append(seq_group)

        num_batched_tokens = sum(
            seq_group.num_seqs(status=SequenceStatus.RUNNING)
            for seq_group in self.running
        )

        # 如果可能，加入等待序列。
        prompt_group_ids: List[str] = []
        # NOTE(woosuk): SWAPPED 状态的序列组被严格
        # 优先于 WAITING 状态的序列组。
        # 这是因为我们希望限制被交换的序列组占用的 CPU 内存量。
        if not self.swapped:
            # 优化：我们不排序等待队列，因为被抢占的
            # 序列组被添加到前面，新序列组
            # 被添加到后面。
            while self.waiting:
                seq_group = self.waiting[0]
                # 如果序列组在此步骤中被抢占，则停止。
                if seq_group in preempted:
                    break
                # 如果序列组无法分配，则停止。
                if not self.block_manager.can_allocate(seq_group):
                    break

                # 如果批处理 tokens 的数量超过限制，则停止。
                num_prompt_tokens = seq_group.get_seqs()[0].get_len()
                if (num_batched_tokens + num_prompt_tokens
                    > self.scheduler_config.max_num_batched_tokens):
                    break

                # RUNNING 状态的序列总数不应
                # 超过最大序列数。
                num_new_seqs = seq_group.num_seqs(status=SequenceStatus.WAITING)
                num_curr_seqs = len(self.running)
                if num_curr_seqs + num_new_seqs > self.scheduler_config.max_num_seqs:
                    break

                seq_group = self.waiting.pop(0)
                self._allocate(seq_group)
                self.running.append(seq_group)
                num_batched_tokens += num_prompt_tokens
                prompt_group_ids.append(seq_group.request_id)

        scheduler_outputs = SchedulerOutputs(
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
        )
        if not self.log_stats:
            return scheduler_outputs, prompt_group_ids

        # TODO(woosuk): 将以下代码移到引擎中。
        now = time.time()
        if num_batched_tokens > 0:
            self.num_input_tokens.append((now, num_batched_tokens))
        elapsed_time = now - self.last_logging_time
        if elapsed_time > _LOGGING_INTERVAL_SEC:
            self.last_logging_time = now
            self.num_input_tokens = [
                (t, n) for t, n in self.num_input_tokens
                if now - t < _LOGGING_INTERVAL_SEC
            ]
            if len(self.num_input_tokens) > 1:
                total_num_tokens = sum(n for _, n in self.num_input_tokens[:-1])
                window = now - self.num_input_tokens[0][0]
                avg_throughput = total_num_tokens / window
            else:
                avg_throughput = 0.0

            total_num_gpu_blocks = self.cache_config.num_gpu_blocks
            num_free_gpu_blocks = self.block_manager.get_num_free_gpu_blocks()
            num_used_gpu_blocks = total_num_gpu_blocks - num_free_gpu_blocks
            gpu_cache_usage = num_used_gpu_blocks / total_num_gpu_blocks

            total_num_cpu_blocks = self.cache_config.num_cpu_blocks
            if total_num_cpu_blocks > 0:
                num_free_cpu_blocks = self.block_manager.get_num_free_cpu_blocks()
                num_used_cpu_blocks = total_num_cpu_blocks - num_free_cpu_blocks
                cpu_cache_usage = num_used_cpu_blocks / total_num_cpu_blocks
            else:
                cpu_cache_usage = 0.0

            logger.info(
                f"吞吐量: {avg_throughput:.1f} tokens/s, "
                f"运行中: {len(self.running)} 个请求, "
                f"已交换: {len(self.swapped)} 个请求, "
                f"等待中: {len(self.waiting)} 个请求, "
                f"GPU KV 缓存使用率: {gpu_cache_usage * 100:.1f}%, "
                f"CPU KV 缓存使用率: {cpu_cache_usage * 100:.1f}%")
        return scheduler_outputs, prompt_group_ids

    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs]:
        # 调度序列组。
        # 此函数调用会更改调度程序的内部状态，
        # 例如 self.running、self.swapped 和 self.waiting。
        scheduler_outputs, prompt_group_ids = self._schedule()

        # 创建输入数据结构。
        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for seq_group in self.running:
            is_prompt = seq_group.request_id in prompt_group_ids

            seq_data: Dict[int, List[SequenceData]] = {}
            block_tables: Dict[int, List[int]] = {}
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)

            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=is_prompt,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
            )
            seq_group_metadata_list.append(seq_group_metadata)
        return seq_group_metadata_list, scheduler_outputs

    def update(
        self,
        seq_outputs: Dict[int, SequenceOutputs],
    ) -> List[SequenceGroup]:
        # 更新正在运行的序列并释放块。
        for seq_group in self.running:
            # 在处理新 token 之前处理束搜索结果。
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                output = seq_outputs[seq.seq_id]
                if seq.seq_id != output.parent_seq_id:
                    # 该序列是父序列的分支（束搜索）。
                    # 释放当前序列。
                    self.block_manager.free(seq)
                    # 分支父序列。
                    parent_seq = seq_group.find(output.parent_seq_id)
                    parent_seq.fork(seq)
                    self.block_manager.fork(parent_seq, seq)

            # 处理新 token。
            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                # 向序列追加新 token。
                output = seq_outputs[seq.seq_id]
                seq.append_token_id(output.output_token, output.logprobs)
        # 返回运行队列的浅拷贝，以防止队列
        # 被调用者修改。
        return self.running.copy()

    def free_seq(self, seq: Sequence, finish_status: SequenceStatus) -> None:
        seq.status = finish_status
        self.block_manager.free(seq)

    def free_finished_seq_groups(self) -> None:
        self.running = [
            seq_group for seq_group in self.running
            if not seq_group.is_finished()
        ]

    def _allocate(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs():
            seq.status = SequenceStatus.RUNNING

    def _append_slot(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: Dict[int, List[int]],
    ) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            ret = self.block_manager.append_slot(seq)
            if ret is not None:
                src_block, dst_block = ret
                if src_block in blocks_to_copy:
                    blocks_to_copy[src_block].append(dst_block)
                else:
                    blocks_to_copy[src_block] = [dst_block]

    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> None:
        # 如果未指定抢占模式，我们按如下方式确定模式：
        # 我们默认使用重新计算，因为它比交换产生更低的开销。
        # 但是，当序列组有多个序列
        # （例如束搜索）时，不支持重新计算。在这种情况下，
        # 我们改用交换。
        # FIXME(woosuk): 这使我们的调度策略变得有些奇怪。
        # 由于交换的序列优先于等待的序列，
        # 多序列组隐式地优先于单序列组。
        # TODO(woosuk): 支持多序列组的重新计算。
        # 这可能需要更复杂的 CUDA 内核。
        if preemption_mode is None:
            seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
            if len(seqs) == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP
        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            assert False, '无效的抢占模式。'

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.block_manager.free(seq)
        # NOTE: 对于 FCFS，我们将被抢占的序列组插入到
        # 等待队列的前面。
        self.waiting.insert(0, seq_group)

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        for seq in seqs:
            seq.status = SequenceStatus.SWAPPED
        self._swap_out(seq_group, blocks_to_swap_out)
        self.swapped.append(seq_group)

    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: Dict[int, int],
    ) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): 中止序列组而不是中止
            # 整个引擎。
            raise RuntimeError(
                "由于缺乏 CPU 交换空间而中止。请增加 "
                "交换空间以避免此错误。")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED
