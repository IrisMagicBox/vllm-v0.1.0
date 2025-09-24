"""管理 token 块的块管理器。"""
from typing import Dict, List, Optional, Set, Tuple

from vllm.block import PhysicalTokenBlock
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device


class BlockAllocator:
    """管理设备的空闲物理 token 块。

    分配器维护一个空闲块列表，并在请求时分配一个块。
    当一个块被释放时，其引用计数将减少。如果引用计数变为零，则将该块添加回空闲列表。
    """

    def __init__(
        self,
        device: Device,
        block_size: int,
        num_blocks: int,
    ) -> None:
        self.device = device
        self.block_size = block_size
        self.num_blocks = num_blocks

        # 初始化空闲块。
        self.free_blocks: List[PhysicalTokenBlock] = []
        for i in range(num_blocks):
            block = PhysicalTokenBlock(
                device=device, block_number=i, block_size=block_size)
            self.free_blocks.append(block)

    def allocate(self) -> PhysicalTokenBlock:
        if not self.free_blocks:
            raise ValueError("内存不足！没有可用的空闲块。")
        block = self.free_blocks.pop()
        block.ref_count = 1
        return block

    def free(self, block: PhysicalTokenBlock) -> None:
        if block.ref_count == 0:
            raise ValueError(f"重复释放！{block} 已经被释放。")
        block.ref_count -= 1
        if block.ref_count == 0:
            self.free_blocks.append(block)

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)


# 映射：逻辑块号 -> 物理块。
BlockTable = List[PhysicalTokenBlock]


class BlockSpaceManager:
    """管理逻辑和物理 token 块之间的映射。"""

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks
        self.watermark = watermark
        assert watermark >= 0.0

        self.watermark_blocks = int(watermark * num_gpu_blocks)
        self.gpu_allocator = BlockAllocator(Device.GPU, block_size,
                                            num_gpu_blocks)
        self.cpu_allocator = BlockAllocator(Device.CPU, block_size,
                                            num_cpu_blocks)
        # 映射：seq_id -> BlockTable。
        self.block_tables: Dict[int, BlockTable] = {}

    def can_allocate(self, seq_group: SequenceGroup) -> bool:
        # FIXME(woosuk): 这里我们假设组中的所有序列共享
        # 相同的提示。对于被抢占的序列，这可能不成立。
        seq = seq_group.get_seqs()[0]
        num_required_blocks = len(seq.logical_token_blocks)
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        # 使用水印来避免频繁的缓存逐出。
        return num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks

    def allocate(self, seq_group: SequenceGroup) -> None:
        # NOTE: 这里我们假设组中的所有序列具有相同的提示。
        seq = seq_group.get_seqs()[0]

        # 分配存储提示 token 的新物理 token 块。
        block_table: BlockTable = []
        for _ in range(len(seq.logical_token_blocks)):
            block = self.gpu_allocator.allocate()
            # 设置 token 块的引用计数。
            block.ref_count = seq_group.num_seqs()
            block_table.append(block)

        # 为每个序列分配块表。
        for seq in seq_group.get_seqs():
            self.block_tables[seq.seq_id] = block_table.copy()

    def can_append_slot(self, seq_group: SequenceGroup) -> bool:
        # 简单启发式：如果每个序列至少有一个空闲块，
        # 我们可以追加。
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
        return num_seqs <= num_free_gpu_blocks

    def append_slot(self, seq: Sequence) -> Optional[Tuple[int, int]]:
        """为新 token 分配物理槽位。"""
        logical_blocks = seq.logical_token_blocks
        block_table = self.block_tables[seq.seq_id]

        if len(block_table) < len(logical_blocks):
            # 该序列有一个新的逻辑块。
            # 分配一个新的物理块。
            block = self.gpu_allocator.allocate()
            block_table.append(block)
            return None

        # 我们想将 token 添加到上一个物理块。
        last_block = block_table[-1]
        assert last_block.device == Device.GPU
        if last_block.ref_count == 1:
            # 不与其他序列共享。可附加。
            return None
        else:
            # 上一个块与其他序列共享。
            # 写时复制：分配一个新块并复制 token。
            new_block = self.gpu_allocator.allocate()
            block_table[-1] = new_block
            self.gpu_allocator.free(last_block)
            return last_block.block_number, new_block.block_number

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        # NOTE: fork 不分配新的物理块。
        # 因此，它总是不会出现内存不足。
        src_block_table = self.block_tables[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.copy()
        for block in src_block_table:
            block.ref_count += 1

    def _get_physical_blocks(self, seq_group: SequenceGroup) -> List[PhysicalTokenBlock]:
        # NOTE: 这里，我们假设物理块仅由
        # 相同组中的序列共享。
        blocks: Set[PhysicalTokenBlock] = set()
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            block_table = self.block_tables[seq.seq_id]
            for block in block_table:
                blocks.add(block)
        return list(blocks)

    def can_swap_in(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        num_swapped_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
        num_free_blocks = self.gpu_allocator.get_num_free_blocks()
        # NOTE: 保守地，我们假设每个序列将在
        # 交换进来后立即分配至少一个空闲块。
        # NOTE: 这应该与 can_append_slot() 中的逻辑匹配。
        num_required_blocks = len(blocks) + num_swapped_seqs
        return num_free_blocks - num_required_blocks >= self.watermark_blocks

    def swap_in(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # CPU 块 -> GPU 块。
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]

            for cpu_block in block_table:
                if cpu_block in mapping:
                    gpu_block = mapping[cpu_block]
                    gpu_block.ref_count += 1
                else:
                    gpu_block = self.gpu_allocator.allocate()
                    mapping[cpu_block] = gpu_block
                new_block_table.append(gpu_block)
                # 释放交换到 GPU 的 CPU 块。
                self.cpu_allocator.free(cpu_block)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            cpu_block.block_number: gpu_block.block_number
            for cpu_block, gpu_block in mapping.items()
        }
        return block_number_mapping

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        return len(blocks) <= self.cpu_allocator.get_num_free_blocks()

    def swap_out(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # GPU 块 -> CPU 块。
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]

            for gpu_block in block_table:
                if gpu_block in mapping:
                    cpu_block = mapping[gpu_block]
                    cpu_block.ref_count += 1
                else:
                    cpu_block = self.cpu_allocator.allocate()
                    mapping[gpu_block] = cpu_block
                new_block_table.append(cpu_block)
                # 释放交换到 CPU 的 GPU 块。
                self.gpu_allocator.free(gpu_block)
            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            gpu_block.block_number: cpu_block.block_number
            for gpu_block, cpu_block in mapping.items()
        }
        return block_number_mapping

    def _free_block_table(self, block_table: BlockTable) -> None:
        for block in block_table:
            if block.device == Device.GPU:
                self.gpu_allocator.free(block)
            else:
                self.cpu_allocator.free(block)

    def free(self, seq: Sequence) -> None:
        if seq.seq_id not in self.block_tables:
            # 已经被释放或尚未被调度。
            return
        block_table = self.block_tables[seq.seq_id]
        self._free_block_table(block_table)
        del self.block_tables[seq.seq_id]

    def reset(self) -> None:
        for block_table in self.block_tables.values():
            self._free_block_table(block_table)
        self.block_tables.clear()

    def get_block_table(self, seq: Sequence) -> List[int]:
        block_table = self.block_tables[seq.seq_id]
        return [block.block_number for block in block_table]

    def get_num_free_gpu_blocks(self) -> int:
        return self.gpu_allocator.get_num_free_blocks()

    def get_num_free_cpu_blocks(self) -> int:
        return self.cpu_allocator.get_num_free_blocks()
