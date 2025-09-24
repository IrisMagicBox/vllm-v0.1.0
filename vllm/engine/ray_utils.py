import random
from typing import List, Optional, Tuple

try:
    import ray
except ImportError:
    ray = None

from vllm.config import ParallelConfig

DeviceID = Tuple[int, Optional[str], int]  # 排名，节点资源（节点 IP），设备 ID


def initialize_cluster(
    parallel_config: ParallelConfig,
    engine_use_ray: bool = False,
    ray_address: Optional[str] = None,
) -> Tuple[str, List[List[DeviceID]]]:
    """使用 Ray 初始化分布式集群。

    Args:
        parallel_config: 并行执行的配置。
        engine_use_ray: 是否对异步引擎使用 Ray。
        ray_address: Ray 集群的地址。如果为 None，则使用
            默认的 Ray 集群地址。

    Returns:
        一个 (`distributed_init_method`, `all_stage_devices`) 元组。
        `distributed_init_method` 是用于初始化分布式后端的地址。
        `all_stage_devices` 包含每个流水线阶段中每个工作进程的设备 ID。
        每个设备 ID 是一个 (rank, node resource, device id) 元组。
    """
    if parallel_config.worker_use_ray or engine_use_ray:
        if ray is None:
            raise ImportError(
                "Ray 未安装。请安装 Ray 以使用分布式服务。")
        # 连接到 ray 集群。
        ray.init(address=ray_address)

    if not parallel_config.worker_use_ray:
        # 在本地初始化集群。
        port = random.randint(10000, 20000)
        # 我们需要设置分布式初始化方法以确保
        # 分布式 megatron 代码（例如，获取世界大小）正常工作。
        distributed_init_method = f"tcp://localhost:{port}"
        all_stage_devices = [[(0, None, 0)]]
        return distributed_init_method, all_stage_devices

    # 假设我们有一个均匀的集群，目前每个节点的 GPU 数量相同。
    valid_node_resources = []
    num_devices_per_node = None
    for node in ray.nodes():
        if (not node['Alive']) or node['Resources']['GPU'] <= 0:
            continue
        if num_devices_per_node is None:
            num_devices_per_node = node['Resources']['GPU']
        else:
            assert num_devices_per_node == node['Resources']['GPU'], (
                "每个节点的 GPU 数量不统一。")
        for key in node['Resources']:
            if key.startswith('node:'):
                valid_node_resources.append(key)

    # 验证并行配置。
    num_nodes = len(valid_node_resources)
    if parallel_config.world_size > num_nodes * num_devices_per_node:
        raise ValueError(
            "所需的 GPU 数量超过了可用 GPU 的总数。")
    if parallel_config.tensor_parallel_size >= num_devices_per_node:
        if parallel_config.tensor_parallel_size % num_devices_per_node != 0:
            raise ValueError(
                "张量并行数量不能被每个节点的 GPU 数量整除。")
    else:
        if num_devices_per_node % parallel_config.tensor_parallel_size != 0:
            raise ValueError(
                "每个节点的 GPU 数量不能被张量并行数量整除。")

    # 将 GPU 分配到流水线阶段。
    rank = 0
    current_node_id = 0
    current_device_id = 0
    distributed_init_method = None
    all_stage_devices = []

    for _ in range(parallel_config.pipeline_parallel_size):
        stage_devices = []
        for _ in range(parallel_config.tensor_parallel_size):
            node_resource = valid_node_resources[current_node_id]
            stage_devices.append((rank, node_resource, current_device_id))
            if distributed_init_method is None:
                ip = node_resource.split("node:")[-1]
                port = random.randint(10000, 20000)
                distributed_init_method = f"tcp://{ip}:{port}"
            rank += 1
            current_device_id += 1
            if current_device_id >= num_devices_per_node:
                current_node_id += 1
                current_device_id = 0
        all_stage_devices.append(stage_devices)

    return distributed_init_method, all_stage_devices
