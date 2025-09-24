"""用于下载和初始化模型权重的工具."""
import filelock
import glob
import json
import os
from typing import Iterator, List, Optional, Tuple

from huggingface_hub import snapshot_download
import numpy as np
import torch
from tqdm.auto import tqdm


class Disabledtqdm(tqdm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


def hf_model_weights_iterator(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    use_np_cache: bool = False,
) -> Iterator[Tuple[str, torch.Tensor]]:
    # 准备文件锁目录以防止多个进程同时
    # 下载相同的模型权重.
    lock_dir = cache_dir if cache_dir is not None else "/tmp"
    lock_file_name = model_name_or_path.replace("/", "-") + ".lock"
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name))

    # 从huggingface下载模型权重.
    is_local = os.path.isdir(model_name_or_path)
    if not is_local:
        with lock:
            hf_folder = snapshot_download(model_name_or_path,
                                          allow_patterns="*.bin",
                                          cache_dir=cache_dir,
                                          tqdm_class=Disabledtqdm)
    else:
        hf_folder = model_name_or_path

    hf_bin_files = glob.glob(os.path.join(hf_folder, "*.bin"))

    if use_np_cache:
        # 将模型权重从torch张量转换为numpy数组以
        # 加快加载速度.
        np_folder = os.path.join(hf_folder, 'np')
        os.makedirs(np_folder, exist_ok=True)
        weight_names_file = os.path.join(np_folder, 'weight_names.json')
        with lock:
            if not os.path.exists(weight_names_file):
                weight_names = []
                for bin_file in hf_bin_files:
                    state = torch.load(bin_file, map_location="cpu")
                    for name, param in state.items():
                        param_path = os.path.join(np_folder, name)
                        with open(param_path, "wb") as f:
                            np.save(f, param.cpu().detach().numpy())
                        weight_names.append(name)
                with open(weight_names_file, 'w') as f:
                    json.dump(weight_names, f)

        with open(weight_names_file, 'r') as f:
            weight_names = json.load(f)

        for name in weight_names:
            param_path = os.path.join(np_folder, name)
            with open(param_path, "rb") as f:
                param = np.load(f)
            yield name, torch.from_numpy(param)
    else:
        for bin_file in hf_bin_files:
            state = torch.load(bin_file, map_location="cpu")
            for name, param in state.items():
                yield name, param


def load_tensor_parallel_weights(
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    param_name: str,
    column_parallel_weight_names: List[str],
    row_parallel_weight_names: List[str],
    tensor_model_parallel_rank: int,
) -> None:
    for p in column_parallel_weight_names:
        if p in param_name:
            shard_size = param.shape[0]
            loaded_weight = loaded_weight[
                shard_size * tensor_model_parallel_rank
                :shard_size * (tensor_model_parallel_rank + 1)]
            break
    for p in row_parallel_weight_names:
        if p in param_name:
            shard_size = param.shape[1]
            loaded_weight = loaded_weight[
                :,
                shard_size * tensor_model_parallel_rank
                :shard_size * (tensor_model_parallel_rank + 1)]
            break
    assert param.shape == loaded_weight.shape
    param.data.copy_(loaded_weight)


def initialize_dummy_weights(
    model: torch.nn.Module,
    low: float = -1e-3,
    high: float = 1e-3,
) -> None:
    """使用随机值初始化模型权重.

    模型权重必须随机初始化以进行准确的性能
    测量. 此外, 模型权重不应在
    前向传递中导致NaN. 我们根据经验发现, 在
    -1e-3 和 1e-3 之间的值初始化权重
    对大多数模型效果良好.
    """
    for param in model.state_dict().values():
        param.data.uniform_(low, high)
