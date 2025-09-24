"""自定义激活函数。"""
import torch
import torch.nn as nn

from vllm import activation_ops

_ACTIVATION_REGISTRY = {
    "gelu": nn.GELU(),
    "gelu_new": nn.GELU(approximate="tanh"),   # 注意：这可能会引入小的舍入误差。
    "gelu_fast": nn.GELU(approximate="tanh"),  # 注意：这可能会引入小的舍入误差。
    "relu": nn.ReLU(),
}


def get_act_fn(act_fn: str) -> nn.Module:
    """根据名称获取激活函数。"""
    act_fn = act_fn.lower()
    if act_fn in _ACTIVATION_REGISTRY:
        return _ACTIVATION_REGISTRY[act_fn]
    raise ValueError(f"不支持的激活函数 {act_fn!r}。")


class SiluAndMul(nn.Module):
    """用于 SwiGLU 的激活函数。

    该函数计算 x -> silu(x[:d]) * x[d:]，其中 d = x.shape[1] // 2.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,        # (num_tokens, 2 * d)
    ) -> torch.Tensor:          # (num_tokens, d)
        num_tokens = x.shape[0]
        d = x.shape[1] // 2
        out = torch.empty(num_tokens, d, dtype=x.dtype, device=x.device)
        activation_ops.silu_and_mul(out, x)
        return out
