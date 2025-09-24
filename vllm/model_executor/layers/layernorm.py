"""自定义归一化层。"""
import torch
import torch.nn as nn

from vllm import layernorm_ops


class RMSNorm(nn.Module):
    """均方根归一化。

    计算 x -> w * x / sqrt(E[x^2] + eps)，其中 w 是学习到的权重。
    参考 https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        layernorm_ops.rms_norm(
            out,
            x,
            self.weight.data,
            self.variance_epsilon,
        )
        return out
