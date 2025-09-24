"""用于选择和加载模型的工具."""
from typing import Type

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.config import ModelConfig
from vllm.model_executor.models import (GPT2LMHeadModel, GPTNeoXForCausalLM,
                                        LlamaForCausalLM, OPTForCausalLM)
from vllm.model_executor.weight_utils import initialize_dummy_weights

# TODO(woosuk): 延迟加载模型类.
_MODEL_REGISTRY = {
    "GPT2LMHeadModel": GPT2LMHeadModel,
    "GPTNeoXForCausalLM": GPTNeoXForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "OPTForCausalLM": OPTForCausalLM,
}


def _get_model_architecture(config: PretrainedConfig) -> Type[nn.Module]:
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[arch]
    raise ValueError(
        f"目前不支持模型架构 {architectures}. "
        f"支持的架构: {list(_MODEL_REGISTRY.keys())}"
    )


def get_model(model_config: ModelConfig) -> nn.Module:
    model_class = _get_model_architecture(model_config.hf_config)
    torch.set_default_dtype(model_config.dtype)

    # 创建模型实例.
    # 权重将被初始化为空张量.
    model = model_class(model_config.hf_config)
    if model_config.use_dummy_weights:
        model = model.cuda()
        # NOTE(woosuk): 为了准确的性能评估，我们将
        # 随机值分配给权重.
        initialize_dummy_weights(model)
    else:
        # 从缓存或下载的文件中加载权重.
        model.load_weights(
            model_config.model, model_config.download_dir,
            model_config.use_np_weights)
        model = model.cuda()
    return model.eval()
