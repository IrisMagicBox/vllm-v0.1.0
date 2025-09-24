from typing import List, Tuple, Union

from transformers import (AutoConfig, AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

from vllm.logger import init_logger

logger = init_logger(__name__)

_MODEL_TYPES_WITH_SLOW_TOKENIZER = []


def get_tokenizer(
    model_name: str,
    *args,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """通过 Huggingface 获取给定模型名称的分词器。"""
    config = AutoConfig.from_pretrained(model_name)
    if "open_llama" in model_name:
        kwargs["use_fast"] = False
        logger.info(
            "OpenLLaMA 模型不支持快速分词器。"
            "改用慢速分词器。")
    elif config.model_type == "llama" and getattr(kwargs, "use_fast", True):
        # LLaMA 快速分词器在某些环境中会导致 protobuf 错误。
        # 然而，我们发现下面的 LLaMA 快速分词器在大多数环境中运行良好。
        model_name = "hf-internal-testing/llama-tokenizer"
        logger.info(
            f"使用 '{model_name}' 中的 LLaMA 快速分词器以避免"
            "潜在的 protobuf 错误。")
    elif config.model_type in _MODEL_TYPES_WITH_SLOW_TOKENIZER:
        if getattr(kwargs, "use_fast", False) == True:
            raise ValueError(
                f"由于快速分词器的错误，无法对 {config.model_type} 使用快速分词器。")
        logger.info(
            f"由于快速分词器的错误，对 {config.model_type} 使用慢速分词器。"
            "这可能会导致性能下降。")
        kwargs["use_fast"] = False
    return AutoTokenizer.from_pretrained(model_name, *args, **kwargs)


def detokenize_incrementally(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    prev_output_tokens: List[str],
    new_token_id: int,
    skip_special_tokens: bool,
) -> Tuple[str, str]:
    """将新令牌与之前的输出令牌一起进行去分词化。

    注意：此函数不会更新 prev_output_tokens。

    Returns:
        new_token: 作为字符串的新令牌。
        output_text: 作为字符串的新输出文本。
    """
    new_token = tokenizer.convert_ids_to_tokens(
        new_token_id, skip_special_tokens=skip_special_tokens)
    output_tokens = prev_output_tokens + [new_token]

    # 将令牌转换为字符串。
    # 优化：如果分词器没有 `added_tokens_encoder`，
    # 则可以直接使用 `convert_tokens_to_string`。
    if not getattr(tokenizer, "added_tokens_encoder", {}):
        output_text = tokenizer.convert_tokens_to_string(output_tokens)
        return new_token, output_text

    # 改编自 https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/tokenization_utils.py#L921
    # 注意(woosuk)：以下代码很慢，因为它是对 output_tokens 运行 for 循环。
    # 在 Python 中，即使循环体很简单，对列表运行 for 循环也可能很慢。
    sub_texts = []
    current_sub_text = []
    for token in output_tokens:
        if skip_special_tokens and token in tokenizer.all_special_ids:
            continue
        if token in tokenizer.added_tokens_encoder:
            if current_sub_text:
                sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
                sub_texts.append(sub_text)
                current_sub_text = []
            sub_texts.append(token)
        else:
            current_sub_text.append(token)
    if current_sub_text:
        sub_text = tokenizer.convert_tokens_to_string(current_sub_text)
        sub_texts.append(sub_text)
    output_text = " ".join(sub_texts)
    return new_token, output_text
