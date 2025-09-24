"""基准测试离线推理吞吐量。"""
import argparse
import json
import random
import time
from typing import List, Tuple

import torch
from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM,
                          PreTrainedTokenizerBase)
from tqdm import tqdm

from vllm import LLM, SamplingParams


def get_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    config = AutoConfig.from_pretrained(model_name)
    if config.model_type == "llama":
        # 潜在的protobuf错误的解决方法。
        model_name = "hf-internal-testing/llama-tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # 为了在HF后端启用填充。
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    return AutoTokenizer.from_pretrained(model_name)


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    # 加载数据集。
    with open(dataset_path) as f:
        dataset = json.load(f)
    # 过滤掉少于2轮的对话。
    dataset = [
        data for data in dataset
        if len(data["conversations"]) >= 2
    ]
    # 只保留每段对话的前两轮。
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # 对提示和补全进行分词。
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # 过滤掉过长的序列。
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # 修剪过短的序列。
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # 修剪过长的序列。
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # 采样请求。
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


def run_vllm(
    requests: List[Tuple[str, int, int]],
    model: str,
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
) -> float:
    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
    )

    # 将请求添加到引擎。
    for prompt, _, output_len in requests:
        sampling_params = SamplingParams(
            n=n,
            temperature=0.0 if use_beam_search else 1.0,
            top_p=1.0,
            use_beam_search=use_beam_search,
            ignore_eos=True,
            max_tokens=output_len,
        )
        # FIXME(woosuk): 不要使用内部方法。
        llm._add_request(
            prompt=prompt,
            prompt_token_ids=None,
            sampling_params=sampling_params,
        )

    start = time.time()
    # FIXME(woosuk): 不要使用内部方法。
    llm._run_engine(use_tqdm=True)
    end = time.time()
    return end - start


def run_hf(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: PreTrainedTokenizerBase,
    n: int,
    use_beam_search: bool,
    max_batch_size: int,
) -> float:
    assert not use_beam_search
    tokenizer = get_tokenizer(model)
    llm = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float16)
    llm = llm.cuda()

    pbar = tqdm(total=len(requests))
    start = time.time()
    batch: List[str] = []
    max_prompt_len = 0
    max_output_len = 0
    for i in range(len(requests)):
        prompt, prompt_len, output_len = requests[i]
        # 将提示添加到批次。
        batch.append(prompt)
        max_prompt_len = max(max_prompt_len, prompt_len)
        max_output_len = max(max_output_len, output_len)
        if len(batch) < max_batch_size and i != len(requests) - 1:
            # 检查是否可以向批次添加更多请求。
            _, next_prompt_len, next_output_len = requests[i + 1]
            if (max(max_prompt_len, next_prompt_len) + max(
                max_output_len, next_output_len)) <= 2048:
                # 我们可以向批次添加更多请求。
                continue

        # 生成序列。
        input_ids = tokenizer(batch, return_tensors="pt", padding=True).input_ids
        llm_outputs = llm.generate(
            input_ids=input_ids.cuda(),
            do_sample=not use_beam_search,
            num_return_sequences=n,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=max_output_len,
        )
        # 包括解码时间。
        tokenizer.batch_decode(llm_outputs, skip_special_tokens=True)
        pbar.update(len(batch))

        # 清空批次。
        batch = []
        max_prompt_len = 0
        max_output_len = 0
    end = time.time()
    return end - start


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)

    # 采样请求。
    tokenizer = get_tokenizer(args.model)
    requests = sample_requests(args.dataset, args.num_prompts, tokenizer)

    if args.backend == "vllm":
        elapsed_time = run_vllm(
            requests, args.model, args.tensor_parallel_size, args.seed, args.n,
            args.use_beam_search)
    elif args.backend == "hf":
        assert args.tensor_parallel_size == 1
        elapsed_time = run_hf(requests, args.model, tokenizer, args.n,
                              args.use_beam_search, args.hf_max_batch_size)
    else:
        raise ValueError(f"未知后端: {args.backend}")
    total_num_tokens = sum(
        prompt_len + output_len
        for _, prompt_len, output_len in requests
    )
    print(f"吞吐量: {len(requests) / elapsed_time:.2f} 请求/秒, "
          f"{total_num_tokens / elapsed_time:.2f} token/秒")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基准测试吞吐量。")
    parser.add_argument("--backend", type=str, choices=["vllm", "hf"],
                        default="vllm")
    parser.add_argument("--dataset", type=str, required=True,
                        help="数据集路径。")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--n", type=int, default=1,
                        help="每个提示生成的序列数。")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=1000,
                        help="要处理的提示数量。")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-max-batch-size", type=int, default=None,
                        help="HF后端的最大批次大小。")
    args = parser.parse_args()
    if args.backend == "vllm":
        if args.hf_max_batch_size is not None:
            raise ValueError("HF最大批次大小仅适用于HF后端。")
    elif args.backend == "hf":
        if args.hf_max_batch_size is None:
            raise ValueError("HF后端需要HF最大批次大小。")

    main(args)
