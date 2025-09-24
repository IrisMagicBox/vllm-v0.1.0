"""基准测试在线服务吞吐量。

在服务器端，运行以下命令之一：
    (vLLM 后端)
    python -m vllm.entrypoints.api_server \
        --model <your_model> --swap-space 16 \
        --disable-log-requests

    (TGI 后端)
    ./launch_hf_server.sh <your_model>

在客户端，运行：
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --tokenizer <your_model> --dataset <target_dataset> \
        --request-rate <request_rate>
"""
import argparse
import asyncio
import json
import random
import time
from typing import AsyncGenerator, List, Tuple

import aiohttp
import numpy as np
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase

# (提示长度，输出长度，延迟)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []


def get_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    config = AutoConfig.from_pretrained(model_name)
    if config.model_type == "llama":
        # 潜在的protobuf错误的解决方法。
        model_name = "hf-internal-testing/llama-tokenizer"
    return AutoTokenizer.from_pretrained(model_name)


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    # 加载数据集。
    with open(dataset_path) as f:
        dataset = json.load(f)
    # 过滤掉少于2轮对话的数据。
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
            # 剪除过短的序列。
            # 这是因为当输入或输出长度太短时，TGI会导致错误。
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # 剪除过长的序列。
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # 采样请求。
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # 如果请求率是无穷大，则我们不需要等待。
            continue
        # 从指数分布中采样请求间隔。
        interval = np.random.exponential(1.0 / request_rate)
        # 下一个请求将在间隔后发送。
        await asyncio.sleep(interval)


async def send_request(
    backend: str,
    api_url: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    best_of: int,
    use_beam_search: bool,
) -> None:
    request_start_time = time.time()

    headers = {"User-Agent": "Benchmark Client"}
    if backend == "vllm":
        pload = {
            "prompt": prompt,
            "n": 1,
            "best_of": best_of,
            "use_beam_search": use_beam_search,
            "temperature": 0.0 if use_beam_search else 1.0,
            "top_p": 1.0,
            "max_tokens": output_len,
            "ignore_eos": True,
            "stream": False,
        }
    elif backend == "tgi":
        assert not use_beam_search
        params = {
            "best_of": best_of,
            "max_new_tokens": output_len,
            "do_sample": True,
        }
        pload = {
            "inputs": prompt,
            "parameters": params,
        }
    else:
        raise ValueError(f"Unknown backend: {backend}")

    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.post(api_url, headers=headers, json=pload) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            output = json.loads(output)

            # 如果请求失败，则重新发送请求。
            if "error" not in output:
                break

    request_end_time = time.time()
    request_latency = request_end_time - request_start_time
    REQUEST_LATENCY.append((prompt_len, output_len, request_latency))


async def benchmark(
    backend: str,
    api_url: str,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
) -> None:
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        task = asyncio.create_task(send_request(backend, api_url, prompt,
                                                prompt_len, output_len,
                                                best_of, use_beam_search))
        tasks.append(task)
    await asyncio.gather(*tasks)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"http://{args.host}:{args.port}/generate"
    tokenizer = get_tokenizer(args.tokenizer)
    input_requests = sample_requests(args.dataset, args.num_prompts, tokenizer)

    benchmark_start_time = time.time()
    asyncio.run(benchmark(args.backend, api_url, input_requests, args.best_of,
                          args.use_beam_search, args.request_rate))
    benchmark_end_time = time.time()
    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"总时间: {benchmark_time:.2f} 秒")
    print(f"吞吐量: {args.num_prompts / benchmark_time:.2f} 请求/秒")

    # 计算延迟统计信息。
    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    print(f"平均延迟: {avg_latency:.2f} 秒")
    avg_per_token_latency = np.mean([
        latency / (prompt_len + output_len)
        for prompt_len, output_len, latency in REQUEST_LATENCY
    ])
    print(f"每个token的平均延迟: {avg_per_token_latency:.2f} 秒")
    avg_per_output_token_latency = np.mean([
        latency / output_len
        for _, output_len, latency in REQUEST_LATENCY
    ])
    print("每个输出token的平均延迟: "
          f"{avg_per_output_token_latency:.2f} 秒")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="基准测试在线服务吞吐量。")
    parser.add_argument("--backend", type=str, default="vllm",
                        choices=["vllm", "tgi"])
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--dataset", type=str, required=True,
                        help="数据集路径。")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="分词器的名称或路径。")
    parser.add_argument("--best-of", type=int, default=1,
                        help="为每个提示生成 `best_of` 个序列并返回最佳序列。")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=1000,
                        help="要处理的提示数量。")
    parser.add_argument("--request-rate", type=float, default=float("inf"),
                        help="每秒请求数。如果是无穷大，则所有请求都在时间0发送。"
                             "否则，我们使用泊松过程来合成请求到达时间。")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
