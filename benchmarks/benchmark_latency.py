"""基准测试处理单个请求批次的延迟。"""
import argparse
import time

import numpy as np
import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams


def main(args: argparse.Namespace):
    print(args)

    # 尽可能在单个批次中处理所有请求。
    # 注意(woosuk): 如果请求无法在单个批次中处理，
    # 引擎将自动在多个批次中处理请求。
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=args.batch_size * args.input_len,
    )

    sampling_params = SamplingParams(
        n=args.n,
        temperature=0.0 if args.use_beam_search else 1.0,
        top_p=1.0,
        use_beam_search=args.use_beam_search,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    print(sampling_params)
    dummy_prompt_token_ids = [[0] * args.input_len] * args.batch_size

    def run_to_completion(profile: bool = False):
        if profile:
            torch.cuda.cudart().cudaProfilerStart()
        start_time = time.time()

        llm.generate(prompt_token_ids=dummy_prompt_token_ids,
                     sampling_params=sampling_params,
                     use_tqdm=False)

        end_time = time.time()
        latency = end_time - start_time
        if profile:
            torch.cuda.cudart().cudaProfilerStop()
        return latency

    print("预热中...")
    run_to_completion(profile=False)

    # 基准测试。
    latencies = []
    for _ in tqdm(range(args.num_iters), desc="性能分析迭代"):
        latencies.append(run_to_completion(profile=False))
    print(f'平均延迟: {np.mean(latencies)} 秒')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='基准测试处理单个请求批次直到完成的延迟。')
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n', type=int, default=1,
                        help='每个提示生成的序列数。')
    parser.add_argument('--use-beam-search', action='store_true')
    parser.add_argument('--num-iters', type=int, default=3,
                        help='运行的迭代次数。')
    args = parser.parse_args()
    main(args)
