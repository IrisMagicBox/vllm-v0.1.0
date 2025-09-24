import argparse

from vllm import EngineArgs, LLMEngine, SamplingParams


def main(args: argparse.Namespace):
    # 解析CLI参数并初始化引擎。
    engine_args = EngineArgs.from_cli_args(args)
    engine = LLMEngine.from_engine_args(engine_args)

    # 测试以下提示。
    test_prompts = [
        ("A robot may not injure a human being", SamplingParams()),
        ("To be or not to be,",
         SamplingParams(temperature=0.8, top_k=5, presence_penalty=0.2)),
        ("What is the meaning of life?",
         SamplingParams(n=2, best_of=5, temperature=0.8, top_p=0.95, frequency_penalty=0.1)),
        ("It is only with the heart that one can see rightly",
         SamplingParams(n=3, best_of=3, use_beam_search=True, temperature=0.0)),
    ]

    # 通过手动调用 `engine.step()` 来运行引擎。
    request_id = 0
    while True:
        # 为了测试迭代级调度，我们在每个步骤添加一个请求。
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            engine.add_request(str(request_id), prompt, sampling_params)
            request_id += 1

        request_outputs = engine.step()
        for request_output in request_outputs:
            if request_output.finished():
                print(request_output)

        if not (engine.has_unfinished_requests() or test_prompts):
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='直接使用LLMEngine类的演示')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    main(args)
