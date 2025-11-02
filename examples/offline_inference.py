from vllm import LLM, SamplingParams


# 示例提示。
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# 创建一个采样参数对象。
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# 创建一个LLM。
llm = LLM(model="/Users/liziang/code/QingCloud/python/vllm-study/models/Qwen2-0.5B-Instruct")
# 从提示生成文本。输出是一个包含提示、生成文本和其他信息的RequestOutput对象列表。
outputs = llm.generate(prompts, sampling_params)
# 打印输出。
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"提示: {prompt!r}, 生成文本: {generated_text!r}")
