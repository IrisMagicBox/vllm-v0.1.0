import openai

# 修改OpenAI的API密钥和API基础URL以使用vLLM的API服务器。
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"
model = "facebook/opt-125m"

# 测试列出模型API
models = openai.Model.list()
print("模型:", models)

# 测试补全API
stream = True
completion = openai.Completion.create(
    model=model, prompt="A robot may not injure a human being", echo=False, n=2,
    best_of=3, stream=stream, logprobs=3)

# 打印补全结果
if stream:
    for c in completion:
        print(c)
else:
    print("补全结果:", completion)
