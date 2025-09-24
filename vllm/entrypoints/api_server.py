import argparse
import json
from typing import AsyncGenerator

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import Response, StreamingResponse
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

TIMEOUT_KEEP_ALIVE = 5 # 秒
TIMEOUT_TO_PREVENT_DEADLOCK = 1 # 秒
app = FastAPI()


@app.post("/generate")
async def generate(request: Request) -> Response:
    """为请求生成补全。

    请求应该是一个JSON对象，包含以下字段：
    - prompt: 用于生成的提示。
    - stream: 是否流式传输结果。
    - 其他字段: 采样参数（详情参见 `SamplingParams`）。
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()
    results_generator = engine.generate(prompt, sampling_params, request_id)

    # 流式传输情况
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [
                prompt + output.text
                for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    async def abort_request() -> None:
        await engine.abort(request_id)

    if stream:
        background_tasks = BackgroundTasks()
        # 如果客户端断开连接则中止请求。
        background_tasks.add_task(abort_request)
        return StreamingResponse(stream_results(), background=background_tasks)

    # 非流式传输情况
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # 如果客户端断开连接则中止请求。
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [
        prompt + output.text
        for output in final_output.outputs
    ]
    ret = {"text": text_outputs}
    return Response(content=json.dumps(ret))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost", help="主机名")
    parser.add_argument("--port", type=int, default=8000, help="端口号")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    uvicorn.run(app, host=args.host, port=args.port, log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
