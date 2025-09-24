# 改编自 https://github.com/lm-sys/FastChat/blob/168ccc29d3f7edc50823016105c024fe2282732a/fastchat/serve/openai_api_server.py

import argparse
from http import HTTPStatus
import json
import time
from typing import AsyncGenerator, Dict, List, Optional

import fastapi
from fastapi import BackgroundTasks, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.tokenizer_utils import get_tokenizer
from vllm.entrypoints.openai.protocol import (
    CompletionRequest, CompletionResponse, CompletionResponseChoice,
    CompletionResponseStreamChoice, CompletionStreamResponse, ErrorResponse,
    LogProbs, ModelCard, ModelList, ModelPermission, UsageInfo)
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

TIMEOUT_KEEP_ALIVE = 5 # 秒

logger = init_logger(__name__)
served_model = None
app = fastapi.FastAPI()


def create_error_response(status_code: HTTPStatus,
                          message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, type="invalid_request_error").dict(),
        status_code=status_code.value
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return create_error_response(HTTPStatus.BAD_REQUEST, str(exc))


async def check_model(request) -> Optional[JSONResponse]:
    if request.model == served_model:
        return
    ret = create_error_response(
        HTTPStatus.NOT_FOUND,
        f"模型 `{request.model}` 不存在.",
    )
    return ret


@app.get("/v1/models")
async def show_available_models():
    """显示可用模型。目前我们只有一个模型。"""
    model_cards = [ModelCard(id=served_model, root=served_model,
                             permission=[ModelPermission()])]
    return ModelList(data=model_cards)


def create_logprobs(token_ids: List[int],
                    id_logprobs: List[Dict[int, float]],
                    initial_text_offset: int = 0) -> LogProbs:
    """创建 OpenAI 风格的 logprobs。"""
    logprobs = LogProbs()
    last_token_len = 0
    for token_id, id_logprob in zip(token_ids, id_logprobs):
        token = tokenizer.convert_ids_to_tokens(token_id)
        logprobs.tokens.append(token)
        logprobs.token_logprobs.append(id_logprob[token_id])
        if len(logprobs.text_offset) == 0:
            logprobs.text_offset.append(initial_text_offset)
        else:
            logprobs.text_offset.append(logprobs.text_offset[-1] + last_token_len)
        last_token_len = len(token)

        logprobs.top_logprobs.append(
            {tokenizer.convert_ids_to_tokens(i): p
             for i, p in id_logprob.items()})
    return logprobs


@app.post("/v1/completions")
async def create_completion(raw_request: Request):
    """类似于 OpenAI API 的 Completion API。

    请参阅 https://platform.openai.com/docs/api-reference/completions/create
    了解 API 规范。此 API 模仿 OpenAI Completion API。

    注意：目前我们不支持以下功能：
        - echo (因为 vLLM 引擎目前不支持获取 prompt tokens 的 logprobs)
        - suffix (我们目前支持的语言模型不支持 suffix)
        - logit_bias (vLLM 引擎将支持)
    """
    request = CompletionRequest(**await raw_request.json())
    logger.info(f"收到 completion 请求: {request}")

    error_check_ret = await check_model(request)
    if error_check_ret is not None:
        return error_check_ret

    if request.echo:
        # 我们不支持 echo，因为 vLLM 引擎目前不支持获取 prompt tokens 的 logprobs。
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "echo 目前不被支持")

    if request.suffix is not None:
        # 我们目前支持的语言模型不支持 suffix。
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                    "suffix 目前不被支持")

    if request.logit_bias is not None:
        # TODO: 在 vLLM 引擎中支持 logit_bias。
        return create_error_response(HTTPStatus.BAD_REQUEST,
                                     "logit_bias 目前不被支持")

    model_name = request.model
    request_id = f"cmpl-{random_uuid()}"
    prompt = request.prompt
    created_time = int(time.time())
    try:
        sampling_params = SamplingParams(
            n=request.n,
            best_of=request.best_of,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=request.stop,
            ignore_eos=request.ignore_eos,
            max_tokens=request.max_tokens,
            logprobs=request.logprobs,
            use_beam_search=request.use_beam_search,
        )
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))

    result_generator = engine.generate(prompt, sampling_params,
                                       request_id)

    # 类似于 OpenAI API，当 n != best_of 时，我们不流式传输结果。
    # 此外，使用 beam search 时不流式传输结果。
    stream = (request.stream and
              (request.best_of is None or request.n == request.best_of) and
              not request.use_beam_search)

    async def abort_request() -> None:
        await engine.abort(request_id)

    def create_stream_response_json(index: int,
                                    text: str,
                                    logprobs: Optional[LogProbs] = None,
                                    finish_reason: Optional[str] = None) -> str:
        choice_data = CompletionResponseStreamChoice(
            index=index,
            text=text,
            logprobs=logprobs,
            finish_reason=finish_reason,
        )
        response = CompletionStreamResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=[choice_data],
        )
        response_json = response.json(ensure_ascii=False)

        return response_json

    async def completion_stream_generator() -> AsyncGenerator[str, None]:
        previous_texts = [""] * request.n
        previous_num_tokens = [0] * request.n
        async for res in result_generator:
            res: RequestOutput
            for output in res.outputs:
                i = output.index
                delta_text = output.text[len(previous_texts[i]):]
                if request.logprobs is not None:
                    logprobs = create_logprobs(
                        output.token_ids[previous_num_tokens[i]:],
                        output.logprobs[previous_num_tokens[i]:],
                        len(previous_texts[i]))
                else:
                    logprobs = None
                previous_texts[i] = output.text
                previous_num_tokens[i] = len(output.token_ids)
                response_json = create_stream_response_json(
                    index=i,
                    text=delta_text,
                    logprobs=logprobs,
                )
                yield f"data: {response_json}\n\n"
                if output.finish_reason is not None:
                    logprobs = LogProbs() if request.logprobs is not None else None
                    response_json = create_stream_response_json(
                        index=i,
                        text="",
                        logprobs=logprobs,
                        finish_reason=output.finish_reason,
                    )
                    yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"

    # 流式响应
    if stream:
        background_tasks = BackgroundTasks()
        # 如果客户端断开连接，则中止请求。
        background_tasks.add_task(abort_request)
        return StreamingResponse(completion_stream_generator(),
                                 media_type="text/event-stream",
                                 background=background_tasks)

    # 非流式响应
    final_res: RequestOutput = None
    async for res in result_generator:
        if await raw_request.is_disconnected():
            # 如果客户端断开连接，则中止请求。
            await abort_request()
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "客户端断开连接")
        final_res = res
    assert final_res is not None
    choices = []
    for output in final_res.outputs:
        if request.logprobs is not None:
            logprobs = create_logprobs(output.token_ids, output.logprobs)
        else:
            logprobs = None
        choice_data = CompletionResponseChoice(
            index=output.index,
            text=output.text,
            logprobs=logprobs,
            finish_reason=output.finish_reason,
        )
        choices.append(choice_data)

    num_prompt_tokens = len(final_res.prompt_token_ids)
    num_generated_tokens = sum(len(output.token_ids)
                               for output in final_res.outputs)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = CompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )

    if request.stream:
        # 当用户请求流式传输但我们不流式传输时，我们仍需要返回单个事件的流式响应。
        response_json = response.json(ensure_ascii=False)
        async def fake_stream_generator() -> AsyncGenerator[str, None]:
            yield f"data: {response_json}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(fake_stream_generator(),
                                 media_type="text/event-stream")

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="vLLM OpenAI 兼容 RESTful API 服务器。"
    )
    parser.add_argument("--host", type=str, default="localhost", help="主机名")
    parser.add_argument("--port", type=int, default=8000, help="端口号")
    parser.add_argument(
        "--allow-credentials", action="store_true", help="允许凭据"
    )
    parser.add_argument(
        "--allowed-origins", type=json.loads, default=["*"], help="允许的来源"
    )
    parser.add_argument(
        "--allowed-methods", type=json.loads, default=["*"], help="允许的方法"
    )
    parser.add_argument(
        "--allowed-headers", type=json.loads, default=["*"], help="允许的请求头"
    )
    parser.add_argument("--served-model-name", type=str, default=None,
                        help="API 中使用的模型名称。如果未指定，"
                             "模型名称将与 huggingface 名称相同。")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )

    logger.info(f"args: {args}")

    served_model = args.served_model_name or args.model

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # 单独的 tokenizer 用于将 token ID 映射到字符串。
    tokenizer = get_tokenizer(args.model)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
