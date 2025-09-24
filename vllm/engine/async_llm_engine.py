import asyncio
import time
from typing import Dict, List, Optional

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.engine.ray_utils import initialize_cluster, ray
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)

TIMEOUT_TO_PREVENT_DEADLOCK = 1 # seconds


class AsyncLLMEngine:
    """LLMEngine 的异步包装器。

    此类用于包装 LLMEngine 类以使其异步化。它使用 asyncio 创建一个后台循环，
    持续处理传入的请求。当等待队列中有请求时，generate 方法会触发 LLMEngine。
    generate 方法将输出从 LLMEngine 生成给调用者。

    注意：有关参数的完整列表，请参见 `LLMEngine`。

    Args:
        worker_use_ray: 是否对模型工作进程使用 Ray。分布式执行时必需。
            应与 `parallel_config.worker_use_ray` 相同。
        engine_use_ray: 是否将 LLMEngine 作为一个 Ray actor。如果是这样，
            异步前端将在一个单独的进程中执行，作为模型工作进程。
        log_requests: 是否记录请求日志。
        *args, *kwargs: LLMEngine 的参数。
    """
    def __init__(self, worker_use_ray: bool, engine_use_ray: bool,
                 log_requests: bool = True, *args, **kwargs) -> None:
        self.worker_use_ray = worker_use_ray
        self.engine_use_ray = engine_use_ray
        self.log_requests = log_requests
        if not self.engine_use_ray:
            engine_class = LLMEngine
        elif self.worker_use_ray:
            engine_class = ray.remote(num_cpus=0)(LLMEngine).remote
        else:
            engine_class = ray.remote(num_gpus=1)(LLMEngine).remote
        self.engine = engine_class(*args, **kwargs)
        # 请求 ID -> 请求输出。
        self.request_outputs: Dict[str, RequestOutput] = {}
        # 请求 ID -> 通知有新输出的事件。
        self.request_events: Dict[str, asyncio.Event] = {}
        self.is_engine_running = False
        self.kicking_request_id: Optional[str] = None

    async def engine_step(self, kicking_request_id: Optional[str] = None):
        """触发引擎处理等待的请求。"""
        self.is_engine_running = True
        self.kicking_request_id = kicking_request_id
        if self.engine_use_ray:
            request_outputs = await self.engine.step.remote()
        else:
            # 让出事件循环以允许其他协程运行
            # 当 is_engine_running 为 True 时。这允许引擎将新请求添加到队列中。
            await asyncio.sleep(0)
            request_outputs = self.engine.step()
        self.is_engine_running = False
        self.kicking_request_id = None

        # 通知等待的协程有新输出已就绪。
        for request_output in request_outputs:
            request_id = request_output.request_id
            self.request_outputs[request_id] = request_output
            self.request_events[request_id].set()

    async def generate(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        request_id: str,
        prompt_token_ids: Optional[List[int]] = None
    ) -> RequestOutput:
        """为请求生成输出。

        为请求生成输出。此方法是一个协程。它将请求添加到 LLMEngine 的等待队列中，
        并将输出从 LLMEngine 流式传输给调用者。

        Args:
            prompt: 提示字符串。如果提供了 prompt_token_ids，则可以为 None。
            sampling_params: 请求的采样参数。
            request_id: 请求的唯一 ID。
            prompt_token_ids: 提示的令牌 ID。如果为 None，我们使用
                分词器将提示转换为令牌 ID。

        Yields:
            LLMEngine 为请求生成的输出 `RequestOutput` 对象。
        """
        # 预处理请求。
        arrival_time = time.time()

        # 创建一个事件来通知我们 vLLM 引擎有新输出。
        request_event = asyncio.Event()
        self.request_events[request_id] = request_event

        if self.log_requests:
            logger.info(f"收到请求 {request_id}: "
                        f"提示: {prompt!r}, "
                        f"采样参数: {sampling_params}, "
                        f"提示令牌 ID: {prompt_token_ids}.")

        # 将请求添加到 vLLM 引擎的等待队列中。
        if self.engine_use_ray:
            await self.engine.add_request.remote(
                request_id, prompt, sampling_params,
                prompt_token_ids=prompt_token_ids,
                arrival_time=arrival_time)
        else:
            self.engine.add_request(
                request_id, prompt, sampling_params,
                prompt_token_ids=prompt_token_ids,
                arrival_time=arrival_time)

        # vLLM 引擎没有后台循环来持续处理传入请求。
        # 因此，我们需要不断触发引擎来处理请求。
        while True:
            if request_id not in self.request_events:
                # 请求已被中止。
                return

            # 如果引擎未运行，则触发引擎。
            if not self.is_engine_running:
                await self.engine_step(request_id)

            # 等待新输出。当序列组有可用输出时，group_event 将在 engine_step 中设置。
            # 添加了超时以防止死锁。
            try:
                await asyncio.wait_for(request_event.wait(),
                                       timeout=TIMEOUT_TO_PREVENT_DEADLOCK)
            except asyncio.TimeoutError:
                continue
            # 重置事件以等待下一个输出。
            request_event.clear()

            # 解码并返回新输出。
            request_output = self.request_outputs[request_id]
            yield request_output

            # 完成后，释放序列组的资源。
            if request_output.finished():
                if self.log_requests:
                    logger.info(f"完成请求 {request_id}.")

                del self.request_outputs[request_id]
                del self.request_events[request_id]
                # 如果引擎未运行，则触发引擎。这是为了防止引擎等待队列中
                # 仍有待执行的请求。
                if not self.is_engine_running:
                    await self.engine_step()
                break

    async def abort(self, request_id: str) -> None:
        """中止一个请求。

        中止提交的请求。如果请求已完成或找不到，则此方法将不执行任何操作。

        Args:
            request_id: 请求的唯一 ID。
        """
        if request_id not in self.request_events:
            # 请求已经完成或已被中止。
            return

        if self.log_requests:
            logger.info(f"中止请求 {request_id}.")

        if self.engine_use_ray:
            await self.engine.abort_request.remote(request_id)
        else:
            self.engine.abort_request(request_id)

        if request_id in self.request_events:
            del self.request_events[request_id]
        if request_id in self.request_outputs:
            del self.request_outputs[request_id]

        # 为防止在引擎运行时中止请求导致死锁。
        if self.kicking_request_id == request_id:
            self.is_engine_running = False
            self.kicking_request_id = None

    @classmethod
    def from_engine_args(cls, engine_args: AsyncEngineArgs) -> "AsyncLLMEngine":
        """根据引擎参数创建异步 LLM 引擎。"""
        # 创建引擎配置。
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]
        # 初始化集群。
        distributed_init_method, devices = initialize_cluster(
            parallel_config, engine_args.engine_use_ray)
        # 创建异步 LLM 引擎。
        engine = cls(engine_args.worker_use_ray,
                     engine_args.engine_use_ray,
                     not engine_args.disable_log_requests,
                     *engine_configs,
                     distributed_init_method, devices,
                     log_stats=not engine_args.disable_log_stats)
        return engine
