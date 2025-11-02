from typing import List, Optional, Union

from tqdm import tqdm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import Counter


class LLM:
    """一个用于根据给定提示和采样参数生成文本的大语言模型。

    该类包括一个分词器、一个语言模型（可能分布在多个GPU上），
    以及为中间状态（也称为KV缓存）分配的GPU内存空间。给定一批提示和采样参数，
    该类使用智能批处理机制和高效内存管理从模型生成文本。

    注意：该类用于离线推理。对于在线服务，请使用 `AsyncLLMEngine` 类。
    注意：有关参数的完整列表，请参见 `EngineArgs`。

    Args:
        model: HuggingFace Transformers 模型的名称或路径。
        tensor_parallel_size: 使用张量并行进行分布式执行时要使用的GPU数量。
        dtype: 模型权重和激活的数据类型。目前，我们支持 `float32`、`float16` 和 `bfloat16`。如果为 `auto`，我们使用
            模型配置文件中指定的 `torch_dtype` 属性。
            但是，如果配置中的 `torch_dtype` 为 `float32`，我们将
            改为使用 `float16`。
        seed: 用于初始化采样随机数生成器的种子。
    """

    def __init__(
        self,
        model: str,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        seed: int = 0,
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        engine_args = EngineArgs(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            seed=seed,
            **kwargs,
        )
        #初始化时就已经把模型的各个配置（分词器、模型结构、分布式参数、批处理等等）加载完成
        self.llm_engine = LLMEngine.from_engine_args(engine_args)
        self.request_counter = Counter()

    def get_tokenizer(
        self,
    ) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        return self.llm_engine.tokenizer

    def generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[SamplingParams] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
    ) -> List[RequestOutput]:
        """为输入提示生成补全。

        注意：该类会自动批处理给定的提示，同时考虑内存约束。为了获得最佳性能，
        请将所有提示放入单个列表中并将其传递给此方法。

        Args:
            prompts: 用于生成补全的提示列表。
            sampling_params: 文本生成的采样参数。如果
                为None，我们使用默认的采样参数。
            prompt_token_ids: 提示的token ID列表。如果为None，我们
                使用分词器将提示转换为token ID。
            use_tqdm: 是否使用tqdm显示进度条。

        Returns:
            包含生成补全的 `RequestOutput` 对象列表，顺序与输入提示相同。
        """
        if prompts is None and prompt_token_ids is None:
            raise ValueError("Either prompts or prompt_token_ids must be "
                             "provided.")
        if isinstance(prompts, str):
            # 将单个提示转换为列表。
            prompts = [prompts]
        if prompts is not None and prompt_token_ids is not None:
            if len(prompts) != len(prompt_token_ids):
                raise ValueError("提示和提示token ID的长度必须相同。")
        if sampling_params is None:
            # 使用默认采样参数。
            sampling_params = SamplingParams()

        # 将请求添加到引擎。
        if prompts is not None:
            num_requests = len(prompts)
        else:
            num_requests = len(prompt_token_ids)
        # 将prompts列表中的prompt放置到_add_request
        # prompt_token_ids 是一个可选的参数，用于直接指定提示（prompt）的token ID列表，而不是通过文本来表示，当你不希望vLLM自动对文本进行分词时，可以直接提供已经分好词的token ID，这样可以避免重复分词，提高效率
        for i in range(num_requests):
            prompt = prompts[i] if prompts is not None else None
            if prompt_token_ids is None:
                token_ids = None
            else:
                token_ids = prompt_token_ids[i]
            self._add_request(prompt, sampling_params, token_ids)
        return self._run_engine(use_tqdm)

    def _add_request(
        self,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]],
    ) -> None:
        # 默认从 0 开始计数，每次调用 next(counter) 时，会返回当前计数值并将计数器加 1，确保了每个请求都有一个唯一的 ID
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(request_id, prompt, sampling_params,
                                    prompt_token_ids)

    def _run_engine(self, use_tqdm: bool) -> List[RequestOutput]:
        # 初始化tqdm。
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(total=num_requests, desc="已处理的提示")
        # 运行引擎。
        outputs: List[RequestOutput] = []
        while self.llm_engine.has_unfinished_requests():
            #调试查看这里输出的是什么？
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished():
                    outputs.append(output)
                    if use_tqdm:
                        pbar.update(1)
        if use_tqdm:
            pbar.close()
        return outputs
