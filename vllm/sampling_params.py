"""Sampling parameters for text generation."""
from typing import List, Optional, Union


class SamplingParams:
    """文本生成的采样参数。

    总体而言，我们遵循OpenAI文本补全API的采样参数
    (https://platform.openai.com/docs/api-reference/completions/create)。
    此外，我们还支持beam search，这是OpenAI不支持的。

    Args:
        n: 为给定提示返回的输出序列数量。
        best_of: 从提示生成的输出序列数量。
            从这些`best_of`序列中，返回top `n`序列。
            `best_of`必须大于或等于`n`。当`use_beam_search`为True时，这被视为
            beam宽度。默认情况下，`best_of`
            设置为`n`。
        presence_penalty: 基于新token是否出现在已生成的文本中来惩罚新token的浮点数。
            大于0的值鼓励模型使用新token，而小于0的值鼓励模型重复
            token。
        frequency_penalty: 基于新token在已生成的文本中的频率来惩罚新token的浮点数。
            大于0的值鼓励模型使用新token，而小于0的值鼓励模型
            重复token。
        temperature: 控制采样随机性的浮点数。较低
            的值使模型更加确定性，而较高的值使
            模型更加随机。零表示贪婪采样。
        top_p: 控制要考虑的最高token的累积概率的浮点数。
            必须在(0, 1]范围内。设置为1以考虑所有token。
        top_k: 控制要考虑的最高token数量的整数。设置
            为-1以考虑所有token。
        use_beam_search: 是否使用beam search代替采样。
        stop: 生成时停止的字符串列表。
            返回的输出将不包含停止字符串。
        ignore_eos: 是否忽略EOS token并在
            生成EOS token后继续生成token。
        max_tokens: 每个输出序列要生成的最大token数。
        logprobs: 每个输出token要返回的对数概率数。
    """

    def __init__(
        self,
        n: int = 1,
        best_of: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        use_beam_search: bool = False,
        stop: Union[str, List[str]] = [],
        ignore_eos: bool = False,
        max_tokens: int = 16,
        logprobs: Optional[int] = None,
    ) -> None:
        self.n = n
        self.best_of = best_of if best_of is not None else n
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.use_beam_search = use_beam_search
        self.stop = [stop] if isinstance(stop, str) else list(stop)
        self.ignore_eos = ignore_eos
        self.max_tokens = max_tokens
        self.logprobs = logprobs

        self._verify_args()
        if self.use_beam_search:
            self._verity_beam_search()
        elif self.temperature == 0.0:
            # 零温度表示贪婪采样。
            self._verify_greedy_sampling()

    def _verify_args(self) -> None:
        if self.n < 1:
            raise ValueError(f"n必须至少为1，得到 {self.n}.")
        if self.best_of < self.n:
            raise ValueError(f"best_of必须大于或等于n，"
                             f"得到 n={self.n} 和 best_of={self.best_of}.")
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise ValueError("presence_penalty必须在[-2, 2]范围内，得到 "
                             f"{self.presence_penalty}.")
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError("frequency_penalty必须在[-2, 2]范围内，得到 "
                             f"{self.frequency_penalty}.")
        if self.temperature < 0.0:
            raise ValueError(
                f"temperature必须为非负数，得到 {self.temperature}.")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p必须在(0, 1]范围内，得到 {self.top_p}.")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError(f"top_k必须为-1（禁用），或者至少为1，"
                             f"得到 {self.top_k}.")
        if self.max_tokens < 1:
            raise ValueError(
                f"max_tokens必须至少为1，得到 {self.max_tokens}.")
        if self.logprobs is not None and self.logprobs < 0:
            raise ValueError(
                f"logprobs必须为非负数，得到 {self.logprobs}.")

    def _verity_beam_search(self) -> None:
        if self.best_of == 1:
            raise ValueError("使用beam search时best_of必须大于1。"
                             f"得到 {self.best_of}.")
        if self.temperature > 0.0:
            raise ValueError("使用beam search时temperature必须为0。")
        if self.top_p < 1.0:
            raise ValueError("使用beam search时top_p必须为1。")
        if self.top_k != -1:
            raise ValueError("使用beam search时top_k必须为-1。")

    def _verify_greedy_sampling(self) -> None:
        if self.best_of > 1:
            raise ValueError("使用贪婪采样时best_of必须为1。"
                             f"得到 {self.best_of}.")
        if self.top_p < 1.0:
            raise ValueError("使用贪婪采样时top_p必须为1。")
        if self.top_k != -1:
            raise ValueError("使用贪婪采样时top_k必须为-1。")

    def __repr__(self) -> str:
        return (f"SamplingParams(n={self.n}, "
                f"best_of={self.best_of}, "
                f"presence_penalty={self.presence_penalty}, "
                f"frequency_penalty={self.frequency_penalty}, "
                f"temperature={self.temperature}, "
                f"top_p={self.top_p}, "
                f"top_k={self.top_k}, "
                f"use_beam_search={self.use_beam_search}, "
                f"stop={self.stop}, "
                f"ignore_eos={self.ignore_eos}, "
                f"max_tokens={self.max_tokens}, "
                f"logprobs={self.logprobs})")
