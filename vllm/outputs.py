from typing import Dict, List, Optional

from vllm.sequence import SequenceGroup, SequenceStatus


class CompletionOutput:
    """一个请求的一个完成输出的数据。

    Args:
        index: 请求中输出的索引。
        text: 生成的输出文本。
        token_ids: 生成的输出文本的token ID。
        cumulative_logprob: 生成的输出文本的累积对数概率。
        logprobs: 如果请求了logprobs，则为每个位置的最高概率词的对数概率。
        finish_reason: 序列完成的原因。
    """

    def __init__(
        self,
        index: int,
        text: str,
        token_ids: List[int],
        cumulative_logprob: float,
        logprobs: Optional[List[Dict[int, float]]],
        finish_reason: Optional[str] = None,
    ) -> None:
        self.index = index
        self.text = text
        self.token_ids = token_ids
        self.cumulative_logprob = cumulative_logprob
        self.logprobs = logprobs
        self.finish_reason = finish_reason

    def finished(self) -> bool:
        return self.finish_reason is not None

    def __repr__(self) -> str:
        return (f"CompletionOutput(index={self.index}, "
                f"text={self.text!r}, "
                f"token_ids={self.token_ids}, "
                f"cumulative_logprob={self.cumulative_logprob}, "
                f"logprobs={self.logprobs}, "
                f"finish_reason={self.finish_reason})")


class RequestOutput:
    """LLM请求的输出数据。

    Args:
        request_id: 请求的唯一ID。
        prompt: 请求的提示字符串。
        prompt_token_ids: 提示的token ID。
        outputs: 请求的输出序列。
    """
    def __init__(
        self,
        request_id: str,
        prompt: str,
        prompt_token_ids: List[int],
        outputs: List[CompletionOutput],
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.outputs = outputs

    @classmethod
    def from_seq_group(cls, seq_group: SequenceGroup) -> "RequestOutput":
        # 获取top-n序列。
        n = seq_group.sampling_params.n
        seqs = seq_group.get_seqs()
        assert n <= len(seqs)
        sorted_seqs = sorted(
            seqs, key=lambda seq: seq.get_cumulative_logprob(), reverse=True)
        top_n_seqs = sorted_seqs[:n]

        # 创建输出。
        outputs: List[CompletionOutput] = []
        for seq in top_n_seqs:
            logprobs = seq.output_logprobs
            if seq_group.sampling_params.logprobs is None:
                # NOTE: 我们需要留意这种情况，因为序列
                # 总是具有采样token的logprobs，即使
                # 没有请求logprobs。
                logprobs = {}
            finshed_reason = SequenceStatus.get_finished_reason(seq.status)
            output = CompletionOutput(seqs.index(seq), seq.output_text,
                                      seq.get_output_token_ids(),
                                      seq.get_cumulative_logprob(), logprobs,
                                      finshed_reason)
            outputs.append(output)

        # 序列组中的每个序列都应该具有相同的提示。
        prompt = top_n_seqs[0].prompt
        prompt_token_ids = top_n_seqs[0].data.prompt_token_ids
        return cls(seq_group.request_id, prompt, prompt_token_ids, outputs)

    def __repr__(self) -> str:
        return (f"RequestOutput(request_id={self.request_id}, "
                f"prompt={self.prompt!r}, "
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"outputs={self.outputs})")

    def finished(self) -> bool:
        return all(output.finished() for output in self.outputs)
