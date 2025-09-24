.. _supported_models:

支持的模型
================

vLLM 支持 `HuggingFace Transformers <https://huggingface.co/models>`_ 中的各种生成式 Transformer 模型。
以下是当前 vLLM 支持的模型架构列表。
对于每个架构，我们列出了一些使用它的流行模型。

.. list-table::
  :widths: 25 25 50
  :header-rows: 1

  * - 架构
    - 模型
    - 例如 HuggingFace 模型
  * - :code:`GPT2LMHeadModel`
    - GPT-2
    - :code:`gpt2`, :code:`gpt2-xl`, 等
  * - :code:`GPTNeoXForCausalLM`
    - GPT-NeoX, Pythia, OpenAssistant, Dolly V2, StableLM
    - :code:`EleutherAI/gpt-neox-20b`, :code:`EleutherAI/pythia-12b`, :code:`OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5`, :code:`databricks/dolly-v2-12b`, :code:`stabilityai/stablelm-tuned-alpha-7b`, 等
  * - :code:`LlamaForCausalLM`
    - LLaMA, Vicuna, Alpaca, Koala, Guanaco
    - :code:`openlm-research/open_llama_13b`, :code:`lmsys/vicuna-13b-v1.3`, :code:`young-geng/koala`, :code:`JosephusCheung/Guanaco`, 等
  * - :code:`OPTForCausalLM`
    - OPT, OPT-IML
    - :code:`facebook/opt-66b`, :code:`facebook/opt-iml-max-30b`, 等

如果您的模型使用以上模型架构之一，则可以无缝地使用 vLLM 运行您的模型。
否则，请参阅 :ref:`添加新模型 <adding_a_new_model>` 了解如何实现对您的模型的支持。
或者，您可以在我们的 `GitHub <https://github.com/vllm-project/vllm/issues>`_ 项目中提交问题。

.. tip::
    检查您的模型是否受支持的最简单方法是运行以下程序：

    .. code-block:: python

        from vllm import LLM

        llm = LLM(model=...)  # 您的模型名称或路径
        output = llm.generate("Hello, my name is")
        print(output)

    如果 vLLM 成功生成文本，则表明您的模型受支持。
