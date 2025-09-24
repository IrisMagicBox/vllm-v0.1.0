.. _adding_a_new_model:

添加新模型
==================

本文档提供了将 `HuggingFace Transformers <https://github.com/huggingface/transformers>`_ 模型集成到 vLLM 的高级指南。

.. note::
    添加新模型的复杂性很大程度上取决于模型的架构。
    如果模型与 vLLM 中的现有模型共享相似的架构，则此过程相当简单。
    但是，对于包含新操作符（例如，新的注意力机制）的模型，此过程可能稍微复杂一些。

.. tip::
    如果在将模型集成到 vLLM 时遇到问题，请随时在我们的 `GitHub <https://github.com/vllm-project/vllm/issues>`_ 仓库中提交问题。
    我们很乐意为您提供帮助！


0. Fork vLLM 仓库
--------------------------------

首先 Fork 我们的 `GitHub <https://github.com/vllm-project/vllm/>`_ 仓库，然后 :ref:`从源码构建 <build_from_source>`。
这使您能够修改代码库并测试您的模型。


1. 引入您的模型代码
------------------------

从 HuggingFace Transformers 仓库克隆 PyTorch 模型代码并将其放入 `vllm/model_executor/models <https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models>`_ 目录。
例如，vLLM 的 `OPT 模型 <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/opt.py>`_ 适配自 HuggingFace 的 `modeling_opt.py <https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py>`_ 文件。

.. warning::
    复制模型代码时，请确保查看并遵守代码的版权和许可条款。


2. 重写 :code:`forward` 方法
--------------------------------------

接下来，您需要按照以下步骤重写模型的 :code:`forward` 方法：

1. 删除任何不必要的代码，例如仅用于训练的代码。
2. 更改输入参数：

.. code-block:: diff

    def forward(
        self,
        input_ids: torch.Tensor,
    -    attention_mask: Optional[torch.Tensor] = None,
    -    position_ids: Optional[torch.LongTensor] = None,
    -    past_key_values: Optional[List[torch.FloatTensor]] = None,
    -    inputs_embeds: Optional[torch.FloatTensor] = None,
    -    labels: Optional[torch.LongTensor] = None,
    -    use_cache: Optional[bool] = None,
    -    output_attentions: Optional[bool] = None,
    -    output_hidden_states: Optional[bool] = None,
    -    return_dict: Optional[bool] = None,
    -) -> Union[Tuple, CausalLMOutputWithPast]:
    +    positions: torch.Tensor,
    +    kv_caches: List[KVCache],
    +    input_metadata: InputMetadata,
    +    cache_events: Optional[List[torch.cuda.Event]],
    +) -> Dict[int, SequenceOutputs]:

3. 考虑到 :code:`input_ids` 和 :code:`positions` 现在是展平的张量，更新代码。
4. 根据模型的架构，用 :code:`GPTPagedAttention` 或 :code:`GPTNeoXPagedAttention` 替换注意力操作。

.. note::
    目前，vLLM 支持基本的多头注意力机制及其带旋转位置嵌入的变体。
    如果您的模型采用不同的注意力机制，则需要在 vLLM 中实现新的注意力层。


3. （可选）实现张量并行支持
--------------------------------------------------

如果您的模型太大，无法放入单个 GPU，则可以使用张量并行来管理它。
为此，将模型的线性和嵌入层替换为其张量并行版本。
对于嵌入层，您可以简单地将 :code:`nn.Embedding` 替换为 :code:`VocabParallelEmbedding`。
对于线性层，您应该使用 :code:`RowParallelLinear` 或 :code:`ColumnParallelLinear`。
通常，:code:`ColumnParallelLinear` 用于 QKV 线性层和 MLP 块的第一个线性层。
对于其余线性层，使用 :code:`RowParallelLinear`。


4. 实现权重加载逻辑
-------------------------------------

您现在需要在 :code:`*ForCausalLM` 类中实现 :code:`load_weights` 方法。
此方法应从 HuggingFace 的检查点文件加载权重并将其分配给模型中的相应层。
虽然对于大多数层来说这个过程很简单，但张量并行层需要一些额外的注意，因为它们的权重应该被分割到多个 GPU 上。


5. 注册您的模型
----------------------

最后，将您的 :code:`*ForCausalLM` 类包含在 `vllm/model_executor/models/__init__.py <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/__init__.py>`_ 中，并将其注册到 `vllm/model_executor/model_loader.py <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/model_loader.py>`_ 中的 :code:`_MODEL_REGISTRY`。
