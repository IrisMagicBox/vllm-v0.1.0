<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/source/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="docs/source/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
简单、快速、廉价的 LLM 推理与部署
</h3>

<p align="center">
| <a href="https://vllm.readthedocs.io/en/latest/"><b>文档</b></a> | <a href="https://vllm.ai"><b>博客</b></a> | <a href="https://github.com/vllm-project/vllm/discussions"><b>讨论</b></a> |

</p>

---

*最新消息* 🔥

- [2023/06] 我们正式发布了 vLLM！ 自四月中旬以来，vLLM 一直为 [LMSYS Vicuna 和 Chatbot Arena](https://chat.lmsys.org) 提供支持。 请查看我们的 [博客文章](https://vllm.ai)。

---

vLLM 是一个快速且易于使用的 LLM 推理和部署库。

vLLM 之所以快速是因为：

- 最先进的服务吞吐量
- 通过 **PagedAttention** 高效管理注意力键和值内存
- 动态批处理传入请求
- 优化的 CUDA 内核

vLLM 之所以灵活且易于使用是因为：

- 与流行的 HuggingFace 模型无缝集成
- 使用各种解码算法进行高吞吐量服务，包括 *并行采样*、*束搜索* 等
- 支持分布式推理的张量并行
- 流式输出
- 与 OpenAI 兼容的 API 服务器

vLLM 无缝支持许多 Huggingface 模型，包括以下架构：

- GPT-2 (`gpt2`, `gpt2-xl`, 等)
- GPTNeoX (`EleutherAI/gpt-neox-20b`, `databricks/dolly-v2-12b`, `stabilityai/stablelm-tuned-alpha-7b`, 等)
- LLaMA (`lmsys/vicuna-13b-v1.3`, `young-geng/koala`, `openlm-research/open_llama_13b`, 等)
- OPT (`facebook/opt-66b`, `facebook/opt-iml-max-30b`, 等)

使用 pip 安装 vLLM 或 [从源代码构建](docs/source/getting_started/installation.rst)：

```bash
pip install vllm
```

## 入门指南

访问我们的 [文档](https://vllm.readthedocs.io/en/latest/) 开始使用。
- [安装](docs/source/getting_started/installation.rst)
- [快速入门](docs/source/getting_started/quickstart.rst)
- [支持的模型](docs/source/models/supported_models.rst)
- [添加新模型](docs/source/models/adding_model.rst)

## 性能

vLLM 在吞吐量方面比 HuggingFace Transformers (HF) 快达 24 倍，比 Text Generation Inference (TGI) 快达 3.5 倍。
有关详细信息，请查看我们的 [博客文章](https://vllm.ai)。

<p align="center">
  <picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/source/assets/figures/perf_a10g_n1_dark.png">
  <img src="docs/source/assets/figures/perf_a10g_n1_light.png" width="45%">
  </picture>
  <picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/source/assets/figures/perf_a100_n1_dark.png">
  <img src="docs/source/assets/figures/perf_a100_n1_light.png" width="45%">
  </picture>
  <br>
  <em> 每个请求请求 1 个输出完成时的服务吞吐量。 </em>
</p>

<p align="center">
  <picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/source/assets/figures/perf_a10g_n3_dark.png">
  <img src="docs/source/assets/figures/perf_a10g_n3_light.png" width="45%">
  </picture>
  <picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/source/assets/figures/perf_a100_n3_dark.png">
  <img src="docs/source/assets/figures/perf_a100_n3_light.png" width="45%">
  </picture>  <br>
  <em> 每个请求请求 3 个输出完成时的服务吞吐量。 </em>
</p>

## 贡献

我们欢迎并重视任何贡献和合作。
请查看 [CONTRIBUTING.md](./CONTRIBUTING.md) 了解如何参与。
