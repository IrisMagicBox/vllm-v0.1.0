.. _quickstart:

快速开始
========

本指南展示了如何使用 vLLM 来：

* 对数据集运行离线批处理推理；
* 为大语言模型构建API服务器；
* 启动与OpenAI兼容的API服务器。

在继续本指南之前，请确保完成 :ref:`安装说明 <installation>`。

离线批处理推理
--------------

我们首先展示一个使用vLLM进行离线批处理推理的示例。换句话说，我们使用vLLM为输入提示列表生成文本。

从vLLM中导入``LLM``和``SamplingParams``。``LLM``类是使用vLLM引擎进行离线推理的主要类。``SamplingParams``类指定采样过程的参数。

.. code-block:: python

    from vllm import LLM, SamplingParams

定义输入提示列表和生成的采样参数。采样温度设置为0.8，核采样概率设置为0.95。有关采样参数的更多信息，请参阅 `类定义 <https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py>`_。

.. code-block:: python

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

使用``LLM``类和 `OPT-125M 模型 <https://arxiv.org/abs/2205.01068>`_ 初始化vLLM的离线推理引擎。支持的模型列表可以在 :ref:`支持的模型 <supported_models>` 中找到。

.. code-block:: python

    llm = LLM(model="facebook/opt-125m")

调用``llm.generate``生成输出。这会将输入提示添加到vLLM引擎的等待队列中，并执行vLLM引擎以高吞吐量生成输出。输出作为``RequestOutput``对象列表返回，其中包括所有输出token。

.. code-block:: python

    outputs = llm.generate(prompts, sampling_params)

    # 打印输出。
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


代码示例也可以在 `examples/offline_inference.py <https://github.com/vllm-project/vllm/blob/main/examples/offline_inference.py>`_ 中找到。


API 服务器
----------

vLLM可以部署为LLM服务。我们提供了一个示例 `FastAPI <https://fastapi.tiangolo.com/>`_ 服务器。查看 `vllm/entrypoints/api_server.py <https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/api_server.py>`_ 了解服务器实现。服务器使用``AsyncLLMEngine``类来支持传入请求的异步处理。

启动服务器：

.. code-block:: console

    $ python -m vllm.entrypoints.api_server

默认情况下，此命令在``http://localhost:8000``启动带有OPT-125M模型的服务器。

在shell中查询模型：

.. code-block:: console

    $ curl http://localhost:8000/generate \
    $     -d '{
    $         "prompt": "San Francisco is a",
    $         "use_beam_search": true,
    $         "n": 4,
    $         "temperature": 0
    $     }'

查看 `examples/api_client.py <https://github.com/vllm-project/vllm/blob/main/examples/api_client.py>`_ 了解更详细的客户端示例。

与OpenAI兼容的服务器
--------------------

vLLM可以部署为模仿OpenAI API协议的服务器。这允许vLLM用作使用OpenAI API的应用程序的直接替代品。

启动服务器：

.. code-block:: console

    $ python -m vllm.entrypoints.openai.api_server \
    $     --model facebook/opt-125m

默认情况下，它在``http://localhost:8000``启动服务器。您可以使用``--host``和``--port``参数指定地址。服务器当前一次托管一个模型（上述命令中的OPT-125M）并实现 `列出模型 <https://platform.openai.com/docs/api-reference/models/list>`_ 和 `创建完成 <https://platform.openai.com/docs/api-reference/completions/create>`_ 端点。我们正在积极添加对更多端点的支持。

此服务器可以使用与OpenAI API相同的格式进行查询。例如，列出模型：

.. code-block:: console

    $ curl http://localhost:8000/v1/models

使用输入提示查询模型：

.. code-block:: console

    $ curl http://localhost:8000/v1/completions \
    $     -H "Content-Type: application/json" \
    $     -d '{
    $         "model": "facebook/opt-125m",
    $         "prompt": "San Francisco is a",
    $         "max_tokens": 7,
    $         "temperature": 0
    $     }'

由于此服务器与OpenAI API兼容，您可以将其用作使用OpenAI API的任何应用程序的直接替代品。例如，查询服务器的另一种方法是通过``openai`` python包：

.. code-block:: python

    import openai
    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai.api_key = "EMPTY"
    openai.api_base = "http://localhost:8000/v1"
    completion = openai.Completion.create(model="facebook/opt-125m",
                                          prompt="San Francisco is a")
    print("Completion result:", completion)

有关更详细的客户端示例，请参阅 `examples/openai_client.py <https://github.com/vllm-project/vllm/blob/main/examples/openai_client.py>`_。
