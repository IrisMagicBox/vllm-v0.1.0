.. _installation:

安装
====

vLLM是一个包含C++和CUDA代码的Python库。
这些额外的代码需要在用户的机器上进行编译。

要求
----

* 操作系统: Linux
* Python: 3.8 或更高版本
* CUDA: 11.0 -- 11.8
* GPU: 计算能力 7.0 或更高 (例如，V100, T4, RTX20xx, A100, L4, 等)

.. note::
    目前为止，vLLM 不支持 CUDA 12。
    如果您使用 Hopper 或 Lovelace GPU，请使用 CUDA 11.8 而不是 CUDA 12。

.. tip::
    如果您在安装 vLLM 时遇到问题，我们建议使用 NVIDIA PyTorch Docker 镜像。

    .. code-block:: console

        $ # Pull the Docker image with CUDA 11.8.
        $ docker run --gpus all -it --rm --shm-size=8g nvcr.io/nvidia/pytorch:22.12-py3

    Inside the Docker container, please execute :code:`pip uninstall torch` before installing vLLM.

使用 pip 安装
----------------

您可以使用 pip 安装 vLLM：

.. code-block:: console

    $ # (可选) 创建新的 conda 环境。
    $ conda create -n myenv python=3.8 -y
    $ conda activate myenv

    $ # 安装 vLLM。
    $ pip install vllm  # 这可能需要 5-10 分钟。


.. _build_from_source:

从源码构建
----------

您也可以从源码构建和安装 vLLM：

.. code-block:: console

    $ git clone https://github.com/vllm-project/vllm.git
    $ cd vllm
    $ pip install -e .  # 这可能需要 5-10 分钟。
