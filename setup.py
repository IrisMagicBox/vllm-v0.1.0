import io
import os
import re
import subprocess
from typing import List, Set

from packaging.version import parse, Version
import setuptools
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

ROOT_DIR = os.path.dirname(__file__)

# 编译器标志。
CXX_FLAGS = ["-g", "-O2", "-std=c++17"]
# TODO(woosuk): 我们应该使用 -O3 吗？
NVCC_FLAGS = ["-O2", "-std=c++17"]

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

if not torch.cuda.is_available():
    raise RuntimeError(
        f"Cannot find CUDA at CUDA_HOME: {CUDA_HOME}. "
        "CUDA must be available in order to build the package.")


def get_nvcc_cuda_version(cuda_dir: str) -> Version:
    """从nvcc获取CUDA版本。

    来源于 https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    nvcc_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                          universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


# 收集所有可用GPU的计算能力。
device_count = torch.cuda.device_count()
compute_capabilities: Set[int] = set()
for i in range(device_count):
    major, minor = torch.cuda.get_device_capability(i)
    if major < 7:
        raise RuntimeError(
            "不支持计算能力低于7.0的GPU。")
    compute_capabilities.add(major * 10 + minor)
# 如果没有可用的GPU，添加所有支持的计算能力。
if not compute_capabilities:
    compute_capabilities = {70, 75, 80, 86, 90}
# 将目标计算能力添加到NVCC标志中。
for capability in compute_capabilities:
    NVCC_FLAGS += ["-gencode", f"arch=compute_{capability},code=sm_{capability}"]

# 验证NVCC CUDA版本。
nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)
if nvcc_cuda_version < Version("11.0"):
    raise RuntimeError("构建包需要CUDA 11.0或更高版本。")
if 86 in compute_capabilities and nvcc_cuda_version < Version("11.1"):
    raise RuntimeError(
        "计算能力为8.6的GPU需要CUDA 11.1或更高版本。")
if 90 in compute_capabilities and nvcc_cuda_version < Version("11.8"):
    raise RuntimeError(
        "计算能力为9.0的GPU需要CUDA 11.8或更高版本。")

# 使用NVCC线程来并行化构建。
if nvcc_cuda_version >= Version("11.2"):
    num_threads = min(os.cpu_count(), 8)
    NVCC_FLAGS += ["--threads", str(num_threads)]

ext_modules = []

# 缓存操作。
cache_extension = CUDAExtension(
    name="vllm.cache_ops",
    sources=["csrc/cache.cpp", "csrc/cache_kernels.cu"],
    extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
)
ext_modules.append(cache_extension)

# 注意力内核。
attention_extension = CUDAExtension(
    name="vllm.attention_ops",
    sources=["csrc/attention.cpp", "csrc/attention/attention_kernels.cu"],
    extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
)
ext_modules.append(attention_extension)

# 位置编码内核。
positional_encoding_extension = CUDAExtension(
    name="vllm.pos_encoding_ops",
    sources=["csrc/pos_encoding.cpp", "csrc/pos_encoding_kernels.cu"],
    extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
)
ext_modules.append(positional_encoding_extension)

# 层归一化内核。
layernorm_extension = CUDAExtension(
    name="vllm.layernorm_ops",
    sources=["csrc/layernorm.cpp", "csrc/layernorm_kernels.cu"],
    extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
)
ext_modules.append(layernorm_extension)

# 激活内核。
activation_extension = CUDAExtension(
    name="vllm.activation_ops",
    sources=["csrc/activation.cpp", "csrc/activation_kernels.cu"],
    extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
)
ext_modules.append(activation_extension)


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def find_version(filepath: str):
    """从给定的文件路径中提取版本信息。

    来源于 https://github.com/ray-project/ray/blob/0b190ee1160eeca9796bc091e07eaebf4c85b511/python/setup.py
    """
    with open(filepath) as fp:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


def read_readme() -> str:
    """读取 README 文件。"""
    return io.open(get_path("README.md"), "r", encoding="utf-8").read()


def get_requirements() -> List[str]:
    """从 requirements.txt 获取 Python 包依赖。"""
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements


setuptools.setup(
    name="vllm",
    version=find_version(get_path("vllm", "__init__.py")),
    author="vLLM Team",
    license="Apache 2.0",
    description="一个高吞吐量和内存高效的 LLM 推理和服务引擎",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/vllm-project/vllm",
    project_urls={
        "Homepage": "https://github.com/vllm-project/vllm",
        "Documentation": "https://vllm.readthedocs.io/en/latest/",
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=setuptools.find_packages(
        exclude=("assets", "benchmarks", "csrc", "docs", "examples", "tests")),
    python_requires=">=3.8",
    install_requires=get_requirements(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
