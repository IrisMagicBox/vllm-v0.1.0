import vllm.model_executor.parallel_utils.parallel_state
import vllm.model_executor.parallel_utils.tensor_parallel

# 将 parallel_state 别名为 mpu，这是它的旧名称
mpu = parallel_state

__all__ = [
    "parallel_state",
    "tensor_parallel",
]
