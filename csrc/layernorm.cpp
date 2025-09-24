#include <torch/extension.h>

void rms_norm(
  torch::Tensor& out,
  torch::Tensor& input,
  torch::Tensor& weight,
  float epsilon);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "rms_norm",
    &rms_norm,
    "对输入张量应用均方根(RMS)归一化。");
}
