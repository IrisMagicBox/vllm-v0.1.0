#include <torch/extension.h>

#include <map>
#include <vector>

void swap_blocks(
  torch::Tensor& src,
  torch::Tensor& dst,
  const std::map<int64_t, int64_t>& block_mapping);

void copy_blocks(
  std::vector<torch::Tensor>& key_caches,
  std::vector<torch::Tensor>& value_caches,
  const std::map<int64_t, std::vector<int64_t>>& block_mapping);

void reshape_and_cache(
  torch::Tensor& key,
  torch::Tensor& value,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& slot_mapping);

void gather_cached_kv(
  torch::Tensor& key,
  torch::Tensor& value,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& slot_mapping);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "swap_blocks",
    &swap_blocks,
    "将缓存块从源交换到目标");
  m.def(
    "copy_blocks",
    &copy_blocks,
    "将缓存块从源复制到目标");
  m.def(
    "reshape_and_cache",
    &reshape_and_cache,
    "重塑键和值张量并将它们缓存");
  m.def(
    "gather_cached_kv",
    &gather_cached_kv,
    "从缓存中收集键和值到连续的QKV张量中");
}
