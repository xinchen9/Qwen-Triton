#include <torch/extension.h>

torch::Tensor rope_tensor_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& cos,
    const torch::Tensor& sin);

torch::Tensor rope_tensor_forward(
    const torch::Tensor& x,
    const torch::Tensor& cos,
    const torch::Tensor& sin) {
  TORCH_CHECK(x.is_cuda(), "qwen_triton::rope_tensor_forward expects a CUDA input tensor");
  TORCH_CHECK(cos.is_cuda() && sin.is_cuda(), "qwen_triton::rope_tensor_forward expects CUDA cos/sin tensors");
  return rope_tensor_forward_cuda(x, cos, sin);
}

TORCH_LIBRARY(qwen_triton, m) {
  m.def("rope_tensor_forward(Tensor x, Tensor cos, Tensor sin) -> Tensor");
}

TORCH_LIBRARY_IMPL(qwen_triton, CUDA, m) {
  m.impl("rope_tensor_forward", &rope_tensor_forward);
}
