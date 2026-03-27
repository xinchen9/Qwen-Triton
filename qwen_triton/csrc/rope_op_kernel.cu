#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAMacros.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

template <typename scalar_t>
__global__ void rope_tensor_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ cos,
    const scalar_t* __restrict__ sin,
    scalar_t* __restrict__ out,
    int64_t num_heads,
    int64_t seq_len,
    int64_t hidden_size,
    int64_t rotary_dim) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t half_dim = rotary_dim / 2;
  const int64_t rows_per_batch = num_heads * seq_len;
  const int64_t batch_idx = row / rows_per_batch;
  const int64_t rem = row % rows_per_batch;
  const int64_t seq_idx = rem % seq_len;
  const int64_t x_offset = row * hidden_size;
  const int64_t rope_offset = (batch_idx * seq_len + seq_idx) * rotary_dim;

  for (int64_t idx = threadIdx.x; idx < half_dim; idx += blockDim.x) {
    const float x1 = static_cast<float>(x[x_offset + idx]);
    const float x2 = static_cast<float>(x[x_offset + idx + half_dim]);
    const float c = static_cast<float>(cos[rope_offset + idx]);
    const float s = static_cast<float>(sin[rope_offset + idx]);
    out[x_offset + idx] = static_cast<scalar_t>(x1 * c - x2 * s);
    out[x_offset + idx + half_dim] = static_cast<scalar_t>(x2 * c + x1 * s);
  }
}

}  // namespace

torch::Tensor rope_tensor_forward_cuda(
    const torch::Tensor& x,
    const torch::Tensor& cos,
    const torch::Tensor& sin) {
  TORCH_CHECK(x.dim() == 4, "qwen_triton::rope_tensor_forward expects x to have shape [batch, heads, seq, hidden]");
  TORCH_CHECK(cos.dim() == 3 && sin.dim() == 3, "qwen_triton::rope_tensor_forward expects cos/sin to have shape [batch, seq, rotary_dim]");
  TORCH_CHECK(cos.sizes() == sin.sizes(), "qwen_triton::rope_tensor_forward expects cos and sin to have identical shapes");
  TORCH_CHECK(x.size(0) == cos.size(0), "qwen_triton::rope_tensor_forward batch dimension mismatch");
  TORCH_CHECK(x.size(2) == cos.size(1), "qwen_triton::rope_tensor_forward sequence dimension mismatch");
  TORCH_CHECK(cos.size(2) % 2 == 0, "qwen_triton::rope_tensor_forward expects an even rotary dimension");
  TORCH_CHECK(cos.size(2) <= x.size(3), "qwen_triton::rope_tensor_forward rotary dimension exceeds hidden size");
  TORCH_CHECK(x.scalar_type() == cos.scalar_type(), "qwen_triton::rope_tensor_forward expects x and cos to share a dtype");
  TORCH_CHECK(x.scalar_type() == sin.scalar_type(), "qwen_triton::rope_tensor_forward expects x and sin to share a dtype");

  const auto device = x.device();
  c10::cuda::CUDAGuard device_guard(device);

  auto x_contig = x.contiguous();
  auto cos_contig = cos.contiguous();
  auto sin_contig = sin.contiguous();
  auto out = x_contig.clone();

  const auto num_rows = x_contig.size(0) * x_contig.size(1) * x_contig.size(2);
  const auto rotary_dim = cos_contig.size(2);
  if (num_rows == 0 || rotary_dim == 0) {
    return out;
  }

  int threads = 1;
  while (threads < rotary_dim / 2 && threads < 256) {
    threads *= 2;
  }

  const auto stream = at::cuda::getCurrentCUDAStream(device.index());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      x_contig.scalar_type(),
      "rope_tensor_forward_cuda",
      [&] {
        rope_tensor_forward_kernel<scalar_t><<<num_rows, threads, 0, stream>>>(
            x_contig.data_ptr<scalar_t>(),
            cos_contig.data_ptr<scalar_t>(),
            sin_contig.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            x_contig.size(1),
            x_contig.size(2),
            x_contig.size(3),
            rotary_dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
