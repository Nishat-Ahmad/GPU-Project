import torch
from torch.utils.cpp_extension import load

print("Compiling CUDA code... (This might take 1-2 minutes the very first time!)")

# 1. Compile and load the custom CUDA extension (JIT)
dummy_module = load(
    name="dummy_ext",
    sources=["dummy.cu"],
    verbose=True,
    extra_cflags=['/Zc:preprocessor'],
    extra_cuda_cflags=['-Xcompiler', '/Zc:preprocessor', '-diag-suppress', '3189']
)
print("Compilation successful!\n")

# 2. Test the extension
cpu_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
gpu_tensor = cpu_tensor.cuda()

result_gpu = dummy_module.multiply_by_two(gpu_tensor)
result_cpu = result_gpu.cpu()

print(f"Original Tensor: {cpu_tensor}")
print(f"Result Tensor:   {result_cpu}")

if torch.allclose(result_cpu, cpu_tensor * 2):
    print("\nThe Bridge is working perfectly!")
else:
    print("\nSomething went wrong with the math.")
