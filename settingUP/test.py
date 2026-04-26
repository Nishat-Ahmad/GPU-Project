import os

# --- THE NUCLEAR OPTION: FORCE WINDOWS PATHS ---
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"

# Force set both variables PyTorch might look for
os.environ["CUDA_HOME"] = cuda_path
os.environ["CUDA_PATH"] = cuda_path

# Force the compiler (nvcc) directly into the system PATH
os.environ["PATH"] = cuda_path + r"\bin;" + os.environ.get("PATH", "")
# -----------------------------------------------

# ONLY NOW are we allowed to import PyTorch
import torch
from torch.utils.cpp_extension import load

print("Compiling CUDA code... (This might take 1-2 minutes the very first time!)")

# 1. Compile and load the custom CUDA extension
# 1. Compile and load the custom CUDA extension
dummy_module = load(
    name="dummy_ext",
    sources=["dummy.cu"],
    verbose=True,
    extra_cflags=['/Zc:preprocessor'],
    # Add this line to hide warning 3189:
    extra_cuda_cflags=['-Xcompiler', '/Zc:preprocessor', '-diag-suppress', '3189']
)

print("Compilation successful!\n")

# 2. Create a normal PyTorch tensor on the CPU
cpu_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
print(f"Original Tensor (CPU): {cpu_tensor}")

# 3. Move the tensor to the GPU
gpu_tensor = cpu_tensor.cuda()

# 4. Call your custom CUDA kernel!
print("Sending to custom CUDA kernel...")
result_gpu = dummy_module.multiply_by_two(gpu_tensor)

# 5. Bring the result back to the CPU and print it
result_cpu = result_gpu.cpu()
print(f"Result Tensor (CPU):   {result_cpu}")

# If the result is [2., 4., 6., 8., 10.], YOU DID IT!
if torch.allclose(result_cpu, cpu_tensor * 2):
    print("\nThe Bridge is working perfectly!")
else:
    print("\nSomething went wrong with the math.")