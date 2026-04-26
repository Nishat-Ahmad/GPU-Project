#include <torch/extension.h>

// 1. THE CUDA KERNEL (Executes on the GPU)
__global__ void multiply_by_two_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= 2.0f;
    }
}

// 2. THE C++ WRAPPER (Executes on the CPU)
torch::Tensor multiply_by_two(torch::Tensor input) {
    auto device_input = input.cuda().contiguous();
    int size = device_input.numel();
    
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    multiply_by_two_kernel<<<blocks, threads>>>(device_input.data_ptr<float>(), size);
    
    // PyTorch streams are implicitly synchronized, but keep this if you want absolute safety
    cudaDeviceSynchronize(); 
    return device_input;
}

// 3. THE PYTHON BRIDGE (PyBind11)
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("multiply_by_two", &multiply_by_two, "A dummy kernel that multiplies by 2");
}
