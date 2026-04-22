#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void multiply_by_two_kernel(float* data, int size){
    // Calculate which thread this is
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Make sure we don't go out of bounds
    if (idx < size) {
        data[idx] = data[idx] * 2.0f;
    }
}

// 2. THE C++ WRAPPER (Executes on the CPU)
torch::Tensor multiply_by_two(torch::Tensor input) {
    // Force the input to be contiguous in memory and on the GPU
    auto device_input = input.cuda().contiguous();
    
    // Get the size of the array
    int size = device_input.numel();
    
    // Define Grid and Block dimensions
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Launch the CUDA Kernel
    multiply_by_two_kernel<<<blocks, threads>>>(
        device_input.data_ptr<float>(), // Get the raw memory pointer
        size
    );

    // Wait for the GPU to finish before returning to Python
    cudaDeviceSynchronize();

    return device_input;
}

// 3. THE PYTHON BRIDGE (PyBind11)
// This maps the C++ function so Python can import it
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("multiply_by_two", &multiply_by_two, "A dummy kernel that multiplies by 2");
}