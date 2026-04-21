# GPU-Accelerated Sentiment Analysis Engine

**Team Members:** Nishat Ahmad (2023574) | Muhammad Shaheer (2023508)  
**Repository:** [https://github.com/Nishat-Ahmad/GPU-Project](https://github.com/Nishat-Ahmad/GPU-Project)

---

## Scope and Motivation

### Why we chose this project
As Artificial Intelligence students, we wanted to learn more about the backend execution and underlying libraries of how ML models are actually implemented. Standard frameworks like PyTorch handle this well but carry significant overhead. This project bypasses that overhead by processing large batches of tokenized text through a custom-built CUDA pipeline to generate 1-to-5-star sentiment ratings entirely from scratch.

### Use Cases & Benefits
This project is designed for analyzing large datasets. It can be utilized by companies to perform market research and sentiment analysis. The massive parallelism of the GPU enables real-time sentiment tracking across multiple streams of data, making it highly effective for monitoring live social media feeds.

---

## Overall Structure and Working

The system is designed as a streamlined pipeline, distinctly separated into **Host (Python)** and **Device (C++/CUDA)** boundaries.

### The Non-GPU Part (Host / Wrapper / Baseline)
To maximize time spent on GPU optimization rather than boilerplate data parsing, the Host environment is implemented in Python. It acts as our data orchestrator, baseline model, and testing framework:

* **Data Loader & Baseline Model:** We use standard Python libraries (PyTorch, NumPy, Pandas) to handle reading CSV datasets and converting raw text into integer token IDs.
* **CUDA Integration Bridge:** We utilize a binding framework (like PyCUDA or PyTorch Custom C++ Extensions) to interface between the Python host and the GPU.
* **Execution Flow:** The Python host allocates memory, pushes the token batches to the GPU, and dispatches the grid/block configurations. Instead of calling native PyTorch functions, the host sequentially calls our custom-written `.cu` kernels.

### The GPU Part (Device / Accelerators)
This is the core of the project. We systematically swap out standard Python functions with our specialized CUDA kernels. 

#### The 3-Phase Kernel Pipeline:
1.  **Data Preparation:** Kernels for padding sentences, performing 2D Embedding Lookups, and injecting Positional Encoding.
2.  **Feature Extraction (Neural Layers):** Kernels for Weighted Mean Pooling (compressing sequences into vectors), Matrix Multiplication (GEMM) for the hidden layers, Non-linear Activation (Leaky ReLU), and Batch Normalization (Mean, Variance, and Apply kernels).
3.  **Classification:** A final Matrix-Vector (GEMV) projection, followed by a 3-step Softmax pipeline (Row Max, Row Sum, Division) to calculate probabilities, and a parallel Argmax kernel to determine the final 1–5-star prediction.

**Pipeline Interaction:** The Python script loads a batch of sentences and pushes the tensors to the GPU. The host sequentially fires the custom CUDA kernels, processing the data entirely in VRAM. Once the final kernel finishes, the host retrieves a single array of integers representing the final star ratings.

---

## Implementation Strategy

We are executing this project in three distinct phases:

1.  **Host Infrastructure and Baseline Operations:** Develop the Python host environment (data loading/tokenization) and establish the integration bridge. Implement and test the initial element-wise CUDA kernels (Bias Add, Leaky ReLU, and Positional Encoding).
2.  **Core GPU Acceleration:** Focus on heavy mathematical workloads and complex parallel patterns. Replace Python layers with custom CUDA implementations for Tiled Matrix Multiplication (GEMM) and tree-based Parallel Reductions for Batch Normalization and Softmax.
3.  **Integration, Verification, and Optimization:** Link the end-to-end pipeline. Verify GPU outputs against the Python "Golden Reference" for mathematical correctness. Measure throughput speedups and apply advanced memory optimizations (CUDA Streams, Kernel Fusion).

---

## CUDA Techniques Utilized

* **Memory Coalescing (Chapter 6):** Structuring thread accesses during the Embedding Lookup to ensure adjacent threads read adjacent memory addresses, utilizing full warp bandwidth.
* **Tiled Shared Memory (Chapters 4 & 5):** Utilizing `__shared__` memory for the Multi-Layer Perceptron (GEMM) kernels to minimize slow global memory fetches.
* **Parallel Reduction (Chapter 10):** Implementing tree-based reductions with warp-level synchronization (`__shfl_down_sync`) for the Mean Pooling, Batch Normalization, and Softmax layers.

---

## Verification and Benchmarking

### Verification of Correctness
We utilize a "Golden Reference" script in Python using PyTorch. By passing a specific batch of text through both the PyTorch baseline model and our custom CUDA engine, we use standard C++ assertions to compare the output tensors of every single kernel. If the GPU floating-point outputs match the PyTorch outputs within a small tolerance, the implementation is considered mathematically verified.

### Measuring Speed Improvements
Performance is measured by calculating the total **"Sentences Processed Per Second"** using `cudaEvent_t` timers, comparing our custom GPU pipeline against a sequential CPU implementation. Additionally, we use **Nsight Compute** to profile the GPU, measure Arithmetic Intensity, and track Global Memory hit rates to confirm optimization success.
