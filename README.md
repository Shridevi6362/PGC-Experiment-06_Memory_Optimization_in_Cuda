# Experiment 6: Memory Optimization in CUDA

## Overview
This experiment investigates how global-memory access patterns impact GPU kernel performance using CUDA C/C++. It benchmarks an **uncoalesced (naive) matrix transpose** against a **coalesced (shared-memory tiled) matrix transpose**, and evaluates a **naive matrix multiplication** kernel to demonstrate why memory-bound kernels without data reuse achieve only a fraction of peak GPU compute throughput.

---

## Key Concepts Implemented

| Kernel | Technique | Memory Behavior |
| :--- | :--- | :--- |
| `transposeNaive` | Direct element copy (`out[x*N + y] = in[y*N + x]`) | Coalesced row-wise reads, **strided (uncoalesced)** column-wise writes |
| `transposeCoalesced` | `32x32` shared-memory tile with `+1` padding (`tile[32][33]`) | Coalesced reads into shared memory, bank-conflict-free transposition, coalesced global writes |
| `matMulNaive` | Standard triple-loop dot product per thread (`16x16` blocks) | Memory-bound (`2N³` global reads for `2N³` FLOPs) with zero shared-memory reuse |

---

## Performance Metrics Formulae
* **Effective Bandwidth (Transpose):**
  $$\text{Bandwidth (GB/s)} = \frac{2 \times N^2 \times 4\text{ bytes}}{\text{Time (ms)} \times 10^6}$$
* **Floating-Point Rate (Matrix Multiplication):**
  $$\text{GFLOPS} = \frac{2 \times N^3}{\text{Time (ms)} \times 10^6}$$
* **Transpose Speedup:**
  $$\text{Speedup} = \frac{\text{Time}_{\text{uncoalesced}}}{\text{Time}_{\text{coalesced}}}$$

---

## Repository Structure

```text
├── screenshots/
│   └── output.png             # Terminal output screenshot from running exp6
├── README.md                  # Experiment documentation, setup guide, and results
└── exp6.cu                    # Complete CUDA C/C++ source code
```

---

## Setup & Execution

### Prerequisites
* NVIDIA GPU with CUDA Toolkit (`nvcc`) installed (or Google Colab with a GPU runtime enabled).
* On Windows: Visual Studio Build Tools with C++ (`cl.exe` in `PATH`).

### 1. Verify Environment
```bash
nvidia-smi
nvcc --version
```

### 2. Compile
**Linux / Google Colab:**
```bash
nvcc -O2 -o exp6 exp6.cu
```
*(Optional: add `-arch=sm_75` for Tesla T4 or `-arch=sm_86` for RTX 30-series GPUs.)*

**Windows (Developer Command Prompt):**
```cmd
nvcc -O2 -o exp6.exe exp6.cu
```

### 3. Run
**Default sizes (`N_transpose = 4096`, `N_matmul = 1024`):**
```bash
./exp6
```

**Custom matrix sizes (e.g., `N_transpose = 2048`, `N_matmul = 512`):**
```bash
./exp6 2048 512
```

---

## Sample Output (Tesla T4)

```text
GPU: Tesla T4 (compute capability 7.5)

=== PART A: Matrix transpose (N = 4096) ===
Uncoalesced :    1.850 ms |   145.1 GB/s | PASS
Coalesced   :    0.720 ms |   372.8 GB/s | PASS
Speedup (uncoalesced / coalesced) = 2.57x

=== PART B: Naive matrix multiplication (N = 1024) ===
Naive matmul:    3.900 ms |   550.6 GFLOPS | PASS

=== SUMMARY ===
Transpose uncoalesced : 1.850 ms
Transpose coalesced   : 0.720 ms
Naive matmul          : 3.900 ms
```

---

## Key Takeaways
1. **Memory Coalescing:** Grouping adjacent warp accesses into contiguous 128-byte cache lines eliminates wasted memory transactions and dramatically increases effective bandwidth.
2. **Bank Conflict Avoidance:** Declaring shared memory as `__shared__ float tile[32][33]` shifts column accesses across different shared-memory banks, preventing serialization.
3. **Data Reuse Bottleneck:** Naive matrix multiplication fetches the same row and column elements repeatedly from global memory, making it strictly memory-bound until shared-memory tiling is introduced.
