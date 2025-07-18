# Parallel Morphological Image Processing with CUDA and Python

This repository contains a high-performance implementation of fundamental morphological image processing operations, exploring and comparing different parallel programming models. The project includes:

1. A baseline **sequential** implementation in pure Python
2. A **multi-core CPU parallel** version using Python's `multiprocessing` module
3. A **massively parallel GPU** version accelerated with CUDA via the PyCUDA library

The focus of this work is to provide a comprehensive performance analysis, highlighting the architectural trade-offs and scalability characteristics of task-parallel (CPU) versus data-parallel (GPU) approaches for a classic, embarrassingly parallel computer vision problem.

      
<table>
  <tr>
    <td align="center"><strong>Original Image</strong></td>
    <td align="center"><strong>Processed Image (Closing)</strong></td>
  </tr>
  <tr>
    <td><img src="images/lena_gray.png" alt="Lena Gray" width="100%"></td>
    <td><img src="output_gpu_parallel_15x15.png" alt="GPU Erosion Result" width="100%"></td>
  </tr>
</table>

    
## Features

- **Three Distinct Implementations**: Provides a clear comparison between sequential, multi-core CPU, and massively parallel GPU models
- **Core Morphological Operations**: Implements Erosion, Dilation, Opening, and Closing
- **Efficient GPU Kernel**: A generic and reusable CUDA kernel designed for high-throughput image processing
- **Optimized Chained Operations**: The GPU implementation for Opening and Closing uses a "ping-pong" buffer strategy to eliminate intermediate host-device data transfers, significantly reducing overhead
- **Robust Benchmarking Suite**: Includes automated Python scripts to measure and log performance across a wide range of image sizes and structuring element (SE) sizes
- **GPU Tuning Analysis**: A dedicated benchmark to analyze the impact of CUDA block size on kernel performance
- **Automated Plotting**: Generates a full suite of professional, report-ready plots to visualize execution time and speedup

## Getting Started

### Prerequisites

To build and run this project, you will need the following installed on your system:

- **Python 3.x**
- **An NVIDIA GPU** with CUDA support
- **CUDA Toolkit**: Version 11.x or newer is recommended
- **Required Python Packages**: `numpy`, `Pillow`, `matplotlib`, `seaborn`, `pycuda`

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Install Python dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install numpy pillow matplotlib seaborn pycuda
   ```

## How to Run the Project

The project includes several scripts to automate the benchmarking and analysis process.

### 1. Run the Main 3-Way Benchmark

This is the primary benchmark. It compares the `Sequential`, `Multi-Processing`, and `GPU` implementations.

**Warning**: This script can take a very long time to complete due to the slow nature of the sequential version on large images. It is recommended to run it with a smaller set of `IMAGE_SIZES` for initial testing.

```bash
python benchmark.py
```

This will generate a `benchmark_results.csv` file in the `results/` directory.

### 2. Run the GPU Block Size Benchmark (Optional)

This script runs a much faster benchmark that analyzes only the GPU implementation's performance with different CUDA block sizes.

```bash
python benchmark_blocksize.py
```

This will generate a `benchmark_results_blocksize_all_ops.csv` file.

### 3. Generate All Plots

After one or both benchmark scripts have been run, execute the plotting script to generate visualizations from the CSV data.

```bash
python plot_results.py
```

All plots will be saved as `.png` files in the `results/plots/` directory.

## Performance Results

The implementations were tested on an Intel(R) Core(TM) i7- 10700K CPU @
3.80GHz and an NVIDIA GeForce RTX 2060 SUPER with 8 GB VRAM. The results clearly demonstrate the distinct scalability characteristics of each parallel architecture.

The CPU Multi-Processing implementation provides a consistent and valuable speedup, typically plateauing around 7-8x, which is limited by the number of physical CPU cores.

The GPU implementation delivers a massive performance increase, achieving speedups of over 1,000x and scaling dramatically with both image size and structuring element size. For the largest test cases, speedups can exceed 60,000x.

GPU performance tuning shows an optimal CUDA block size of 16x16 (256 threads) for this workload, providing the best balance of parallelism and resource utilization.

This highlights that while multi-core CPUs are effective for task parallelism, the massively parallel architecture of the GPU is fundamentally superior for data-parallel problems like image processing.

## Technical Report

A full technical report is available at the root directory of the project under the name `report.pdf`. Below is the abstract from that report:

> This report presents a comprehensive performance comparison of three implementations for fundamental morphological image processing operations: a sequential Python baseline, a multi-core CPU parallel version using Python's multiprocessing module, and a massively parallel GPU version implemented in CUDA. The objective is to analyze and quantify the performance gains and scalability characteristics of different parallel programming models when applied to a classic, data-parallel computer vision task. Key operations including erosion, dilation, opening, and closing are benchmarked across a wide range of image sizes and structuring element sizes. Results demonstrate that while CPU multiprocessing provides a modest and predictable speedup limited by the number of physical cores, the GPU implementation delivers orders-of-magnitude performance improvement that scales dramatically with problem size. The findings highlight the architectural advantages of GPUs for image processing and explore the trade-offs between different parallelization strategies.
