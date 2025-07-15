import numpy as np
import csv
import os
import time
import pycuda.autoinit
import pycuda.driver as cuda

# Import the modified parallel functions that accept block_size
from parallel_morph import parallel_morph_op, parallel_opening, parallel_closing

# We also need the sequential versions to calculate speedup
from sequential_morph import (
    sequential_erosion,
    sequential_dilation,
    sequential_opening,
    sequential_closing,
)

# --- Configuration ---
BLOCK_SIZES_TO_TEST = [(4, 4, 1), (8, 8, 1), (16, 16, 1), (32, 32, 1)]

FIXED_IMAGE_SIZE = 2048
FIXED_SE_SIZE = 9

OUTPUT_CSV_FILE = "benchmark_results_blocksize_all_ops.csv"


# --- Operation Mapping ---
def parallel_erosion_wrapper(img, se, block_size):
    return parallel_morph_op(img, se, "erosion", block_size=block_size)


def parallel_dilation_wrapper(img, se, block_size):
    return parallel_morph_op(img, se, "dilation", block_size=block_size)


def parallel_opening_wrapper(img, se, block_size):
    return parallel_opening(img, se, block_size=block_size)


def parallel_closing_wrapper(img, se, block_size):
    return parallel_closing(img, se, block_size=block_size)


OPERATIONS = {
    "Erosion": (sequential_erosion, parallel_erosion_wrapper),
    "Dilation": (sequential_dilation, parallel_dilation_wrapper),
    "Opening": (sequential_opening, parallel_opening_wrapper),
    "Closing": (sequential_closing, parallel_closing_wrapper),
}


def run_blocksize_benchmark():
    """
    Runs a benchmark to test the performance of different CUDA block sizes
    across all morphological operations.
    """
    print("=" * 60)
    print("Starting Block Size Performance Benchmark (All Operations)")
    print(f"Fixed Image Size: {FIXED_IMAGE_SIZE}x{FIXED_IMAGE_SIZE}")
    print(f"Fixed SE Size:    {FIXED_SE_SIZE}x{FIXED_SE_SIZE}")
    print("=" * 60)

    # 1. Generate test data
    np.random.seed(0)
    input_image = np.random.randint(
        0, 256, (FIXED_IMAGE_SIZE, FIXED_IMAGE_SIZE), dtype=np.uint8
    )
    structuring_element = np.ones((FIXED_SE_SIZE, FIXED_SE_SIZE), dtype=np.uint8)

    all_results = []

    # 2. Pre-calculate sequential times to avoid re-running them in the loop
    print("Calculating baseline sequential times (once)...")
    sequential_times = {}
    for op_name, (cpu_func, _) in OPERATIONS.items():
        start_cpu = time.time()
        _ = cpu_func(input_image, structuring_element)
        cpu_time = time.time() - start_cpu
        sequential_times[op_name] = cpu_time
        print(f"  - {op_name}: {cpu_time:.4f}s")
    print("...done.\n")

    # 3. GPU Warm-up
    print("Performing GPU warm-up...")
    _ = parallel_erosion_wrapper(
        input_image, structuring_element, block_size=(16, 16, 1)
    )
    print("Warm-up complete.\n")

    # 4. Main benchmark loop
    for block_size in BLOCK_SIZES_TO_TEST:
        block_dim_x, block_dim_y, _ = block_size
        threads_per_block = block_dim_x * block_dim_y
        print(
            f"--- Testing Block Size: {block_dim_x}x{block_dim_y} ({threads_per_block} threads) ---"
        )

        for op_name, (_, gpu_func) in OPERATIONS.items():
            # Time GPU Execution
            start_gpu = cuda.Event()
            end_gpu = cuda.Event()

            start_gpu.record()
            gpu_func(input_image, structuring_element, block_size=block_size)
            end_gpu.record()
            end_gpu.synchronize()
            gpu_time = start_gpu.time_till(end_gpu) * 1e-3

            # Get the corresponding CPU time
            cpu_time = sequential_times[op_name]
            speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")

            # Store results
            result_row = {
                "image_size": FIXED_IMAGE_SIZE,
                "se_size": FIXED_SE_SIZE,
                "operation": op_name,
                "block_dim_x": block_dim_x,
                "threads_per_block": threads_per_block,
                "cpu_time_s": f"{cpu_time:.6f}",
                "gpu_time_s": f"{gpu_time:.6f}",
                "speedup": f"{speedup:.2f}",
            }
            all_results.append(result_row)

    # 5. Save results to CSV
    print("\n" + "=" * 60)
    print(f"Benchmark complete. Saving results to '{OUTPUT_CSV_FILE}'...")
    try:
        with open(OUTPUT_CSV_FILE, "w", newline="") as csvfile:
            fieldnames = all_results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print("Results saved successfully.")
    except (IOError, IndexError):
        print(f"Error writing to file '{OUTPUT_CSV_FILE}'.")


if __name__ == "__main__":
    run_blocksize_benchmark()
