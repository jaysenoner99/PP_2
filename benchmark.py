# benchmark.py (Full version with 3-way comparison)

import numpy as np
import time
import os
import csv
import multiprocessing

import pycuda.autoinit
import pycuda.driver as cuda

from sequential_morph import (
    sequential_erosion,
    sequential_dilation,
    sequential_opening,
    sequential_closing,
)
from multiprocess_ops import erosion_mp, dilation_mp, opening_mp, closing_mp
from parallel_morph import parallel_morph_op, parallel_opening, parallel_closing

# --- Benchmark Configuration ---
IMAGE_SIZES = [256, 512, 1024, 2048, 4096]
SE_SIZES = [3, 7, 9, 15]
OUTPUT_CSV_FILE = "results/benchmark_results_mp.csv"


def parallel_erosion_wrapper(img, se):
    return parallel_morph_op(img, se, "erosion")


def parallel_dilation_wrapper(img, se):
    return parallel_morph_op(img, se, "dilation")


OPERATIONS = {
    "Erosion": (sequential_erosion, erosion_mp, parallel_erosion_wrapper),
    "Dilation": (sequential_dilation, dilation_mp, parallel_dilation_wrapper),
    "Opening": (sequential_opening, opening_mp, parallel_opening),
    "Closing": (sequential_closing, closing_mp, parallel_closing),
}


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def run_full_benchmark():
    """
    Runs a systematic benchmark across all three implementations,
    saving the results to a CSV file.
    """
    ensure_dir("results")
    print("=" * 60)
    print("Starting Comprehensive 3-Way Benchmark")
    print("(Sequential vs. Multi-Processing vs. GPU)")
    print("=" * 60)

    all_results = []

    # --- GPU Warm-up ---
    print("Performing GPU warm-up...")
    warmup_img = np.zeros((128, 128), dtype=np.uint8)
    warmup_se = np.ones((3, 3), dtype=np.uint8)
    _ = parallel_erosion_wrapper(warmup_img, warmup_se)
    print("Warm-up complete.\n")

    total_tests = len(IMAGE_SIZES) * len(SE_SIZES) * len(OPERATIONS)
    test_count = 0

    # --- Main Benchmark Loop ---
    for img_size in IMAGE_SIZES:
        for se_size in SE_SIZES:
            print(
                f"--- Config: Image={img_size}x{img_size}, SE={se_size}x{se_size} ---"
            )

            np.random.seed(0)
            input_image = np.random.randint(
                0, 256, (img_size, img_size), dtype=np.uint8
            )
            structuring_element = np.ones((se_size, se_size), dtype=np.uint8)

            for op_name, (cpu_seq_func, cpu_mp_func, gpu_func) in OPERATIONS.items():
                test_count += 1
                print(f"  ({test_count}/{total_tests}) Running {op_name}...")

                # --- 1. Time CPU-Sequential (The Baseline) ---
                start_cpu_seq = time.time()
                output_cpu_seq = cpu_seq_func(input_image, structuring_element)
                cpu_seq_time = time.time() - start_cpu_seq

                # --- 2. Time CPU-Multi-Processing ---
                start_cpu_mp = time.time()
                output_cpu_mp = cpu_mp_func(input_image, structuring_element)
                cpu_mp_time = time.time() - start_cpu_mp

                # --- 3. Time GPU ---
                start_gpu = cuda.Event()
                end_gpu = cuda.Event()
                start_gpu.record()
                output_gpu = gpu_func(input_image, structuring_element)
                end_gpu.record()
                end_gpu.synchronize()
                gpu_time = start_gpu.time_till(end_gpu) * 1e-3

                # --- Verify Correctness ---
                correct_mp = np.allclose(output_cpu_seq, output_cpu_mp, atol=1)
                correct_gpu = np.allclose(output_cpu_seq, output_gpu, atol=1)
                if not correct_mp or not correct_gpu:
                    print(
                        f"    WARNING: Correctness check failed! MP:{correct_mp}, GPU:{correct_gpu}"
                    )

                # --- Calculate Speedups (relative to sequential) ---
                speedup_mp = (
                    cpu_seq_time / cpu_mp_time if cpu_mp_time > 0 else float("inf")
                )
                speedup_gpu = cpu_seq_time / gpu_time if gpu_time > 0 else float("inf")

                # --- Store Results ---
                result_row = {
                    "image_size": img_size,
                    "se_size": se_size,
                    "operation": op_name,
                    "cpu_seq_time_s": f"{cpu_seq_time:.6f}",
                    "cpu_mp_time_s": f"{cpu_mp_time:.6f}",
                    "gpu_time_s": f"{gpu_time:.6f}",
                    "speedup_mp": f"{speedup_mp:.2f}",
                    "speedup_gpu": f"{speedup_gpu:.2f}",
                }
                all_results.append(result_row)

    # --- Save all results to CSV ---
    print("\n" + "=" * 60)
    print(f"Benchmark complete. Saving results to '{OUTPUT_CSV_FILE}'...")

    # Define the new headers
    fieldnames = [
        "image_size",
        "se_size",
        "operation",
        "cpu_seq_time_s",
        "cpu_mp_time_s",
        "gpu_time_s",
        "speedup_mp",
        "speedup_gpu",
    ]
    try:
        with open(OUTPUT_CSV_FILE, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print("Results saved successfully.")
    except IOError as e:
        print(f"Error: Could not write to file '{OUTPUT_CSV_FILE}'. {e}")

    print("=" * 60)


if __name__ == "__main__":
    # This is important for multiprocessing on Windows/macOS
    multiprocessing.freeze_support()
    run_full_benchmark()
