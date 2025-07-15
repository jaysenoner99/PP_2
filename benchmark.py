import numpy as np
import time
import os
import csv

# PyCUDA imports
import pycuda.autoinit
import pycuda.driver as cuda

# Import your custom functions
from sequential_morph import (
    sequential_erosion,
    sequential_dilation,
    sequential_opening,
    sequential_closing,
)
from parallel_morph import parallel_morph_op, parallel_opening, parallel_closing

# --- Benchmark Configuration ---
IMAGE_SIZES = [256, 512, 1024, 2048, 4096]  # e.g., 256x256, 512x512, etc.
SE_SIZES = [3, 7, 9, 15]
OUTPUT_CSV_FILE = "results/benchmark_results.csv"


# --- Operation Mapping ---
# We need a unified way to call parallel functions
def parallel_erosion_wrapper(img, se):
    return parallel_morph_op(img, se, "erosion")


def parallel_dilation_wrapper(img, se):
    return parallel_morph_op(img, se, "dilation")


OPERATIONS = {
    "Erosion": (sequential_erosion, parallel_erosion_wrapper),
    "Dilation": (sequential_dilation, parallel_dilation_wrapper),
    "Opening": (sequential_opening, parallel_opening),
    "Closing": (sequential_closing, parallel_closing),
}


def run_full_benchmark():
    """
    Runs a systematic benchmark across different image sizes and SE sizes,
    saving the results to a CSV file.
    """
    print("=" * 60)
    print("Starting Comprehensive Morphological Operation Benchmark")
    print("=" * 60)

    # List to hold all result dictionaries
    all_results = []

    # --- GPU Warm-up ---
    # This is crucial for getting stable measurements for the first real run.
    print("Performing GPU warm-up...")
    warmup_img = np.zeros((128, 128), dtype=np.uint8)
    warmup_se = np.ones((3, 3), dtype=np.uint8)
    _ = parallel_morph_op(warmup_img, warmup_se, "erosion")
    print("Warm-up complete.\n")

    total_tests = len(IMAGE_SIZES) * len(SE_SIZES) * len(OPERATIONS)
    test_count = 0

    # --- Main Benchmark Loop ---
    for img_size in IMAGE_SIZES:
        for se_size in SE_SIZES:
            # 1. Generate test data on the fly
            print(
                f"--- Configuration: Image={img_size}x{img_size}, SE={se_size}x{se_size} ---"
            )

            # Create a pseudo-random but repeatable image for consistency
            np.random.seed(0)
            input_image = np.random.randint(
                0, 256, (img_size, img_size), dtype=np.uint8
            )
            structuring_element = np.ones((se_size, se_size), dtype=np.uint8)

            for op_name, (cpu_func, gpu_func) in OPERATIONS.items():
                test_count += 1
                print(f"  ({test_count}/{total_tests}) Running {op_name}...")

                # --- Time CPU Execution ---
                start_cpu = time.time()
                output_cpu = cpu_func(input_image, structuring_element)
                cpu_time = time.time() - start_cpu

                # --- Time GPU Execution ---
                start_gpu = cuda.Event()
                end_gpu = cuda.Event()

                start_gpu.record()
                output_gpu = gpu_func(input_image, structuring_element)
                end_gpu.record()
                end_gpu.synchronize()
                gpu_time = start_gpu.time_till(end_gpu) * 1e-3  # ms to s

                # --- Verify Correctness and Calculate Speedup ---
                correctness = "PASSED"
                if not np.allclose(output_cpu, output_gpu, atol=1):
                    correctness = "FAILED"
                    print(f"    WARNING: Correctness check failed for {op_name}!")

                speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")

                # --- Store Results ---
                result_row = {
                    "image_size": img_size,
                    "se_size": se_size,
                    "operation": op_name,
                    "cpu_time_s": f"{cpu_time:.6f}",
                    "gpu_time_s": f"{gpu_time:.6f}",
                    "speedup": f"{speedup:.2f}",
                    "correctness": correctness,
                }
                all_results.append(result_row)

    # --- Save all results to CSV ---
    print("\n" + "=" * 60)
    print(f"Benchmark complete. Saving results to '{OUTPUT_CSV_FILE}'...")

    try:
        with open(OUTPUT_CSV_FILE, "w", newline="") as csvfile:
            # Define the headers based on the keys of our result dictionary
            fieldnames = [
                "image_size",
                "se_size",
                "operation",
                "cpu_time_s",
                "gpu_time_s",
                "speedup",
                "correctness",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(all_results)
        print("Results saved successfully.")
    except IOError:
        print(f"Error: Could not write to file '{OUTPUT_CSV_FILE}'.")

    print("=" * 60)


if __name__ == "__main__":
    run_full_benchmark()
