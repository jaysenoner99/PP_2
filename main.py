# main.py (Simplified Version)

import numpy as np
from PIL import Image
import time
import os

# PyCUDA imports
import pycuda.autoinit
import pycuda.driver as cuda

# --- Import your custom functions ---
# We only need sequential_erosion and the "naive" parallel_morph_op for this version.
try:
    from sequential_morph import sequential_erosion
    from parallel_morph import parallel_morph_op
except ImportError as e:
    print(f"Error: Could not import necessary functions. {e}")
    print(
        "Please ensure 'sequential_morph.py' and 'parallel_morph.py' are in the same directory."
    )
    exit()


def run_benchmark():
    """
    Main function to run the benchmark for morphological operations,
    comparing a sequential CPU version against a parallel GPU version.
    """
    # --- 1. Configuration ---
    # Change these parameters to test different scenarios
    IMAGE_PATH = "images/lena_gray_512.tif"
    SE_SIZE = 15  # Use an odd number, e.g., 3, 5, 9, 15
    OPERATION = "erosion"  # or 'dilation'

    # --- 2. Setup ---
    print("=" * 50)
    print("Morphological Image Processing Benchmark")
    print("=" * 50)

    try:
        input_pil = Image.open(IMAGE_PATH).convert("L")
        input_np = np.array(input_pil, dtype=np.uint8)
    except FileNotFoundError:
        print(f"Error: Test image not found at '{IMAGE_PATH}'")
        print("Please create an 'images' folder and place a test image inside.")
        return

    # Create a square structuring element
    structuring_element = np.ones((SE_SIZE, SE_SIZE), dtype=np.uint8)

    img_h, img_w = input_np.shape
    print("Configuration:")
    print(f"  - Image: '{os.path.basename(IMAGE_PATH)}' ({img_w}x{img_h})")
    print(f"  - Structuring Element: {SE_SIZE}x{SE_SIZE} square")
    print(f"  - Operation: {OPERATION.capitalize()}")
    print("-" * 50)

    # Dictionary to hold results
    results = {}

    # --- 3. Sequential CPU Execution ---
    print("1. Running Sequential (CPU) version...")
    start_time_cpu = time.time()
    output_seq = sequential_erosion(input_np, structuring_element)
    end_time_cpu = time.time()
    cpu_time = end_time_cpu - start_time_cpu
    results["cpu_time"] = cpu_time
    results["output_seq"] = output_seq
    print(f"   Done. CPU Time: {cpu_time:.6f} seconds")
    Image.fromarray(output_seq).save(f"output_sequential_{SE_SIZE}x{SE_SIZE}.png")

    # --- 4. Parallel GPU Execution ---
    print("\n2. Running Parallel (GPU) version...")
    start_event = cuda.Event()
    end_event = cuda.Event()
    start_event.record()
    output_par = parallel_morph_op(input_np, structuring_element, OPERATION)
    end_event.record()
    end_event.synchronize()
    gpu_time = start_event.time_till(end_event) * 1e-3  # Convert ms to s
    results["gpu_time"] = gpu_time
    results["output_par"] = output_par
    print(f"   Done. GPU Time: {gpu_time:.6f} seconds")
    Image.fromarray(output_par).save(f"output_gpu_parallel_{SE_SIZE}x{SE_SIZE}.png")

    # --- 5. Correctness Check ---
    print("\n" + "-" * 50)
    print("3. Verifying Correctness...")
    try:
        assert np.array_equal(results["output_seq"], results["output_par"])
        print("   - Correctness Check PASSED: CPU and GPU outputs are identical.")
    except AssertionError:
        print("   - CORRECTNESS CHECK FAILED! The outputs do not match.")

    # --- 6. Performance Summary & Speedup ---
    print("\n" + "=" * 50)
    print("4. Performance Summary")
    print("=" * 50)
    print(f"  - Sequential CPU Time:  {results['cpu_time']:.6f} s")
    print(f"  - Parallel GPU Time:    {results['gpu_time']:.6f} s")
    print("-" * 50)

    try:
        speedup = results["cpu_time"] / results["gpu_time"]
        print(f"GPU vs. CPU Speedup: {speedup:.2f}x faster")
    except (ZeroDivisionError, KeyError):
        print("Could not calculate speedup (division by zero or missing result).")

    print("=" * 50)
    print("Benchmark complete.")


if __name__ == "__main__":
    run_benchmark()
