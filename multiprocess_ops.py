import numpy as np
import multiprocessing
from sequential_morph import sequential_erosion, sequential_dilation

# --- Global variables for worker processes ---
# To avoid passing large arrays as arguments (which can be slow due to pickling),
# we can use global variables. The 'initializer' function in Pool will set these
# up for each child process.
_input_image = None
_se = None


def _init_worker(image_data, se_data):
    """Initializer for each worker process."""
    global _input_image, _se
    _input_image = image_data
    _se = se_data


def _process_chunk_erosion(y_range):
    """
    Worker function that performs erosion on a horizontal slice of the image.
    It accesses the image and SE data from global variables.
    """
    y_start, y_end = y_range
    # We must call the original sequential function here on the sub-problem
    # Note: We pass the full image, but the sequential_erosion logic will be
    # modified slightly to only compute for the given y_range.

    # A small helper function that's basically a copy of sequential_erosion
    # but with a constrained outer loop.
    img_h, img_w = _input_image.shape
    se_h, se_w = _se.shape
    se_h_radius, se_w_radius = se_h // 2, se_w // 2

    # Create an output chunk of the correct size
    output_chunk = np.zeros((y_end - y_start, img_w), dtype=_input_image.dtype)

    for y_local, y_global in enumerate(range(y_start, y_end)):
        for x in range(img_w):
            min_val = 255
            for j in range(se_h):
                for i in range(se_w):
                    if _se[j, i] == 1:
                        ny = y_global + j - se_h_radius
                        nx = x + i - se_w_radius
                        if 0 <= ny < img_h and 0 <= nx < img_w:
                            min_val = min(min_val, _input_image[ny, nx])
            output_chunk[y_local, x] = min_val

    return output_chunk


def _process_chunk_dilation(y_range):
    """Worker function for dilation."""
    y_start, y_end = y_range
    img_h, img_w = _input_image.shape
    se_h, se_w = _se.shape
    se_h_radius, se_w_radius = se_h // 2, se_w // 2

    output_chunk = np.zeros((y_end - y_start, img_w), dtype=_input_image.dtype)

    for y_local, y_global in enumerate(range(y_start, y_end)):
        for x in range(img_w):
            max_val = 0
            for j in range(se_h):
                for i in range(se_w):
                    if _se[j, i] == 1:
                        ny = y_global + j - se_h_radius
                        nx = x + i - se_w_radius
                        if 0 <= ny < img_h and 0 <= nx < img_w:
                            max_val = max(max_val, _input_image[ny, nx])
            output_chunk[y_local, x] = max_val

    return output_chunk


def _run_in_parallel(image, se, worker_func):
    """A generic parallel runner."""
    img_h, _ = image.shape

    # Determine the number of processes to use (usually number of CPU cores)
    num_processes = multiprocessing.cpu_count()

    # Divide the image into chunks (list of y-ranges)
    chunk_size = (img_h + num_processes - 1) // num_processes
    chunks = [(i, min(i + chunk_size, img_h)) for i in range(0, img_h, chunk_size)]

    # Create a pool of worker processes
    with multiprocessing.Pool(
        processes=num_processes, initializer=_init_worker, initargs=(image, se)
    ) as pool:
        # Map the worker function to the chunks
        results = pool.map(worker_func, chunks)

    # Stitch the resulting chunks back together
    return np.vstack(results)


# --- Public API Functions ---
def erosion_mp(image, se):
    return _run_in_parallel(image, se, _process_chunk_erosion)


def dilation_mp(image, se):
    return _run_in_parallel(image, se, _process_chunk_dilation)


def opening_mp(image, se):
    eroded = erosion_mp(image, se)
    return dilation_mp(eroded, se)


def closing_mp(image, se):
    dilated = dilation_mp(image, se)
    return erosion_mp(dilated, se)
