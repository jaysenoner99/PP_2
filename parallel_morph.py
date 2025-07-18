import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

CUDA_KERNEL = """
__global__ void morphological_operation(
    const unsigned char* input_image, 
    unsigned char* output_image,
    const int* se,
    const int img_w, const int img_h,
    const int se_w, const int se_h,
    const int is_erosion)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= img_w || y >= img_h) { return; }
    int se_w_radius = se_w / 2;
    int se_h_radius = se_h / 2;
    unsigned char value = is_erosion ? 255 : 0;
    for (int j = 0; j < se_h; ++j) {
        for (int i = 0; i < se_w; ++i) {
            if (se[j * se_w + i] == 1) {
                int nx = x + i - se_w_radius;
                int ny = y + j - se_h_radius;
                if (nx >= 0 && nx < img_w && ny >= 0 && ny < img_h) {
                    unsigned char neighbor_val = input_image[ny * img_w + nx];
                    if (is_erosion) {
                        value = min(value, neighbor_val);
                    } else {
                        value = max(value, neighbor_val);
                    }
                }
            }
        }
    }
    output_image[y * img_w + x] = value;
}
"""

# Compile the kernel once at the module level
mod = SourceModule(CUDA_KERNEL)
_morph_kernel_func = mod.get_function("morphological_operation")


def _execute_gpu_kernel(
    d_input, d_output, d_se, img_w, img_h, se_w, se_h, is_erosion, BLOCK_SIZE
):
    """
    Private helper to execute the kernel on pre-allocated GPU buffers.
    This function does NOT manage memory.
    Accepts a custom block_size
    """
    grid_x = (img_w + BLOCK_SIZE[0] - 1) // BLOCK_SIZE[0]
    grid_y = (img_h + BLOCK_SIZE[1] - 1) // BLOCK_SIZE[1]
    GRID_SIZE = (grid_x, grid_y)

    _morph_kernel_func(
        d_input,
        d_output,
        d_se,
        np.int32(img_w),
        np.int32(img_h),
        np.int32(se_w),
        np.int32(se_h),
        np.int32(is_erosion),
        block=BLOCK_SIZE,
        grid=GRID_SIZE,
    )


def parallel_morph_op(
    image: np.ndarray, se: np.ndarray, operation: str, block_size=(16, 16, 1)
) -> np.ndarray:
    """
    Performs a single GPU-accelerated morphological operation (e.g., erosion).
    This function handles all memory allocation and transfers for one operation.
    """
    if operation.lower() not in ["erosion", "dilation"]:
        raise ValueError("Operation must be 'erosion' or 'dilation'")

    is_erosion = 1 if operation.lower() == "erosion" else 0
    img_h, img_w = image.shape
    se_h, se_w = se.shape

    # Ensure types are correct
    image = image.astype(np.uint8)
    se = se.astype(np.int32)

    # Allocate memory
    d_input = cuda.mem_alloc(image.nbytes)
    d_output = cuda.mem_alloc(image.nbytes)
    d_se = cuda.mem_alloc(se.nbytes)

    # Transfer data to GPU
    cuda.memcpy_htod(d_input, image)
    cuda.memcpy_htod(d_se, se)

    _execute_gpu_kernel(
        d_input, d_output, d_se, img_w, img_h, se_w, se_h, is_erosion, block_size
    )

    # Copy result back and free memory
    output_image = np.empty_like(image)
    cuda.memcpy_dtoh(output_image, d_output)

    d_input.free()
    d_output.free()
    d_se.free()

    return output_image


def _chained_op_template(
    image: np.ndarray, se: np.ndarray, is_erosion_first: bool, block_size=(16, 16, 1)
) -> np.ndarray:
    """
    Template for two-stage operations like opening and closing.
    It uses two GPU buffers to avoid intermediate transfers to the CPU.
    """
    img_h, img_w = image.shape
    se_h, se_w = se.shape

    image = image.astype(np.uint8)
    se = se.astype(np.int32)

    # We need one buffer for the SE and two "ping-pong" buffers for the image
    d_se = cuda.mem_alloc(se.nbytes)
    d_buffer_A = cuda.mem_alloc(image.nbytes)
    d_buffer_B = cuda.mem_alloc(image.nbytes)

    # Copy initial data to GPU
    cuda.memcpy_htod(d_se, se)
    cuda.memcpy_htod(d_buffer_A, image)

    # --- Stage 1 ---
    # Input: d_buffer_A, Output: d_buffer_B
    _execute_gpu_kernel(
        d_buffer_A,
        d_buffer_B,
        d_se,
        img_w,
        img_h,
        se_w,
        se_h,
        int(is_erosion_first),
        block_size,
    )

    # --- Stage 2 ---
    # The output of stage 1 (d_buffer_B) is now the input for stage 2.
    # We can write the final result back into d_buffer_A.
    # Input: d_buffer_B, Output: d_buffer_A
    _execute_gpu_kernel(
        d_buffer_B,
        d_buffer_A,
        d_se,
        img_w,
        img_h,
        se_w,
        se_h,
        int(not is_erosion_first),
        block_size,
    )

    # The final result is now in d_buffer_A. Copy it back to the host.
    output_image = np.empty_like(image)
    cuda.memcpy_dtoh(output_image, d_buffer_A)

    # Free all GPU memory
    d_se.free()
    d_buffer_A.free()
    d_buffer_B.free()

    return output_image


def parallel_erosion(img, se, block_size=(16, 16, 1)):
    return parallel_morph_op(img, se, "erosion", block_size)


def parallel_dilation(img, se, block_size=(16, 16, 1)):
    return parallel_morph_op(img, se, "dilation", block_size)


def parallel_opening(
    image: np.ndarray, se: np.ndarray, block_size=(16, 16, 1)
) -> np.ndarray:
    """
    Performs GPU-accelerated opening (erosion then dilation) by keeping
    the intermediate result on the GPU.
    """
    # Opening: Erosion is the first operation
    return _chained_op_template(image, se, is_erosion_first=True, block_size=block_size)


def parallel_closing(
    image: np.ndarray, se: np.ndarray, block_size=(16, 16, 1)
) -> np.ndarray:
    """
    Performs GPU-accelerated closing (dilation then erosion) by keeping
    the intermediate result on the GPU.
    """
    # Closing: Dilation is the first operation (is_erosion_first = False)
    return _chained_op_template(
        image, se, is_erosion_first=False, block_size=block_size
    )
