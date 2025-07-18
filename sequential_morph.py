import numpy as np
from PIL import Image
import time


def sequential_erosion(image: np.ndarray, se: np.ndarray) -> np.ndarray:
    """
    Performs sequential grayscale erosion on an image.

    Args:
        image: The input grayscale image as a NumPy array.
        se: The structuring element as a NumPy array.

    Returns:
        The eroded image as a NumPy array.
    """
    img_h, img_w = image.shape
    se_h, se_w = se.shape
    se_h_radius = se_h // 2
    se_w_radius = se_w // 2

    output_image = np.zeros_like(image)

    for y in range(img_h):
        for x in range(img_w):
            min_val = 255

            for j in range(se_h):
                for i in range(se_w):
                    if se[j, i] == 1:
                        ny = y + j - se_h_radius
                        nx = x + i - se_w_radius

                        if 0 <= ny < img_h and 0 <= nx < img_w:
                            min_val = min(min_val, image[ny, nx])

            output_image[y, x] = min_val

    return output_image


def sequential_dilation(image: np.ndarray, se: np.ndarray) -> np.ndarray:
    """
    Performs sequential grayscale dilation on an image.

    Args:
        image: The input grayscale image as a NumPy array.
        se: The structuring element as a NumPy array.

    Returns:
        The dilated image as a NumPy array.
    """
    img_h, img_w = image.shape
    se_h, se_w = se.shape
    se_h_radius = se_h // 2
    se_w_radius = se_w // 2

    output_image = np.zeros_like(image)

    for y in range(img_h):
        for x in range(img_w):
            max_val = 0

            for j in range(se_h):
                for i in range(se_w):
                    if se[j, i] == 1:
                        ny = y + j - se_h_radius
                        nx = x + i - se_w_radius

                        # Boundary checking
                        if 0 <= ny < img_h and 0 <= nx < img_w:
                            max_val = max(max_val, image[ny, nx])

            output_image[y, x] = max_val

    return output_image


def sequential_opening(image: np.ndarray, se: np.ndarray) -> np.ndarray:
    """
    Perform sequential opening: Sequential Erosion followed by sequential dilation

    """
    return sequential_dilation(sequential_erosion(image, se), se)


def sequential_closing(image: np.ndarray, se: np.ndarray) -> np.ndarray:
    """
    Perform sequential closing: Sequential Dilation followed by sequential erosion

    """
    return sequential_erosion(sequential_dilation(image, se), se)
