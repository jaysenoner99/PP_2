import numpy as np
from PIL import Image


def generate_donut_image(height=256, width=256, filename="test_image_donut.png"):
    """
    Generates and saves a binary image of a donut (a ring).

    Args:
        height (int): The height of the image in pixels.
        width (int): The width of the image in pixels.
        filename (str): The name of the file to save the image as.
    """
    print(f"Generating donut image ({width}x{height})...")

    image = np.zeros((height, width), dtype=np.uint8)

    center_x, center_y = width // 2, height // 2
    outer_radius = min(width, height) // 3
    inner_radius = outer_radius // 2

    y, x = np.ogrid[-center_y : height - center_y, -center_x : width - center_x]

    dist_from_center_sq = x**2 + y**2

    donut_mask = (dist_from_center_sq <= outer_radius**2) & (
        dist_from_center_sq >= inner_radius**2
    )

    image[donut_mask] = 255

    pil_img = Image.fromarray(image, "L")  # 'L' mode is for grayscale
    pil_img.save(filename)
    print(f"Successfully saved donut image to '{filename}'")
    return image


def generate_noise_image(height=256, width=256, filename="test_image_noise.png"):
    """
    Generates and saves an image with a white square on a black background,
    corrupted by salt and pepper noise.

    Args:
        height (int): The height of the image in pixels.
        width (int): The width of the image in pixels.
        filename (str): The name of the file to save the image as.
    """
    print(f"\nGenerating salt and pepper noise image ({width}x{height})...")

    image = np.zeros((height, width), dtype=np.uint8)
    square_size = height // 2
    start = (height - square_size) // 2
    end = start + square_size
    image[start:end, start:end] = 255

    num_salt_pixels = int(height * width * 0.05)  # 5% of pixels
    salt_rows = np.random.randint(0, height, num_salt_pixels)
    salt_cols = np.random.randint(0, width, num_salt_pixels)
    image[salt_rows, salt_cols] = 255

    num_pepper_pixels = int(
        square_size * square_size * 0.10
    )  # 10% of the square's pixels
    pepper_rows = np.random.randint(start, end, num_pepper_pixels)
    pepper_cols = np.random.randint(start, end, num_pepper_pixels)
    image[pepper_rows, pepper_cols] = 0

    pil_img = Image.fromarray(image, "L")
    pil_img.save(filename)
    print(f"Successfully saved noise image to '{filename}'")
    return image


if __name__ == "__main__":
    print("--- Test Image Generator for Morphological Operations ---")

    height = 512
    width = 512
    filename_donut = "images/test_image_donut.png"
    filename_noise = "images/test_image_noise.png"
    generate_donut_image(height, width, filename_donut)

    generate_noise_image(height, width, filename_noise)

    print("\nAll images generated successfully.")
