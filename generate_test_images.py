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

    # Create a black canvas
    image = np.zeros((height, width), dtype=np.uint8)

    # Define the center and radii of the donut
    center_x, center_y = width // 2, height // 2
    outer_radius = min(width, height) // 3
    inner_radius = outer_radius // 2

    # Use numpy's meshgrid to create coordinate arrays
    # This is much faster than looping through each pixel
    y, x = np.ogrid[-center_y : height - center_y, -center_x : width - center_x]

    # Calculate the squared distance of each pixel from the center
    dist_from_center_sq = x**2 + y**2

    # Create a boolean mask for pixels within the donut shape
    donut_mask = (dist_from_center_sq <= outer_radius**2) & (
        dist_from_center_sq >= inner_radius**2
    )

    # Apply the mask to the image, setting the donut pixels to white (255)
    image[donut_mask] = 255

    # Save the image
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

    # --- 1. Create the base image (white square on black background) ---
    image = np.zeros((height, width), dtype=np.uint8)
    square_size = height // 2
    start = (height - square_size) // 2
    end = start + square_size
    image[start:end, start:end] = 255

    # --- 2. Add Salt noise (random white pixels on the black background) ---
    # We will just add random white pixels everywhere; those that land
    # on the white square won't change anything.
    num_salt_pixels = int(height * width * 0.05)  # 5% of pixels
    salt_rows = np.random.randint(0, height, num_salt_pixels)
    salt_cols = np.random.randint(0, width, num_salt_pixels)
    image[salt_rows, salt_cols] = 255

    # --- 3. Add Pepper noise (random black pixels on the white square) ---
    num_pepper_pixels = int(
        square_size * square_size * 0.10
    )  # 10% of the square's pixels
    pepper_rows = np.random.randint(start, end, num_pepper_pixels)
    pepper_cols = np.random.randint(start, end, num_pepper_pixels)
    image[pepper_rows, pepper_cols] = 0

    # Save the image
    pil_img = Image.fromarray(image, "L")
    pil_img.save(filename)
    print(f"Successfully saved noise image to '{filename}'")
    return image


if __name__ == "__main__":
    # This block will only run when the script is executed directly
    print("--- Test Image Generator for Morphological Operations ---")

    height = 512
    width = 512
    filename_donut = "images/test_image_donut"
    filename_noise = "images/test_image_noise"
    # Generate and save the donut image
    generate_donut_image(height, width, filename_donut)

    # Generate and save the salt & pepper noise image
    generate_noise_image(height, width, filename_noise)

    print("\nAll images generated successfully.")
