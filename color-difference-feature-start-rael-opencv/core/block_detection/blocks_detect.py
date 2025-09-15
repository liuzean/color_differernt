import os

import cv2
import numpy as np
from PIL import Image


def detect_blocks(
    image_path: str,
    output_dir: str = "detected_blocks",
    area_threshold: int = 100,
    aspect_ratio_threshold: float = 0.7,
    min_square_size: int = 10,
    return_individual_blocks: bool = True,
) -> tuple[np.ndarray, list[np.ndarray], int]:
    """
    Detect rectangular blocks in an image using simple edge detection.
    Simplified approach that works better for all colors including light ones.

    Args:
        image_path: Path to the input image
        output_dir: Directory to save detected blocks (optional)
        area_threshold: Minimum area for detected blocks
        aspect_ratio_threshold: Minimum aspect ratio for rectangular blocks
        min_square_size: Minimum width and height for detected blocks (pixels)
        return_individual_blocks: Whether to return individual block images

    Returns:
        Tuple of (result_image_with_boxes, list_of_block_images, block_count)
    """
    # Read image
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
    elif isinstance(image_path, np.ndarray):
        image = image_path
    else:
        # Handle PIL Image input
        image = np.array(image_path)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if image is None:
        raise ValueError("Could not load image")

    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Simple and effective processing pipeline
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 3. Edge detection - this works well for detecting boundaries between colors
    edges = cv2.Canny(blurred, 30, 100, apertureSize=3)

    # 4. Dilate edges to close gaps
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # 5. Close remaining gaps
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 6. Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare result image
    result = image.copy()
    block_images = []
    block_count = 0

    # Process each contour
    for contour in contours:
        # Calculate contour area
        area = cv2.contourArea(contour)

        # Skip if too small
        if area < area_threshold:
            continue

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Check size constraints
        if w < min_square_size or h < min_square_size:
            continue

        # Calculate aspect ratio
        aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0

        # Check aspect ratio
        if aspect_ratio < aspect_ratio_threshold:
            continue

        # Draw rectangle on result image
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add block number label
        cv2.putText(
            result,
            f"{block_count}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        # Extract block image if requested
        if return_individual_blocks:
            block_image = image[y : y + h, x : x + w]
            block_images.append(block_image)

            # Save block image if output directory is specified
            if output_dir:
                block_filename = os.path.join(output_dir, f"block_{block_count}.png")
                cv2.imwrite(block_filename, block_image)

        block_count += 1

    return result, block_images, block_count


def detect_blocks_from_pil(
    pil_image: Image.Image,
    area_threshold: int = 100,
    aspect_ratio_threshold: float = 0.7,
    min_square_size: int = 10,
) -> tuple[Image.Image, list[Image.Image], int]:
    """
    Detect blocks from PIL Image and return PIL Images.
    Wrapper function for Gradio compatibility.

    Args:
        pil_image: PIL Image input
        area_threshold: Minimum area for detected blocks
        aspect_ratio_threshold: Minimum aspect ratio for rectangular blocks
        min_square_size: Minimum width and height for detected blocks (pixels)

    Returns:
        Tuple of (result_PIL_image, list_of_block_PIL_images, block_count)
    """
    # Convert PIL to OpenCV format
    opencv_image = np.array(pil_image)
    if len(opencv_image.shape) == 3:
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

    # Run detection
    result_cv, block_images_cv, count = detect_blocks(
        opencv_image,
        output_dir=None,  # Don't save files
        area_threshold=area_threshold,
        aspect_ratio_threshold=aspect_ratio_threshold,
        min_square_size=min_square_size,
        return_individual_blocks=True,
    )

    # Convert results back to PIL format
    result_pil = Image.fromarray(cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB))

    block_images_pil = []
    for block_cv in block_images_cv:
        block_pil = Image.fromarray(cv2.cvtColor(block_cv, cv2.COLOR_BGR2RGB))
        block_images_pil.append(block_pil)

    return result_pil, block_images_pil, count


# Original script functionality for standalone usage
if __name__ == "__main__":
    # Original script behavior
    image_path = "./datasets/7_conf0_84_0.png"

    try:
        result_image, block_images, count = detect_blocks(
            image_path,
            output_dir="detected_blocks",
            area_threshold=100,
            aspect_ratio_threshold=0.7,
        )

        print(f"Total {count} blocks detected and saved to detected_blocks/")

        # Display results
        cv2.imshow("Result", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Save detection result
        cv2.imwrite("detected_squares.png", result_image)

    except Exception as e:
        print(f"Error: {e}")
