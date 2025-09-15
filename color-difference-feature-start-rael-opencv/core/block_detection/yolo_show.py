# YOLO Colorbar Detection for Gradio Integration
import os

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


def load_yolo_model(model_path: str = None) -> YOLO:
    """
    Load YOLO model for colorbar detection.

    Args:
        model_path: Path to YOLO model weights. If None, uses default path.

    Returns:
        YOLO model instance
    """
    if model_path is None:
        # Try multiple possible paths for the model
        possible_paths = [
            "./core/block_detection/weights/best0710.pt",
            "./runs/train/custom_colorbar2/weights/best.pt",
            "./weights/best.pt",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            raise FileNotFoundError(
                f"YOLO model not found in any of these paths: {possible_paths}"
            )

    print(f"Loading YOLO model from: {model_path}")
    return YOLO(model_path)


def detect_colorbars_yolo(
    image: np.ndarray,
    model: YOLO,
    box_expansion: int = 10,
    confidence_threshold: float = 0.5,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], list[float], list[np.ndarray]]:
    """
    Detect colorbars in an image using YOLO model.

    Args:
        image: Input image as numpy array (BGR format)
        model: YOLO model instance
        box_expansion: Number of pixels to expand detection boxes
        confidence_threshold: Minimum confidence score for detections

    Returns:
        Tuple of (annotated_image, boxes_list, confidences_list, colorbar_segments)
    """
    # Run YOLO inference
    results = model(image)

    # Create annotated image
    img_with_boxes = image.copy()

    boxes_list = []
    confidences_list = []
    colorbar_segments = []

    height, width = image.shape[:2]

    # Process detections
    for result in results:
        boxes = result.boxes.cpu().numpy()

        for _i, box in enumerate(boxes):
            confidence = float(box.conf[0])

            # Filter by confidence threshold
            if confidence < confidence_threshold:
                continue

            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].astype(int)

            # Expand box by specified pixels
            x1_exp = max(0, x1 - box_expansion)
            y1_exp = max(0, y1 - box_expansion)
            x2_exp = min(width, x2 + box_expansion)
            y2_exp = min(height, y2 + box_expansion)

            # Draw rectangle on annotated image
            cv2.rectangle(
                img_with_boxes, (x1_exp, y1_exp), (x2_exp, y2_exp), (0, 255, 0), 2
            )

            # Add confidence label
            label = f"Colorbar: {confidence:.2f}"
            cv2.putText(
                img_with_boxes,
                label,
                (x1_exp, y1_exp - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Extract colorbar segment
            colorbar_segment = image[y1_exp:y2_exp, x1_exp:x2_exp]

            # Store results
            boxes_list.append((x1_exp, y1_exp, x2_exp, y2_exp))
            confidences_list.append(confidence)
            colorbar_segments.append(colorbar_segment)

    return img_with_boxes, boxes_list, confidences_list, colorbar_segments


def detect_colorbars_from_pil(
    pil_image: Image.Image,
    model_path: str = None,
    box_expansion: int = 10,
    confidence_threshold: float = 0.5,
) -> tuple[Image.Image, list[Image.Image], int, list[float]]:
    """
    Detect colorbars from PIL Image and return PIL Images.
    Wrapper function for Gradio compatibility.

    Args:
        pil_image: PIL Image input
        model_path: Path to YOLO model weights
        box_expansion: Pixels to expand detection boxes
        confidence_threshold: Minimum confidence for detections

    Returns:
        Tuple of (annotated_PIL_image, colorbar_PIL_segments, count, confidences)
    """
    if pil_image is None:
        return None, [], 0, []

    # Load YOLO model
    try:
        model = load_yolo_model(model_path)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return pil_image, [], 0, []

    # Convert PIL to OpenCV format
    opencv_image = np.array(pil_image)
    if len(opencv_image.shape) == 3:
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

    # Run YOLO detection
    try:
        annotated_cv, boxes, confidences, segments_cv = detect_colorbars_yolo(
            opencv_image,
            model,
            box_expansion=box_expansion,
            confidence_threshold=confidence_threshold,
        )

        # Convert results back to PIL format
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_cv, cv2.COLOR_BGR2RGB))

        segments_pil = []
        for segment_cv in segments_cv:
            if segment_cv.size > 0:  # Check if segment is not empty
                segment_pil = Image.fromarray(
                    cv2.cvtColor(segment_cv, cv2.COLOR_BGR2RGB)
                )
                segments_pil.append(segment_pil)

        return annotated_pil, segments_pil, len(segments_pil), confidences

    except Exception as e:
        print(f"Error during YOLO detection: {e}")
        return pil_image, [], 0, []


def analyze_colorbar_colors(colorbar_segment: np.ndarray) -> dict:
    """
    Analyze colors in a detected colorbar segment.

    Args:
        colorbar_segment: Colorbar image segment (BGR format)

    Returns:
        Dictionary with color analysis results
    """
    if colorbar_segment.size == 0:
        return {"error": "Empty colorbar segment"}

    # Convert to RGB for analysis
    rgb_segment = cv2.cvtColor(colorbar_segment, cv2.COLOR_BGR2RGB)

    # Get dominant colors (simplified analysis)
    pixels = rgb_segment.reshape(-1, 3)

    # Calculate basic statistics
    mean_color = np.mean(pixels, axis=0)
    std_color = np.std(pixels, axis=0)

    # Get unique colors (simplified)
    unique_colors = np.unique(pixels, axis=0)

    return {
        "mean_rgb": mean_color.tolist(),
        "std_rgb": std_color.tolist(),
        "unique_colors_count": len(unique_colors),
        "segment_shape": colorbar_segment.shape,
        "total_pixels": len(pixels),
    }
