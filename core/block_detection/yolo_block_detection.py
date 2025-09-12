# core/block_detection/yolo_block_detection.py

"""
YOLOv8 Block Detection Module

This module provides functions to detect color blocks within a given image
segment (presumably a colorbar) using a trained YOLOv8 model.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Global variable to hold the loaded model, preventing redundant loads
_yolo_block_model = None

def load_yolo_block_model(model_path: str = None) -> YOLO:
    """
    Load the YOLOv8 model for block detection.

    Args:
        model_path: Optional path to the model weights file.

    Returns:
        The loaded YOLO model.
    """
    global _yolo_block_model
    if _yolo_block_model is not None:
        return _yolo_block_model

    if model_path is None:
        # Determine the path to the weights file relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "weights", "best.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO block model not found at: {model_path}")

    try:
        print(f"Loading YOLO block detection model from: {model_path}")
        _yolo_block_model = YOLO(model_path)
        print("YOLO block model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO block model: {e}")

    return _yolo_block_model


def detect_blocks_with_yolo(
    segment: np.ndarray,
    model: YOLO,
    confidence_threshold: float = 0.5,
    min_area: int = 50,
) -> tuple[np.ndarray, list[np.ndarray], int]:
    """
    Detect color blocks in a given image segment using YOLOv8, sort them,
    and return the annotated segment and individual block images.

    Args:
        segment: The input image segment (a cropped colorbar).
        model: The loaded YOLOv8 model.
        confidence_threshold: The confidence threshold for detection.
        min_area: The minimum area for a detected block to be considered valid.

    Returns:
        A tuple containing:
        - The annotated segment with bounding boxes.
        - A sorted list of cropped color block images.
        - The count of valid blocks.
    """
    if segment is None or segment.size == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8), [], 0

    # Perform prediction
    results = model.predict(segment, conf=confidence_threshold, verbose=False)
    
    # [修正] 获取所有有效的检测框，并增加长宽比过滤
    boxes = []
    aspect_ratio_threshold = 1.5 # 定义长宽比阈值

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            
            # 过滤掉面积过小的框
            if w * h < min_area:
                continue
            
            # [新逻辑] 过滤掉长宽比异常的框
            if w == 0 or h == 0: # 避免除以零
                continue
            
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > aspect_ratio_threshold:
                # 如果长宽比过大，则认为不是有效的色块
                continue

            boxes.append((x1, y1, x2, y2))

    # 判断方向并排序
    h_seg, w_seg = segment.shape[:2]
    is_vertical = h_seg > w_seg

    if is_vertical:
        # 如果是纵向, 按 Y 坐标排序 (从上到下)
        boxes.sort(key=lambda b: b[1])
    else:
        # 如果是横向, 按 X 坐标排序 (从左到右)
        boxes.sort(key=lambda b: b[0])

    annotated_segment = segment.copy()
    color_blocks = []
    block_count = 0

    # 遍历排序后的检测框
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        # Draw bounding box and label on the annotated segment
        cv2.rectangle(annotated_segment, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{i + 1}"
        cv2.putText(
            annotated_segment,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        
        # Crop the block from the original segment
        block_image = segment[y1:y2, x1:x2]
        if block_image.size > 0:
            color_blocks.append(block_image)
            block_count += 1
            
    return annotated_segment, color_blocks, block_count