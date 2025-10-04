# 新建文件：color-difference-feature-start-rael-opencv/core/block_detection/yolo_block_detect.py
import os
import numpy as np
import cv2
from typing import List, Tuple


try:
    from ultralytics import YOLO
    _USE_ULTRALYTICS = True
except ImportError:
    _USE_ULTRALYTICS = False

_MODEL_CACHE = {}

def load_block_model(model_path: str):
    """加载色块检测YOLO模型"""
    if model_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_path]
    
    if not _USE_ULTRALYTICS:
        raise RuntimeError("需要安装 ultralytics: pip install ultralytics")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO模型文件不存在: {model_path}")
    
    model = YOLO(model_path)
    _MODEL_CACHE[model_path] = model
    return model

def detect_blocks_yolo(
    roi_bgr: np.ndarray,
    model_path: str,  # 改为接受路径字符串，与原始函数一致
    output_dir: str = None,  # 添加兼容参数
    area_threshold: int = 100,
    aspect_ratio_threshold: float = 0.7,
    min_square_size: int = 10,
    return_individual_blocks: bool = True,  # 添加兼容参数
    confidence_threshold: float = 0.25,
    **kwargs
):
    """
    使用YOLO检测色块，返回格式与原始detect_blocks兼容
    返回: (result_image_with_boxes, list_of_block_images, block_count)
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return roi_bgr, [], 0
    
    # 加载模型
    model = load_block_model(model_path)
    h, w = roi_bgr.shape[:2]
    
    # YOLO推理
    results = model.predict(
        source=roi_bgr[:, :, ::-1],  # BGR转RGB
        conf=confidence_threshold,
        verbose=False
    )
    
    result_image = roi_bgr.copy()
    block_images = []  # 存储实际的色块图像
    
    if results and len(results) > 0:
        result = results[0]
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            
            for box in boxes:
                x1, y1, x2, y2 = box
                x = max(0, int(x1))
                y = max(0, int(y1))
                w_box = max(1, int(x2 - x1))
                h_box = max(1, int(y2 - y1))
                
                # 确保不超出图像边界
                w_box = min(w_box, w - x)
                h_box = min(h_box, h - y)
                
                # 应用面积过滤
                if w_box * h_box >= area_threshold:
                    # 在结果图像上画框
                    cv2.rectangle(result_image, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                    
                    # 提取色块图像（与原始函数格式一致）
                    if return_individual_blocks:
                        block_image = roi_bgr[y:y + h_box, x:x + w_box]
                        block_images.append(block_image)
    
    block_count = len(block_images)
    return result_image, block_images, block_count