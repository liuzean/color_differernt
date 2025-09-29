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
    model,  # 注意这里是模型对象，不是路径
    confidence_threshold: float = 0.25,
    min_area: int = 100,
    **kwargs
):
    """
    使用YOLO检测色块
    参数:
        roi_bgr: 输入的BGR图像
        model: 已加载的YOLO模型对象
        confidence_threshold: 置信度阈值
        min_area: 最小区域面积
    返回:
        (segmented_colorbar, color_blocks, block_count)
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return roi_bgr, [], 0
    
    h, w = roi_bgr.shape[:2]
    
    # YOLO推理
    results = model.predict(
        source=roi_bgr[:, :, ::-1],  # BGR转RGB
        conf=confidence_threshold,
        verbose=False
    )
    
    blocks = []
    segmented_colorbar = roi_bgr.copy()
    
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
                
                # 应用最小面积过滤
                if w_box * h_box >= min_area:
                    blocks.append((x, y, w_box, h_box))
                    
                    # 在segmented_colorbar上画出检测框（可选）
                    cv2.rectangle(segmented_colorbar, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
    
    # 按x坐标排序
    blocks.sort(key=lambda b: b[0])
    block_count = len(blocks)
    
    return segmented_colorbar, blocks, block_count