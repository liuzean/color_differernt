"""
YOLO色块检测模块
使用训练好的YOLO模型直接检测色块
重构为类结构，提供更好的模型管理和检测功能
"""

import os
import cv2
import numpy as np
import warnings
from typing import List, Dict, Tuple, Optional
from PIL import Image
from ultralytics import YOLO


class YOLOBlockDetector:
    """YOLO色块检测器类"""
    
    def __init__(self, model_path: str = None):
        """
        初始化YOLO色块检测器
        
        Args:
            model_path: YOLO模型文件路径，如果为None则使用默认路径
        """
        self.model = None
        self.model_path = model_path or "core/block_detection/weights/best.pt"
        self.is_loaded = False
        
        # 尝试加载模型
        self.load_model()
    
    def load_model(self) -> bool:
        """
        加载YOLO模型
        
        Returns:
            bool: 模型是否加载成功
        """
        try:
            if not os.path.exists(self.model_path):
                print(f"警告: YOLO模型文件不存在: {self.model_path}")
                print("请确保已训练好的YOLO模型文件存在于指定路径")
                self.is_loaded = False
                return False
            
            self.model = YOLO(self.model_path)
            self.is_loaded = True
            print(f"YOLO模型加载成功: {self.model_path}")
            return True
            
        except Exception as e:
            print(f"YOLO模型加载失败: {e}")
            self.model = None
            self.is_loaded = False
            return False
    
    def detect_blocks(
        self, 
        image: np.ndarray,
        confidence_threshold: float = 0.5,
        min_area: int = 50,
        max_detections: int = 50
    ) -> List[Dict]:
        """
        检测图像中的色块
        
        Args:
            image: 输入图像 (BGR格式)
            confidence_threshold: 置信度阈值
            min_area: 最小色块面积
            max_detections: 最大检测数量
            
        Returns:
            List[Dict]: 检测结果列表，每个元素包含:
                - bbox: [x1, y1, x2, y2] 边界框坐标
                - confidence: 置信度
                - class_id: 类别ID
                - class_name: 类别名称
                - area: 区域面积
                - center: (x, y) 中心点坐标
        """
        if not self.is_loaded:
            print("错误: YOLO模型未加载")
            return []
        
        if image is None or image.size == 0:
            print("错误: 输入图像为空")
            return []
        
        try:
            # 进行检测
            results = self.model(
                image,
                conf=confidence_threshold,
                max_det=max_detections,
                verbose=False
            )
            
            detections = []
            
            # 解析检测结果
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy() if result.boxes.cls is not None else np.zeros(len(boxes))
                    
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.astype(int)
                        confidence = float(confidences[i])
                        class_id = int(classes[i])
                        
                        # 计算面积
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        
                        # 检查最小面积阈值
                        if area < min_area:
                            continue
                        
                        center_x = x1 + width // 2
                        center_y = y1 + height // 2
                        
                        # 获取类别名称
                        class_name = self._get_class_name(class_id)
                        
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name,
                            'area': area,
                            'center': (center_x, center_y),
                            'width': width,
                            'height': height
                        }
                        
                        detections.append(detection)
            
            # 按置信度排序
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            return detections
            
        except Exception as e:
            print(f"YOLO检测失败: {e}")
            return []
    
    def _get_class_name(self, class_id: int) -> str:
        """
        根据类别ID获取类别名称
        
        Args:
            class_id: 类别ID
            
        Returns:
            str: 类别名称
        """
        # 根据实际的YOLO模型类别设置
        class_names = {
            0: 'color_block',
            1: 'color_patch',
            2: 'color_strip'
        }
        
        return class_names.get(class_id, f'class_{class_id}')
    
    def visualize_detections(
        self, 
        image: np.ndarray, 
        detections: List[Dict],
        show_confidence: bool = True,
        show_class: bool = True,
        line_thickness: int = 2
    ) -> np.ndarray:
        """
        在图像上可视化检测结果
        
        Args:
            image: 原始图像
            detections: 检测结果列表
            show_confidence: 是否显示置信度
            show_class: 是否显示类别名称
            line_thickness: 边界框线条粗细
            
        Returns:
            np.ndarray: 标注后的图像
        """
        if image is None or len(detections) == 0:
            return image.copy() if image is not None else np.zeros((100, 100, 3), dtype=np.uint8)
        
        annotated_image = image.copy()
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # 绘制边界框
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), line_thickness)
            
            # 准备标签文本
            label_parts = []
            if show_class:
                label_parts.append(class_name)
            if show_confidence:
                label_parts.append(f"{confidence:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                
                # 计算文本大小
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )
                
                # 绘制文本背景
                cv2.rectangle(
                    annotated_image,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width + 5, y1),
                    (0, 255, 0),
                    -1
                )
                
                # 绘制文本
                cv2.putText(
                    annotated_image,
                    label,
                    (x1 + 2, y1 - baseline - 2),
                    font,
                    font_scale,
                    (255, 255, 255),  # 白色文本
                    thickness
                )
        
        return annotated_image
    
    def extract_color_blocks(
        self, 
        image: np.ndarray, 
        detections: List[Dict],
        padding: int = 5
    ) -> List[np.ndarray]:
        """
        从检测结果中提取色块图像
        
        Args:
            image: 原始图像
            detections: 检测结果列表
            padding: 边界框扩展像素数
            
        Returns:
            List[np.ndarray]: 提取的色块图像列表
        """
        if image is None or len(detections) == 0:
            return []
        
        color_blocks = []
        h, w = image.shape[:2]
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # 添加padding并确保不超出图像边界
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # 提取色块
            color_block = image[y1:y2, x1:x2]
            
            if color_block.size > 0:
                color_blocks.append(color_block)
        
        return color_blocks
    
    def detect_and_extract(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.5,
        min_area: int = 50,
        max_detections: int = 50,
        padding: int = 5
    ) -> Tuple[List[Dict], List[np.ndarray], np.ndarray]:
        """
        一步完成检测和色块提取
        
        Args:
            image: 输入图像
            confidence_threshold: 置信度阈值
            min_area: 最小色块面积
            max_detections: 最大检测数量
            padding: 色块提取时的边界扩展
            
        Returns:
            Tuple[List[Dict], List[np.ndarray], np.ndarray]: 
                (检测结果, 色块图像列表, 标注后的图像)
        """
        # 检测色块
        detections = self.detect_blocks(
            image, confidence_threshold, min_area, max_detections
        )
        
        # 提取色块
        color_blocks = self.extract_color_blocks(image, detections, padding)
        
        # 生成可视化图像
        annotated_image = self.visualize_detections(image, detections)
        
        return detections, color_blocks, annotated_image


# ===== 向后兼容的函数接口（已弃用） =====

def load_yolo_block_model(model_path: str = None) -> YOLO:
    """
    加载用于色块检测的YOLO模型
    
    Args:
        model_path: YOLO模型权重文件路径，默认为 core/block_detection/weights/best.pt
        
    Returns:
        YOLO模型实例
        
    Raises:
        FileNotFoundError: 当模型文件不存在时
    """
    warnings.warn(
        "load_yolo_block_model 函数已弃用，请使用 YOLOBlockDetector 类",
        DeprecationWarning,
        stacklevel=2
    )
    
    if model_path is None:
        # 默认模型路径
        model_path = "core/block_detection/weights/best.pt"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO色块模型文件未找到: {model_path}")
    
    print(f"正在加载YOLO色块检测模型: {model_path}")
    
    try:
        model = YOLO(model_path)
        print("✅ YOLO色块检测模型加载成功")
        return model
    except Exception as e:
        raise RuntimeError(f"加载YOLO色块模型失败: {str(e)}")


def detect_blocks_with_yolo(
    colorbar_image: np.ndarray,
    model: YOLO,
    confidence_threshold: float = 0.5,
    min_area: int = 50
) -> tuple[np.ndarray, list[np.ndarray], int]:
    """
    使用YOLO模型检测颜色条中的色块
    
    Args:
        colorbar_image: 颜色条图像（BGR格式的numpy数组）
        model: YOLO模型实例
        confidence_threshold: 检测置信度阈值 (0.0-1.0)
        min_area: 最小色块面积（像素数）
        
    Returns:
        tuple: (标注图像, 色块图像列表, 检测到的色块数量)
    """
    warnings.warn(
        "detect_blocks_with_yolo 函数已弃用，请使用 YOLOBlockDetector 类",
        DeprecationWarning,
        stacklevel=2
    )
    
    if colorbar_image.size == 0:
        print("⚠️ 输入的颜色条图像为空")
        return colorbar_image, [], 0
    
    print(f"🔍 开始使用YOLO检测色块，置信度阈值: {confidence_threshold}")
    
    # 运行YOLO推理
    try:
        results = model(colorbar_image, verbose=False)  # verbose=False 减少输出
    except Exception as e:
        print(f"❌ YOLO推理失败: {str(e)}")
        return colorbar_image, [], 0
    
    # 创建标注图像的副本
    annotated_image = colorbar_image.copy()
    color_blocks = []
    
    height, width = colorbar_image.shape[:2]
    
    # 处理检测结果
    for result in results:
        if result.boxes is None:
            continue
            
        boxes = result.boxes.cpu().numpy()
        
        for i, box in enumerate(boxes):
            confidence = float(box.conf[0])
            
            # 根据置信度阈值过滤
            if confidence < confidence_threshold:
                continue
            
            # 获取检测框坐标 (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            
            # 确保坐标在图像范围内
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            # 检查面积是否满足最小要求
            area = (x2 - x1) * (y2 - y1)
            if area < min_area:
                print(f"⚠️ 色块面积 {area} 小于最小阈值 {min_area}，跳过")
                continue
            
            # 提取色块图像
            color_block = colorbar_image[y1:y2, x1:x2]
            if color_block.size > 0:
                color_blocks.append(color_block)
                
                # 在标注图像上绘制检测框
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 添加标签文本
                label = f"Block {len(color_blocks)}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # 确保标签不会超出图像边界
                label_y = max(y1 - 10, label_size[1])
                cv2.putText(
                    annotated_image, 
                    label, 
                    (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2
                )
    
    block_count = len(color_blocks)
    print(f"✅ YOLO检测完成，共检测到 {block_count} 个色块")
    
    return annotated_image, color_blocks, block_count


def convert_pil_to_opencv(pil_image: Image.Image) -> np.ndarray:
    """
    将PIL图像转换为OpenCV格式
    
    Args:
        pil_image: PIL图像对象
        
    Returns:
        np.ndarray: BGR格式的OpenCV图像
    """
    # 转换为RGB numpy数组
    rgb_array = np.array(pil_image)
    
    # 如果是灰度图像，转换为3通道
    if len(rgb_array.shape) == 2:
        rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_GRAY2RGB)
    
    # 如果是RGBA，去掉alpha通道
    elif rgb_array.shape[2] == 4:
        rgb_array = rgb_array[:, :, :3]
    
    # 转换为BGR格式（OpenCV标准）
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    
    return bgr_array


def convert_opencv_to_pil(opencv_image: np.ndarray) -> Image.Image:
    """
    将OpenCV图像转换为PIL格式
    
    Args:
        opencv_image: BGR格式的OpenCV图像
        
    Returns:
        Image.Image: PIL图像对象
    """
    # 转换为RGB格式
    rgb_array = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    
    # 转换为PIL图像
    pil_image = Image.fromarray(rgb_array)
    
    return pil_image