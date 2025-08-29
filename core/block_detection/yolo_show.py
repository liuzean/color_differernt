# YOLO色板检测模块 - 用于Gradio集成
# 本模块使用YOLO深度学习模型检测图像中的色板区域

import os

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


def load_yolo_model(model_path: str = None) -> YOLO:
    """
    加载用于色板检测的YOLO模型。
    自动搜索多个可能的模型路径，如果未指定路径的话。

    Args:
        model_path: YOLO模型权重文件路径。如果为None，使用默认路径。

    Returns:
        YOLO模型实例
    """
    if model_path is None:
        # 尝试多个可能的模型路径
        possible_paths = [
            "./core/block_detection/weights/best0710.pt",
            "./runs/train/custom_colorbar2/weights/best.pt",
            "./weights/best.pt",
        ]

        # 查找第一个存在的模型文件
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
    使用YOLO模型在图像中检测色板。
    对检测框进行扩展以确保完整捕获色板区域。

    Args:
        image: 输入图像，numpy数组格式（BGR）
        model: YOLO模型实例
        box_expansion: 检测框扩展的像素数
        confidence_threshold: 检测的最小置信度阈值

    Returns:
        返回元组：(标注图像, 检测框列表, 置信度列表, 色板片段)
    """
    # 运行YOLO推理
    results = model(image)

    # 创建标注图像
    img_with_boxes = image.copy()

    boxes_list = []        # 存储检测框坐标
    confidences_list = []  # 存储置信度分数
    colorbar_segments = [] # 存储提取的色板片段

    height, width = image.shape[:2]

    # 处理检测结果
    for result in results:
        boxes = result.boxes.cpu().numpy()

        for _i, box in enumerate(boxes):
            confidence = float(box.conf[0])

            # 根据置信度阈值过滤
            if confidence < confidence_threshold:
                continue

            # 获取检测框坐标
            x1, y1, x2, y2 = box.xyxy[0].astype(int)

            # 扩展检测框指定像素数
            x1_exp = max(0, x1 - box_expansion)
            y1_exp = max(0, y1 - box_expansion)
            x2_exp = min(width, x2 + box_expansion)
            y2_exp = min(height, y2 + box_expansion)

            # 在标注图像上绘制矩形框
            cv2.rectangle(
                img_with_boxes, (x1_exp, y1_exp), (x2_exp, y2_exp), (0, 255, 0), 2
            )

            # 添加置信度标签
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

            # 提取色板片段
            colorbar_segment = image[y1_exp:y2_exp, x1_exp:x2_exp]

            # 存储结果
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
    从PIL图像检测色板并返回PIL图像。
    用于Gradio兼容性的包装函数。

    Args:
        pil_image: PIL图像输入
        model_path: YOLO模型权重文件路径
        box_expansion: 检测框扩展像素数
        confidence_threshold: 检测的最小置信度

    Returns:
        返回元组：(标注PIL图像, 色板PIL片段, 数量, 置信度列表)
    """
    if pil_image is None:
        return None, [], 0, []

    # 加载YOLO模型
    try:
        model = load_yolo_model(model_path)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return pil_image, [], 0, []

    # 将PIL转换为OpenCV格式
    opencv_image = np.array(pil_image)
    if len(opencv_image.shape) == 3:
        # 从RGB转换为BGR
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

    # 运行YOLO检测
    try:
        annotated_cv, boxes, confidences, segments_cv = detect_colorbars_yolo(
            opencv_image,
            model,
            box_expansion=box_expansion,
            confidence_threshold=confidence_threshold,
        )

        # 将结果转换回PIL格式
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_cv, cv2.COLOR_BGR2RGB))

        # 转换色板片段为PIL格式
        segments_pil = []
        for segment_cv in segments_cv:
            if segment_cv.size > 0:  # 检查片段是否非空
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
    分析检测到的色板片段中的颜色。
    提供基本的颜色统计分析。

    Args:
        colorbar_segment: 色板图像片段（BGR格式）

    Returns:
        包含颜色分析结果的字典
    """
    if colorbar_segment.size == 0:
        return {"error": "Empty colorbar segment"}

    # 转换为RGB进行分析
    rgb_segment = cv2.cvtColor(colorbar_segment, cv2.COLOR_BGR2RGB)

    # 获取主导色（简化分析）
    pixels = rgb_segment.reshape(-1, 3)

    # 计算基本统计信息
    mean_color = np.mean(pixels, axis=0)  # 平均颜色
    std_color = np.std(pixels, axis=0)    # 颜色标准差

    # 获取唯一颜色（简化版）
    unique_colors = np.unique(pixels, axis=0)

    return {
        "mean_rgb": mean_color.tolist(),        # 平均RGB值
        "std_rgb": std_color.tolist(),          # RGB标准差
        "unique_colors_count": len(unique_colors), # 唯一颜色数量
        "segment_shape": colorbar_segment.shape,   # 片段尺寸
        "total_pixels": len(pixels),               # 总像素数
    }
