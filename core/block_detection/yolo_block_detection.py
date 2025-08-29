"""
YOLO色块检测模块
使用训练好的YOLO模型直接检测色块
"""

import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


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
        raise RuntimeError(f"加载YOLO色块模型失败: {str(e)}") from e


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


def test_yolo_block_detection(image_path: str, model_path: str = None):
    """
    测试YOLO色块检测功能

    Args:
        image_path: 测试图像路径
        model_path: 模型路径
    """
    try:
        # 加载检测器
        detector = load_yolo_block_model(model_path)

        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 无法读取图像: {image_path}")
            return

        print(f"📷 测试图像: {image_path}")
        print(f"📐 图像尺寸: {image.shape}")

        # 检测色块
        annotated_image, color_blocks, block_count = detect_blocks_with_yolo(
            image, detector
        )

        print(f"🎯 检测结果: {block_count} 个色块")

        # 保存结果
        output_path = image_path.replace('.', '_yolo_result.')
        cv2.imwrite(output_path, annotated_image)
        print(f"💾 结果已保存到: {output_path}")

    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")


if __name__ == "__main__":
    # 示例测试
    test_image = "test_colorbar.jpg"  # 替换为实际的测试图像路径
    if os.path.exists(test_image):
        test_yolo_block_detection(test_image)
    else:
        print("请提供测试图像路径进行测试")