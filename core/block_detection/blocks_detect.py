"""
当前文件是对某个具体的文件进行色块检测的脚本。
它使用了一个简单的边缘检测算法来检测图像中的矩形块。
该脚本可以处理多种输入格式，包括文件路径、numpy数组和PIL图像。
它还可以返回检测到的块图像列表，并支持保存检测结果到指定目录。
而不是对另一个py文件提取的色板进行检测
"""


# 导入必要的库
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
    使用简单边缘检测算法在图像中检测矩形块。
    简化的方法，对包括浅色在内的所有颜色都有更好的效果。

    Args:
        image_path: 输入图像的路径
        output_dir: 保存检测到的块的目录（可选）
        area_threshold: 检测块的最小面积
        aspect_ratio_threshold: 矩形块的最小长宽比
        min_square_size: 检测块的最小宽度和高度（像素）
        return_individual_blocks: 是否返回单独的块图像

    Returns:
        返回元组：(带框的结果图像, 块图像列表, 块数量)
    """
    # 读取图像 - 支持多种输入格式
    if isinstance(image_path, str):
        # 从文件路径读取图像
        image = cv2.imread(image_path)
    elif isinstance(image_path, np.ndarray):
        # 直接使用numpy数组
        image = image_path
    else:
        # 处理PIL图像输入
        image = np.array(image_path)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # 将RGB转换为BGR（OpenCV格式）
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if image is None:
        raise ValueError("Could not load image")

    # 如果需要，创建输出目录
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 简单有效的处理流水线
    # 1. 转换为灰度图像 - 简化后续处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. 应用高斯模糊以减少噪声
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 3. 边缘检测 - 这对检测颜色之间的边界很有效
    edges = cv2.Canny(blurred, 30, 100, apertureSize=3)

    # 4. 膨胀边缘以闭合间隙
    kernel = np.ones((2, 2), np.uint8)  # 定义形态学操作的核
    edges = cv2.dilate(edges, kernel, iterations=1)

    # 5. 闭合剩余的间隙
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 6. 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 准备结果图像
    result = image.copy()
    block_images = []  # 存储检测到的块图像
    block_count = 0    # 计数器

    # 处理每个轮廓
    for contour in contours:
        # 计算轮廓面积
        area = cv2.contourArea(contour)

        # 如果面积太小则跳过
        if area < area_threshold:
            continue

        # 获取边界矩形
        x, y, w, h = cv2.boundingRect(contour)

        # 检查尺寸约束
        if w < min_square_size or h < min_square_size:
            continue

        # 计算长宽比
        aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0

        # 检查长宽比
        if aspect_ratio < aspect_ratio_threshold:
            continue

        # 在结果图像上绘制矩形框
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 添加块编号标签
        cv2.putText(
            result,
            f"{block_count}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

        # 如果需要，提取块图像
        if return_individual_blocks:
            # 从原图像中裁剪出块
            block_image = image[y : y + h, x : x + w]
            block_images.append(block_image)

            # 如果指定了输出目录，则保存块图像
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
    从PIL图像检测块并返回PIL图像。
    用于Gradio兼容性的包装函数。

    Args:
        pil_image: PIL图像输入
        area_threshold: 检测块的最小面积
        aspect_ratio_threshold: 矩形块的最小长宽比
        min_square_size: 检测块的最小宽度和高度（像素）

    Returns:
        返回元组：(结果PIL图像, 块PIL图像列表, 块数量)
    """
    # 将PIL转换为OpenCV格式
    opencv_image = np.array(pil_image)
    if len(opencv_image.shape) == 3:
        # 从RGB转换为BGR
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

    # 运行检测
    result_cv, block_images_cv, count = detect_blocks(
        opencv_image,
        output_dir=None,  # 不保存文件
        area_threshold=area_threshold,
        aspect_ratio_threshold=aspect_ratio_threshold,
        min_square_size=min_square_size,
        return_individual_blocks=True,
    )

    # 将结果转换回PIL格式
    result_pil = Image.fromarray(cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB))

    # 转换块图像为PIL格式
    block_images_pil = []
    for block_cv in block_images_cv:
        block_pil = Image.fromarray(cv2.cvtColor(block_cv, cv2.COLOR_BGR2RGB))
        block_images_pil.append(block_pil)

    return result_pil, block_images_pil, count


# 独立使用时的原始脚本功能
if __name__ == "__main__":
    # 原始脚本行为
    image_path = "./datasets/7_conf0_84_0.png"  # 测试图像路径

    try:
        # 执行块检测
        result_image, block_images, count = detect_blocks(
            image_path,
            output_dir="detected_blocks",
            area_threshold=100,
            aspect_ratio_threshold=0.7,
        )

        print(f"Total {count} blocks detected and saved to detected_blocks/")

        # 显示结果
        cv2.imshow("Result", result_image)
        cv2.waitKey(0)  # 等待按键
        cv2.destroyAllWindows()  # 关闭所有窗口

        # 保存检测结果
        cv2.imwrite("detected_squares.png", result_image)

    except Exception as e:
        print(f"Error: {e}")
