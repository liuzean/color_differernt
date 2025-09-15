"""
如果要继续使用 OpenCV 处理图像，通常使用 load_image（保持 BGR 格式）
如果要将图像传递给其他库（如 matplotlib、PIL 等）或需要标准 RGB 格式，应使用 load_image_rgb
"""

import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """
    从指定路径读取并加载图像。

    参数:
        image_path (str): 图像文件路径

    返回:
        np.ndarray: BGR格式的图像数组

    异常:
        FileNotFoundError: 如果图像文件不存在
        Exception: 如果图像无法加载
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise Exception(f"无法从 {image_path} 加载图像")
        return image
    except FileNotFoundError:
        raise FileNotFoundError(f"未找到图像文件: {image_path}")
    except Exception as e:
        raise Exception(f"加载图像时出错: {str(e)}")


def load_image_rgb(image_path: str) -> np.ndarray:
    """
    从指定路径读取并加载RGB格式的图像。

    参数:
        image_path (str): 图像文件路径

    返回:
        np.ndarray: RGB格式的图像数组
    """
    image = load_image(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# if __name__ == "__main__":
#     # 测试加载图像
#     image_path = "results/template_20250513_113409/temp/butterfly.png"
#     try:
#         image = load_image_rgb(image_path)
#     except Exception as e:
#         print(f"加载图像时出错: {e}")
