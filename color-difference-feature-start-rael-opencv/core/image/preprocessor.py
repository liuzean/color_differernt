import logging
import traceback
from enum import Enum

import cv2
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ImagePreprocessor")


class ResizeMethod(Enum):
    """调整大小的方法枚举"""

    EXACT = 0  # 精确调整到指定尺寸
    PRESERVE_ASPECT = 1  # 保持宽高比
    FIT_CONTAIN = 2  # 包含图像，可能有填充
    FIT_COVER = 3  # 覆盖区域，可能有裁剪


class FilterType(Enum):
    """图像滤波类型枚举"""

    GAUSSIAN = 0  # 高斯滤波
    MEDIAN = 1  # 中值滤波
    BILATERAL = 2  # 双边滤波
    BOX = 3  # 盒式滤波
    SHARPEN = 4  # 锐化


class ImagePreprocessor:
    """
    图像预处理类
    提供图像尺寸调整、对比度/亮度调整、滤波与平滑等功能
    """

    def __init__(self):
        """初始化图像预处理器"""
        logger.info("图像预处理器初始化")

    def resize_image(
        self, image: np.ndarray, maxdim: int = 1000, **kwargs
    ) -> np.ndarray:
        h, w = image.shape[:2]
        if max(h, w) > maxdim:
            scale = maxdim / max(h, w)
            new_size = (int(w * scale), int(h * scale))
            image = cv2.resize(image, new_size)
        return image

    # def preprocess_image(img):
    #     # 确保图像尺寸合理 (太大的图像可能匹配慢且不准确)
    #     max_dim = 1000
    #     h, w = img.shape[:2]
    #     if max(h, w) > max_dim:
    #         scale = max_dim / max(h, w)
    #         new_size = (int(w * scale), int(h * scale))
    #         img = cv2.resize(img, new_size)

    #     # 转为灰度图
    #     if len(img.shape) == 3:
    #         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     else:
    #         gray = img

    #     # 对比度增强
    #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #     enhanced = clahe.apply(gray)

    #     return enhanced, img  # 返回增强后的灰度图和原始(可能调整大小后)的彩色图

    def apply_filter(
        self, image: np.ndarray, filter_type: FilterType | str | int, **kwargs
    ) -> np.ndarray:
        """
        应用滤波器

        参数:
            image: 输入图像
            filter_type: 滤波器类型，可以是FilterType枚举或其整数值或字符串名称
            **kwargs: 滤波器特定参数

        返回:
            np.ndarray: 滤波后的图像
        """
        if image is None:
            return None

        # 将filter_type参数转换为FilterType枚举类型
        if isinstance(filter_type, str):
            try:
                filter_type = FilterType[filter_type.upper()]
            except KeyError:
                logger.error(f"不支持的滤波器类型: {filter_type}")
                return image
        elif isinstance(filter_type, int):
            try:
                filter_type = FilterType(filter_type)
            except ValueError:
                logger.error(f"无效的滤波器类型ID: {filter_type}")
                return image

        # 根据滤波器类型应用不同的处理
        if filter_type == FilterType.GAUSSIAN:
            # 高斯滤波
            ksize = kwargs.get("kernel_size", (5, 5))
            sigma_x = kwargs.get("sigma_x", 0)
            sigma_y = kwargs.get("sigma_y", 0)
            filtered = cv2.GaussianBlur(image, ksize, sigma_x, sigma_y)

        elif filter_type == FilterType.MEDIAN:
            # 中值滤波
            ksize = kwargs.get("kernel_size", 5)
            filtered = cv2.medianBlur(image, ksize)

        elif filter_type == FilterType.BILATERAL:
            # 双边滤波
            d = kwargs.get("d", 9)
            sigma_color = kwargs.get("sigma_color", 75)
            sigma_space = kwargs.get("sigma_space", 75)
            filtered = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

        elif filter_type == FilterType.BOX:
            # 盒式滤波
            ksize = kwargs.get("kernel_size", (5, 5))
            normalize = kwargs.get("normalize", True)
            filtered = cv2.boxFilter(image, -1, ksize, normalize=normalize)

        elif filter_type == FilterType.SHARPEN:
            # 锐化
            strength = kwargs.get("strength", 1.0)

            # 使用拉普拉斯算子进行锐化
            if len(image.shape) == 3:
                # 彩色图像
                blurred = cv2.GaussianBlur(image, (0, 0), 3)
                filtered = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
            else:
                # 灰度图像
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                filtered = cv2.filter2D(image, -1, kernel)

        else:
            logger.warning(f"未实现的滤波器类型: {filter_type}")
            return image

        return filtered

    def denoise_image(
        self, image: np.ndarray, method: str = "fastNl", strength: float = 10.0
    ) -> np.ndarray:
        """
        图像降噪处理

        参数:
            image: 输入图像
            method: 降噪方法，可选 'fastNl'(快速非局部均值), 'nl'(非局部均值), 'gaussian', 'bilateral'
            strength: 降噪强度

        返回:
            np.ndarray: 降噪后的图像
        """
        if image is None:
            return None

        try:
            if method == "fastNl":
                # 快速非局部均值降噪
                if len(image.shape) == 3:
                    h_param = strength  # 亮度滤波参数
                    denoised = cv2.fastNlMeansDenoisingColored(
                        image, None, h_param, h_param, 7, 21
                    )
                else:
                    h_param = strength
                    denoised = cv2.fastNlMeansDenoising(image, None, h_param, 7, 21)

            elif method == "nl":
                # 非局部均值降噪
                if len(image.shape) == 3:
                    h_param = strength
                    denoised = cv2.fastNlMeansDenoisingColored(
                        image, None, h_param, h_param, 7, 21
                    )
                else:
                    h_param = strength
                    denoised = cv2.fastNlMeansDenoising(image, None, h_param, 7, 21)

            elif method == "gaussian":
                # 高斯降噪
                kernel_size = int(max(3, strength / 2)) * 2 + 1  # 确保是奇数
                denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

            elif method == "bilateral":
                # 双边滤波降噪
                d = int(max(5, strength))
                sigma_color = strength * 2
                sigma_space = strength
                denoised = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

            else:
                logger.warning(f"不支持的降噪方法: {method}")
                return image

            return denoised

        except Exception as e:
            error_msg = f"图像降噪失败: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            return image


class ImageRotator:
    """
    图像旋转和镜像处理类
    提供90°、180°、270°旋转以及水平/垂直镜像功能
    """

    def __init__(self):
        """初始化图像旋转器"""
        logger.info("图像旋转器初始化")

    def rotate_90(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        将图像顺时针旋转90度

        参数:
            image: 输入图像

        返回:
            Tuple[np.ndarray, dict]: 旋转后的图像和旋转信息
        """
        if image is None:
            return None, {"rotation": 0}

        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        info = {"rotation": 90}
        return rotated, info

    def rotate_180(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        将图像旋转180度

        参数:
            image: 输入图像

        返回:
            Tuple[np.ndarray, dict]: 旋转后的图像和旋转信息
        """
        if image is None:
            return None, {"rotation": 0}

        rotated = cv2.rotate(image, cv2.ROTATE_180)
        info = {"rotation": 180}
        return rotated, info

    def rotate_270(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        将图像顺时针旋转270度（逆时针90度）

        参数:
            image: 输入图像

        返回:
            Tuple[np.ndarray, dict]: 旋转后的图像和旋转信息
        """
        if image is None:
            return None, {"rotation": 0}

        rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        info = {"rotation": 270}
        return rotated, info

    def flip_horizontal(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        水平镜像图像

        参数:
            image: 输入图像

        返回:
            Tuple[np.ndarray, dict]: 镜像后的图像和镜像信息
        """
        if image is None:
            return None, {"mirror": None}

        flipped = cv2.flip(image, 1)  # 1表示水平翻转
        info = {"mirror": "horizontal"}
        return flipped, info

    def flip_vertical(self, image: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        垂直镜像图像

        参数:
            image: 输入图像

        返回:
            Tuple[np.ndarray, dict]: 镜像后的图像和镜像信息
        """
        if image is None:
            return None, {"mirror": None}

        flipped = cv2.flip(image, 0)  # 0表示垂直翻转
        info = {"mirror": "vertical"}
        return flipped, info

    def transform(
        self, image: np.ndarray, rotation: int = 0, mirror: str = None
    ) -> tuple[np.ndarray, dict]:
        """
        根据指定的旋转角度和镜像方式变换图像

        参数:
            image: 输入图像
            rotation: 旋转角度，可选值为0、90、180、270
            mirror: 镜像方式，可选值为"horizontal"、"vertical"或None

        返回:
            Tuple[np.ndarray, dict]: 变换后的图像和变换信息
        """
        if image is None:
            return None, {"rotation": 0, "mirror": None}

        result = image.copy()
        info = {"rotation": rotation, "mirror": mirror}

        # 先进行镜像
        if mirror == "horizontal":
            result = cv2.flip(result, 1)
        elif mirror == "vertical":
            result = cv2.flip(result, 0)

        # 再进行旋转
        if rotation == 90:
            result = cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            result = cv2.rotate(result, cv2.ROTATE_180)
        elif rotation == 270:
            result = cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return result, info


# # 示例用法
# if __name__ == "__main__":
#     # 创建预处理器实例
#     preprocessor = ImagePreprocessor()

#     # 加载测试图像
#     image_path = "path/to/test/image.jpg"
#     try:
#         image = cv2.imread(image_path)

#         if image is None:
#             print(f"无法加载图像: {image_path}")
#         else:
#             # 预处理配置
#             resize_config = {
#                 'width': 800,
#                 'height': None,
#                 'method': ResizeMethod.PRESERVE_ASPECT
#             }

#             enhance_config = {
#                 'brightness': 10,
#                 'contrast': 1.2,
#                 'clahe': True
#             }

#             filter_ops = [
#                 {'type': FilterType.BILATERAL, 'params': {'d': 9, 'sigma_color': 75, 'sigma_space': 75}},
#                 {'type': FilterType.SHARPEN, 'params': {'strength': 0.5}}
#             ]

#             # 应用预处理
#             processed = preprocessor.preprocess(
#                 image,
#                 resize=resize_config,
#                 enhance=enhance_config,
#                 filter_ops=filter_ops
#             )

#             # 保存结果
#             cv2.imwrite("preprocessed_image.jpg", processed)
#             print("预处理完成并保存结果")

#     except Exception as e:
#         print(f"处理过程中出现错误: {str(e)}")
#         traceback.print_exc()
