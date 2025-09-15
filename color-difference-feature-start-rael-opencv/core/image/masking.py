import logging
import os

import cv2
import numpy as np

# 获取记录器
logger = logging.getLogger(__name__)


class MaskingProcessor:
    """
    图像掩码处理类，提供Alpha通道提取和背景移除功能
    """

    @staticmethod
    def extract_alpha_mask(
        image: str | np.ndarray, output_path: str | None = None
    ) -> np.ndarray | None:
        """
        提取图像的透明通道掩码，如果没有则返回None

        参数:
            image: 图像路径或numpy数组
            output_path: 输出文件路径，None则不保存

        返回:
            alpha_mask: Alpha通道掩码(前景为白，背景为黑)，无Alpha通道则返回None
        """
        try:
            # 处理图像输入
            if isinstance(image, str):
                if not os.path.exists(image):
                    logger.error(f"图像文件不存在: {image}")
                    return None

                image = cv2.imread(image, cv2.IMREAD_UNCHANGED)

                if image is None:
                    logger.error("无法读取图像")
                    return None
            elif not isinstance(image, np.ndarray):
                logger.error(f"无效的图像类型: {type(image)}")
                return None

            # 提取Alpha通道
            if len(image.shape) == 3 and image.shape[2] == 4:
                # 有Alpha通道
                alpha_mask = image[:, :, 3]
                logger.info("已提取图像Alpha通道")

                # 保存掩码
                if output_path is not None:
                    # 确保输出目录存在
                    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                    cv2.imwrite(output_path, alpha_mask)
                    logger.info(f"掩码已保存至: {output_path}")

                return alpha_mask
            else:
                # 无Alpha通道
                logger.info("图像没有Alpha通道")
                return None
        except Exception as e:
            logger.error(f"处理Alpha通道时出错: {str(e)}")
            return None

    @staticmethod
    def remove_background(
        image: str | np.ndarray,
        mask: str | np.ndarray,
        background_color: tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray | None:
        """
        使用掩码移除图像背景，并用指定颜色填充背景区域

        参数:
            image: 彩色图像（文件路径或numpy数组）
            mask: 掩码图像（文件路径或numpy数组）
            background_color: 背景颜色，默认为白色(255,255,255)

        返回:
            处理后的图像（numpy数组）
        """
        try:
            # 首先检查输入是否为None
            if image is None:
                logger.error("图像输入为None")
                return None

            if mask is None:
                logger.error("掩码输入为None")
                return None

            # 处理图像输入
            if isinstance(image, str):
                if not os.path.exists(image):
                    logger.error(f"图像文件不存在: {image}")
                    return None
                image_data = cv2.imread(image)
                if image_data is None:
                    logger.error(f"无法读取图像: {image}")
                    return None
            elif isinstance(image, np.ndarray):
                image_data = image.copy()
            else:
                logger.error(f"无效的图像类型: {type(image)}")
                return None

            h, w = image_data.shape[:2]

            # 处理掩码输入
            if isinstance(mask, str):
                # 掩码是文件路径
                if not os.path.exists(mask):
                    logger.error(f"掩码文件不存在: {mask}")
                    return None

                mask_data = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
                if mask_data is None:
                    logger.error(f"无法读取掩码: {mask}")
                    return None
            elif isinstance(mask, np.ndarray):
                # 掩码是numpy数组
                if len(mask.shape) == 3:
                    if mask.shape[2] == 3:  # BGR图像
                        mask_data = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    elif mask.shape[2] == 4:  # BGRA图像
                        mask_data = mask[:, :, 3]  # 提取Alpha通道
                    else:
                        logger.error(f"不支持的掩码格式: {mask.shape}")
                        return None
                elif len(mask.shape) == 2:  # 已经是灰度图
                    mask_data = mask.copy()
                else:
                    logger.error(f"不支持的掩码格式: {mask.shape}")
                    return None
            else:
                logger.error(f"无效的掩码类型: {type(mask)}")
                return None

            # 调整掩码尺寸
            if mask_data.shape[:2] != (h, w):
                logger.info(f"调整掩码尺寸: {mask_data.shape[:2]} -> {(h, w)}")
                mask_data = cv2.resize(mask_data, (w, h))

            # 二值化掩码确保值为0或255
            _, mask_data = cv2.threshold(mask_data, 127, 255, cv2.THRESH_BINARY)

            # 创建指定颜色的背景
            bg = np.ones((h, w, 3), dtype=np.uint8)
            bg[:, :, 0] = background_color[0]  # B
            bg[:, :, 1] = background_color[1]  # G
            bg[:, :, 2] = background_color[2]  # R

            # 计算Alpha混合
            alpha = mask_data.astype(float) / 255.0
            alpha_3ch = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

            # 应用掩码: 前景*掩码 + 背景*(1-掩码)
            result = cv2.convertScaleAbs(image_data * alpha_3ch + bg * (1 - alpha_3ch))

            logger.info("背景已成功替换为指定颜色")
            return result

        except Exception as e:
            logger.error(f"背景移除过程中出错: {str(e)}")
            return None


# if __name__ == "__main__":
#     # 示例用法
#     image_path = "results/template_20250513_113409/temp/butterfly.png"
#     mask_output_path = "results/template_20250513_113409/temp/alpha_mask_output.png"
#     background_image_path = "results/template_20250513_113409/temp/processed_20250513_113419.png"
#     result_output_path = "results/template_20250513_113409/temp/result_image.png"

#     # 提取Alpha通道掩码
#     alpha_mask = MaskingProcessor.extract_alpha_mask(image_path, None)

#     # 移除背景
#     if alpha_mask is not None:
#         result_image = MaskingProcessor.remove_background(background_image_path, alpha_mask)
#         if result_image is not None:
#             cv2.imwrite(result_output_path, result_image)
#             print(f"结果图像已保存到: {result_output_path}")
#     else:
#         print("图像没有Alpha通道，无法提取掩码")
