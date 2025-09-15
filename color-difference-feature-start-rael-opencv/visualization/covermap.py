import logging

import cv2
import numpy as np

logger = logging.getLogger("covermap")


def calculate_difference_map(
    template_roi: np.ndarray, aligned_roi: np.ndarray, method: str = "absolute"
) -> np.ndarray:
    """
    计算两个ROI之间的差异图

    参数:
        template_roi: 模板ROI
        aligned_roi: 对齐后的ROI
        method: 差异计算方法, 'absolute'(绝对差), 'squared'(平方差), 'structural'(结构相似性)

    返回:
        np.ndarray: 差异图，越亮的区域差异越大
    """
    if template_roi is None or aligned_roi is None:
        return None

    # 确保ROI具有相同的形状
    if template_roi.shape != aligned_roi.shape:
        logger.warning(
            f"ROI形状不匹配: 模板={template_roi.shape}, 对齐={aligned_roi.shape}"
        )
        # 调整到最小共同尺寸
        min_h = min(template_roi.shape[0], aligned_roi.shape[0])
        min_w = min(template_roi.shape[1], aligned_roi.shape[1])
        template_roi = template_roi[:min_h, :min_w]
        aligned_roi = aligned_roi[:min_h, :min_w]

    # 转换为浮点型以便计算
    template_float = template_roi.astype(np.float32)
    aligned_float = aligned_roi.astype(np.float32)

    # 根据指定方法计算差异
    if method == "absolute":
        # 绝对差异
        diff = cv2.absdiff(template_float, aligned_float)
        # 对多通道图像，取通道最大值
        if len(diff.shape) == 3:
            diff = np.max(diff, axis=2)

    elif method == "squared":
        # 平方差异
        diff = cv2.absdiff(template_float, aligned_float)
        diff = diff * diff
        # 对多通道图像，取通道最大值
        if len(diff.shape) == 3:
            diff = np.max(diff, axis=2)

    elif method == "structural":
        # 结构相似性 (负相关区域会更明显)
        if len(template_roi.shape) == 3:
            # 转换为灰度
            template_gray = cv2.cvtColor(template_roi, cv2.COLOR_BGR2GRAY)
            aligned_gray = cv2.cvtColor(aligned_roi, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template_roi
            aligned_gray = aligned_roi

        # 计算SSIM
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        template_gray = template_gray.astype(np.float32)
        aligned_gray = aligned_gray.astype(np.float32)

        # 均值
        mu_x = cv2.GaussianBlur(template_gray, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(aligned_gray, (11, 11), 1.5)

        # 方差和协方差
        mu_x_sq = mu_x**2
        mu_y_sq = mu_y**2
        mu_xy = mu_x * mu_y

        sigma_x_sq = cv2.GaussianBlur(template_gray**2, (11, 11), 1.5) - mu_x_sq
        sigma_y_sq = cv2.GaussianBlur(aligned_gray**2, (11, 11), 1.5) - mu_y_sq
        sigma_xy = cv2.GaussianBlur(template_gray * aligned_gray, (11, 11), 1.5) - mu_xy

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
            (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
        )

        # 转换为差异图 (1-SSIM)，值越大表示差异越大
        diff = 1.0 - ssim_map

    else:
        logger.warning(f"不支持的差异计算方法: {method}")
        # 默认使用绝对差异
        diff = cv2.absdiff(template_float, aligned_float)
        if len(diff.shape) == 3:
            diff = np.max(diff, axis=2)

    # 标准化并转换为可视化图像
    diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    diff_vis = diff_norm.astype(np.uint8)

    # 应用伔彩色映射以增强可视化效果
    diff_color = cv2.applyColorMap(diff_vis, cv2.COLORMAP_JET)

    return diff_color
