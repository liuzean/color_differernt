import logging
import traceback
from dataclasses import dataclass

import cv2
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ImageAlignment")


@dataclass
class AlignmentResult:
    """图像对齐结果数据类"""

    success: bool
    message: str = ""
    aligned_image: np.ndarray | None = None
    roi_template: np.ndarray | None = None
    roi_aligned: np.ndarray | None = None
    homography: np.ndarray | None = None
    roi_coords: dict[str, list[int]] | None = None
    similarity_score: float = 0.0


class ImageAlignment:
    """
    图像对齐类
    - 使用单应性矩阵对齐图像
    - 提取感兴趣区域(ROI)
    - 评估对齐质量
    """

    def __init__(self, detector=None, matcher=None):
        """
        初始化图像对齐器

        参数:
            detector: 特征检测器实例
            matcher: 特征匹配器实例
        """
        try:
            self.detector = detector
            self.matcher = matcher

            if self.detector:
                logger.info(
                    f"图像对齐器初始化成功: 检测器={self.detector.get_name()}, 匹配器={self.matcher.get_name() if self.matcher else 'None'}"
                )
            else:
                logger.warning("图像对齐器初始化时未提供检测器和匹配器")

        except Exception as e:
            error_msg = f"图像对齐器初始化失败: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            raise RuntimeError(error_msg)

    def align_images(
        self,
        template_img: np.ndarray,
        target_img: np.ndarray,
        roi: dict | None = None,
        ratio_test: bool = True,
        ratio_threshold: float = 0.75,
        min_matches: int = 4,
        max_iteration: int = 2000,
        ransac_reproj_threshold: float = 5.0,
        return_roi_only: bool = False,
    ) -> AlignmentResult:
        """
        对齐图像并提取ROI

        参数:
            template_img: 模板图像
            target_img: 待对齐的目标图像
            roi: 感兴趣区域参数，格式为 {'x': x, 'y': y, 'width': w, 'height': h} 或 {'points': [(x1,y1), (x2,y2), ...]}
                 如果未提供，则使用整个模板图像
            ratio_test: 是否使用Lowe比率测试
            ratio_threshold: 比率测试阈值
            min_matches: 最小匹配点数量
            max_iteration: RANSAC最大迭代次数
            ransac_reproj_threshold: RANSAC重投影阈值
            return_roi_only: 如果为True，仅返回ROI，不返回完整对齐图像

        返回:
            AlignmentResult: 对齐结果对象
        """
        result = AlignmentResult(success=False)

        # 检查检测器和匹配器是否有效
        if self.detector is None or self.matcher is None:
            result.message = "检测器或匹配器未初始化"
            logger.error(result.message)
            return result

        # 参数验证
        if template_img is None or target_img is None:
            result.message = "输入图像无效"
            logger.error(result.message)
            return result

        try:
            # 1. 检测特征点
            kp_template, desc_template = self.detector.detect(template_img)
            kp_target, desc_target = self.detector.detect(target_img)

            if len(kp_template) < min_matches or len(kp_target) < min_matches:
                result.message = f"检测到的特征点不足: 模板={len(kp_template)}, 目标={len(kp_target)}"
                logger.warning(result.message)
                return result

            logger.info(f"检测到特征点: 模板={len(kp_template)}, 目标={len(kp_target)}")

            # 2. 匹配特征点
            if ratio_test:
                # KNN匹配 + Lowe比率测试
                raw_matches = self.matcher.match(desc_target, desc_template, k=2)
                good_matches = self.matcher.filter_matches(
                    raw_matches, ratio=ratio_threshold
                )
            else:
                # 简单匹配
                good_matches = self.matcher.match(desc_target, desc_template, k=1)

            if len(good_matches) < min_matches:
                result.message = f"有效匹配点数量不足: {len(good_matches)}"
                logger.warning(result.message)
                return result

            logger.info(f"找到 {len(good_matches)} 个有效匹配点")

            # 3. 计算单应性矩阵
            H, mask = self.compute_homography(
                kp_target,
                kp_template,
                good_matches,
                min_matches,
                max_iteration,
                ransac_reproj_threshold,
            )

            if H is None:
                result.message = "无法计算有效的单应性矩阵"
                logger.warning(result.message)
                return result

            inliers_count = np.sum(mask) if mask is not None else 0
            logger.info(f"计算单应性矩阵成功, 内点数量: {inliers_count}")

            # 4. 计算相似度得分
            similarity_score = self.compute_alignment_score(good_matches, mask)

            # 5. 使用单应性矩阵对齐图像
            h, w = template_img.shape[:2]
            aligned_image = cv2.warpPerspective(target_img, H, (w, h))

            # 6. 提取ROI
            roi_template, roi_aligned, roi_coords = self.extract_roi(
                template_img, aligned_image, roi
            )

            # 7. 填充结果
            result.success = True
            result.message = "图像对齐成功"
            result.homography = H
            result.similarity_score = similarity_score
            result.roi_template = roi_template
            result.roi_aligned = roi_aligned
            result.roi_coords = roi_coords

            if not return_roi_only:
                result.aligned_image = aligned_image

            return result

        except Exception as e:
            error_msg = f"图像对齐失败: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            result.message = error_msg
            return result

    def align_with_computed_homography(
        self,
        template_img: np.ndarray,
        target_img: np.ndarray,
        homography: np.ndarray,
        mask: np.ndarray | None = None,
        matches: list | None = None,
        roi: dict | None = None,
        return_roi_only: bool = False,
    ) -> AlignmentResult:
        """
        使用预先计算的单应性矩阵和掩码对齐图像

        参数:
            template_img: 模板图像
            target_img: 待对齐的目标图像
            homography: 预先计算的单应性矩阵
            mask: 内点掩码(可选)
            matches: 匹配点列表(可选，用于计算相似度得分)
            roi: 感兴趣区域参数
            return_roi_only: 如果为True，仅返回ROI，不返回完整对齐图像

        返回:
            AlignmentResult: 对齐结果对象
        """
        result = AlignmentResult(success=False)

        # 参数验证
        if template_img is None or target_img is None or homography is None:
            result.message = "输入参数无效"
            logger.error(result.message)
            return result

        try:
            # 1. 计算相似度得分(如果提供了matches和mask)
            similarity_score = 0.0
            if matches is not None:
                similarity_score = self.compute_alignment_score(matches, mask)

            # 2. 使用单应性矩阵对齐图像
            h, w = template_img.shape[:2]
            aligned_image = cv2.warpPerspective(target_img, homography, (w, h))

            # 3. 提取ROI
            roi_template, roi_aligned, roi_coords = self.extract_roi(
                template_img, aligned_image, roi
            )

            # 4. 填充结果
            result.success = True
            result.message = "图像对齐成功"
            result.homography = homography
            result.similarity_score = similarity_score
            result.roi_template = roi_template
            result.roi_aligned = roi_aligned
            result.roi_coords = roi_coords

            if not return_roi_only:
                result.aligned_image = aligned_image

            return result

        except Exception as e:
            error_msg = f"使用预计算单应性矩阵对齐图像失败: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            result.message = error_msg
            return result

    def compute_homography(
        self,
        kp_src: list,
        kp_dst: list,
        matches: list,
        min_matches: int = 4,
        max_iteration: int = 2000,
        ransac_threshold: float = 5.0,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        根据匹配点计算单应性矩阵

        参数:
            kp_src: 源图像关键点
            kp_dst: 目标图像关键点
            matches: 匹配点列表
            min_matches: 最小匹配点数量
            max_iteration: RANSAC最大迭代次数
            ransac_threshold: RANSAC重投影阈值 - 这个参数决定了内点的判定标准

        返回:
            Tuple[np.ndarray, np.ndarray]: 单应性矩阵和内点掩码
        """
        if len(matches) < min_matches:
            logger.warning(f"匹配点数量 {len(matches)} 少于所需的 {min_matches}")
            return None, None

        # 提取匹配点坐标，兼容DMatch对象和元组
        try:
            if hasattr(matches[0], "queryIdx") and hasattr(matches[0], "trainIdx"):
                # 标准DMatch对象
                src_pts = np.float32([kp_src[m.queryIdx].pt for m in matches]).reshape(
                    -1, 1, 2
                )
                dst_pts = np.float32([kp_dst[m.trainIdx].pt for m in matches]).reshape(
                    -1, 1, 2
                )
            elif isinstance(matches[0], tuple) and len(matches[0]) >= 2:
                # 元组格式 (可能是 (trainIdx, queryIdx, distance) 或其他格式)
                src_pts = np.float32([kp_src[m[0]].pt for m in matches]).reshape(
                    -1, 1, 2
                )
                dst_pts = np.float32([kp_dst[m[1]].pt for m in matches]).reshape(
                    -1, 1, 2
                )
            else:
                logger.error(
                    f"不支持的匹配格式: {type(matches[0])}, 匹配对象属性: {dir(matches[0])}"
                )
                return None, None
        except (IndexError, AttributeError) as e:
            logger.error(f"提取匹配点坐标时出错: {str(e)}")
            logger.error(f"匹配格式: {type(matches[0])}, 第一个匹配: {matches[0]}")
            return None, None

        # 使用RANSAC算法计算单应性矩阵
        # 这里的cv2.RANSAC是关键 - 它使用RANSAC算法来识别内点
        # ransacReprojThreshold参数定义了内点的判定阈值
        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,  # 这个阈值很关键
            maxIters=max_iteration,
            confidence=0.995,
        )

        return H, mask  # mask是一个二进制掩码，1表示内点，0表示外点

    def extract_roi(
        self,
        template_img: np.ndarray,
        aligned_img: np.ndarray,
        roi: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """
        从模板和对齐图像中提取ROI区域

        参数:
            template_img: 模板图像
            aligned_img: 已对齐的图像
            roi: ROI参数，格式为 {'x': x, 'y': y, 'width': w, 'height': h} 或 {'points': [(x1,y1), (x2,y2), ...]}
                 如果未提供，则使用整个图像

        返回:
            Tuple[np.ndarray, np.ndarray, Dict]: 模板ROI, 对齐图像ROI, ROI坐标信息
        """
        h, w = template_img.shape[:2]

        if roi is None:
            # 使用整个图像作为ROI
            return template_img, aligned_img, {"x": 0, "y": 0, "width": w, "height": h}

        # 处理矩形ROI
        if all(key in roi for key in ["x", "y", "width", "height"]):
            x = int(roi["x"])
            y = int(roi["y"])
            width = int(roi["width"])
            height = int(roi["height"])

            # 确保ROI在图像范围内
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            width = max(1, min(width, w - x))
            height = max(1, min(height, h - y))

            # 提取ROI
            roi_template = template_img[y : y + height, x : x + width]
            roi_aligned = aligned_img[y : y + height, x : x + width]

            roi_coords = {"x": x, "y": y, "width": width, "height": height}

            return roi_template, roi_aligned, roi_coords

        # 处理多边形ROI
        elif (
            "points" in roi
            and isinstance(roi["points"], list)
            and len(roi["points"]) > 0
        ):
            # 创建掩码
            mask = np.zeros(template_img.shape[:2], dtype=np.uint8)
            points = np.array([roi["points"]], dtype=np.int32)
            cv2.fillPoly(mask, points, 255)

            # 计算边界框
            x, y, width, height = cv2.boundingRect(points)

            # 提取ROI
            roi_template = cv2.bitwise_and(template_img, template_img, mask=mask)
            roi_aligned = cv2.bitwise_and(aligned_img, aligned_img, mask=mask)

            # 裁剪到边界框
            roi_template = roi_template[y : y + height, x : x + width]
            roi_aligned = roi_aligned[y : y + height, x : x + width]

            roi_coords = {"points": roi["points"]}

            return roi_template, roi_aligned, roi_coords

        # 未知ROI格式
        else:
            logger.warning(f"未知的ROI格式: {roi}")
            return template_img, aligned_img, {"x": 0, "y": 0, "width": w, "height": h}

    def compute_alignment_score(self, matches: list, mask: np.ndarray = None) -> float:
        """
        计算对齐质量得分

        参数:
            matches: 匹配点列表
            mask: 内点掩码

        返回:
            float: 0-1之间的评分，1表示完美对齐
        """
        if not matches:
            return 0.0

        # 如果有内点掩码，计算内点比例
        if mask is not None:
            inliers = np.sum(mask)
            inlier_ratio = inliers / len(matches)
            return min(inlier_ratio, 1.0)  # 确保分数不超过1
        else:
            # 根据匹配点的平均距离计算分数，兼容DMatch对象和元组
            try:
                if hasattr(matches[0], "distance"):
                    # 标准DMatch对象
                    avg_distance = np.mean([m.distance for m in matches])
                elif isinstance(matches[0], tuple) and len(matches[0]) >= 3:
                    # 元组格式，假设distance是第三个元素
                    avg_distance = np.mean([m[2] for m in matches])
                else:
                    # 无法获取距离信息，返回基础分数
                    return 0.5

                # 距离越小，匹配质量越高，分数越高
                score = 1.0 / (1.0 + avg_distance / 100.0)
                return min(score, 1.0)
            except (IndexError, AttributeError):
                return 0.5  # 返回中等分数

    def visualize_alignment(
        self,
        template_img: np.ndarray,
        aligned_img: np.ndarray,
        roi: dict | None = None,
    ) -> np.ndarray:
        """
        可视化对齐结果

        参数:
            template_img: 模板图像
            aligned_img: 对齐后的图像
            roi: ROI参数

        返回:
            np.ndarray: 可视化结果图像
        """
        if template_img is None or aligned_img is None:
            return None

        # 确保图像是RGB格式，便于显示
        if len(template_img.shape) == 2:
            template_rgb = cv2.cvtColor(template_img, cv2.COLOR_GRAY2BGR)
        else:
            template_rgb = template_img.copy()

        if len(aligned_img.shape) == 2:
            aligned_rgb = cv2.cvtColor(aligned_img, cv2.COLOR_GRAY2BGR)
        else:
            aligned_rgb = aligned_img.copy()

        # 创建带有两幅图像的水平拼接结果
        h1, w1 = template_rgb.shape[:2]
        h2, w2 = aligned_rgb.shape[:2]
        h = max(h1, h2)
        vis = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
        vis[:h1, :w1] = template_rgb
        vis[:h2, w1 : w1 + w2] = aligned_rgb

        # 在图像上标记ROI
        if roi is not None:
            if all(key in roi for key in ["x", "y", "width", "height"]):
                x, y, w, h = roi["x"], roi["y"], roi["width"], roi["height"]
                # 在模板上绘制ROI
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # 在对齐图像上也绘制相同位置的ROI
                cv2.rectangle(vis, (w1 + x, y), (w1 + x + w, y + h), (0, 255, 0), 2)

            elif (
                "points" in roi
                and isinstance(roi["points"], list)
                and len(roi["points"]) > 0
            ):
                points = np.array(roi["points"], dtype=np.int32)
                # 在模板上绘制多边形ROI
                cv2.polylines(vis, [points], True, (0, 255, 0), 2)
                # 在对齐图像上也绘制相同的多边形
                points_aligned = points.copy()
                points_aligned[:, 0] += w1  # 水平偏移到第二幅图像
                cv2.polylines(vis, [points_aligned], True, (0, 255, 0), 2)

        # 添加标签
        cv2.putText(
            vis, "Template", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
        cv2.putText(
            vis, "Aligned", (w1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )

        return vis

    def calculate_difference_map(
        self,
        template_roi: np.ndarray,
        aligned_roi: np.ndarray,
        method: str = "absolute",
    ) -> np.ndarray:
        """
        计算两个ROI之间的差异图

        参数:
            template_roi: 模板ROI
            aligned_roi: 对齐后的ROI
            method: 差异计算方法，可选值为'absolute'（绝对差异）、'squared'（平方差异）或'structural'（结构相似性）

        返回:
            np.ndarray: 差异图（彩色映射增强）
        """
        if template_roi is None or aligned_roi is None:
            return None

        # 确保两个ROI具有相同的尺寸
        if template_roi.shape != aligned_roi.shape:
            logger.warning(
                f"ROI尺寸不匹配: 模板={template_roi.shape}, 对齐={aligned_roi.shape}"
            )
            # 调整大小使两者相同
            aligned_roi = cv2.resize(
                aligned_roi, (template_roi.shape[1], template_roi.shape[0])
            )

        # 转换为浮点类型
        template_float = template_roi.astype(np.float32) / 255.0
        aligned_float = aligned_roi.astype(np.float32) / 255.0

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
            sigma_xy = (
                cv2.GaussianBlur(template_gray * aligned_gray, (11, 11), 1.5) - mu_xy
            )

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

    def align_with_homography(
        self,
        target_img: np.ndarray,
        homography: np.ndarray,
        template_shape: tuple[int, int],
    ) -> np.ndarray:
        """
        使用预先计算的单应性矩阵对齐图像

        参数:
            target_img: 待对齐的目标图像
            homography: 单应性矩阵
            template_shape: 模板图像的形状 (height, width)

        返回:
            np.ndarray: 对齐后的图像
        """
        if target_img is None or homography is None:
            return None

        h, w = template_shape
        aligned_img = cv2.warpPerspective(target_img, homography, (w, h))
        return aligned_img


# 示例用法
# if __name__ == "__main__":
#     import os
#     import cv2
#     import glob
#     from pathlib import Path

#     # 导入特征检测和匹配模块
#     from features.detector import FeatureDetector, DetectorFactory
#     from features.matcher import FeatureMatcher, MatcherFactory

#     # 创建结果目录
#     output_dir = "alignment_results"
#     os.makedirs(output_dir, exist_ok=True)

#     # 加载测试图像
#     template_path = "results/template_20250513_113409/temp/kapibala.png"
#     target_path = "results/template_20250513_113409/temp/img008.jpg"
#     # 检查路径是否存在
#     if not os.path.exists(template_path):
#         print(f"模板图像不存在: {template_path}")
#         # 尝试查找其他可能的模板图像
#         possible_templates = glob.glob("results/template_*/temp/*.png")
#         if possible_templates:
#             template_path = possible_templates[0]
#             print(f"使用替代模板: {template_path}")
#         else:
#             print("找不到任何可用的模板图像")
#             exit(1)

#     if not os.path.exists(target_path):
#         print(f"目标图像不存在: {target_path}")
#         # 尝试查找其他可能的目标图像
#         possible_targets = glob.glob("results/template_*/temp/*.tif") or glob.glob("results/template_*/temp/*.jpg")
#         if possible_targets:
#             target_path = possible_targets[0]
#             print(f"使用替代目标图像: {target_path}")
#         else:
#             print("找不到任何可用的目标图像")
#             exit(1)

#     # 读取图像
#     template_img = cv2.imread(template_path)
#     target_img = cv2.imread(target_path)

#     if template_img is None:
#         print(f"无法读取模板图像: {template_path}")
#         exit(1)

#     if target_img is None:
#         print(f"无法读取目标图像: {target_path}")
#         exit(1)

#     # 预处理图像以提高匹配成功率
#     def preprocess_image(img):
#         # 确保图像尺寸合理 (太大的图像可能匹配慢且不准确)
#         max_dim = 1000
#         h, w = img.shape[:2]
#         if max(h, w) > max_dim:
#             scale = max_dim / max(h, w)
#             new_size = (int(w * scale), int(h * scale))
#             img = cv2.resize(img, new_size)

#         # 转为灰度图
#         if len(img.shape) == 3:
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = img

#         # 对比度增强
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         enhanced = clahe.apply(gray)

#         return enhanced, img  # 返回增强后的灰度图和原始(可能调整大小后)的彩色图

#     # 预处理图像
#     template_gray, template_color = preprocess_image(template_img)
#     target_gray, target_color = preprocess_image(target_img)

#     # 定义要测试的特征检测器
#     detector_types = [ 'surf', 'sift']

#     # 存储每种检测器的结果
#     method1_results = {}
#     method2_results = {}

#     # 循环测试每种检测器
#     for det_type in detector_types:
#         print(f"\n\n===== 测试 {det_type.upper()} 特征检测器 =====")

#         try:
#             # 创建特征检测器和匹配器
#             detector = DetectorFactory.create(det_type, nfeatures=2000)
#             matcher = MatcherFactory.create_for_detector(det_type, crossCheck=False)

#             # 创建图像对齐器
#             aligner = ImageAlignment(detector=detector, matcher=matcher)

#             print(f"\n--- 方法1: 使用传统方式 ({det_type}) ---")
#             # 方法1: 直接使用ImageAlignment类的方法进行对齐
#             result1 = aligner.align_images(
#                 template_color,
#                 target_color,
#                 ratio_test=True,
#                 ratio_threshold=0.7
#             )

#             if result1.success:
#                 print(f"方法1成功: 相似度得分 = {result1.similarity_score:.4f}")
#                 # 保存结果
#                 cv2.imwrite(os.path.join(output_dir, f"method1_{det_type}_aligned.jpg"), result1.aligned_image)
#                 diff1 = aligner.calculate_difference_map(result1.roi_template, result1.roi_aligned)
#                 cv2.imwrite(os.path.join(output_dir, f"method1_{det_type}_diff.jpg"), diff1)

#                 # 记录结果
#                 method1_results[det_type] = {
#                     'score': result1.similarity_score,
#                     'result': result1
#                 }
#             else:
#                 print(f"方法1失败: {result1.message}")
#                 method1_results[det_type] = None

#             print(f"\n--- 方法2: 使用分离方式 ({det_type}) ---")
#             # 方法2: 使用matcher计算H和mask，然后单独提供给alignment

#             # 1. 检测特征点
#             kp_template, desc_template = detector.detect(template_color)
#             kp_target, desc_target = detector.detect(target_color)

#             # 2. 匹配特征点
#             raw_matches = matcher.match(desc_target, desc_template, k=2)
#             good_matches = matcher.filter_matches(raw_matches, ratio=0.7)

#             # 3. 使用matcher计算单应性矩阵
#             H, mask = matcher.compute_homography(
#                 kp_target, kp_template, good_matches,
#                 min_matches=4, ransac_thresh=10.0
#             )

#             if H is not None:
#                 # 4. 使用预计算的H和mask对齐图像
#                 result2 = aligner.align_with_computed_homography(
#                     template_color,
#                     target_color,
#                     homography=H,
#                     mask=mask,
#                     matches=good_matches
#                     #roi={'x': 100, 'y': 100, 'width': 300, 'height': 300}
#                 )

#                 if result2.success:
#                     print(f"方法2成功: 相似度得分 = {result2.similarity_score:.4f}")
#                     # 保存结果
#                     cv2.imwrite(os.path.join(output_dir, f"method2_{det_type}_aligned.jpg"), result2.aligned_image)
#                     diff2 = aligner.calculate_difference_map(result2.roi_template, result2.roi_aligned)
#                     cv2.imwrite(os.path.join(output_dir, f"method2_{det_type}_diff.jpg"), diff2)

#                     # 可视化结果
#                     vis = aligner.visualize_alignment(template_color, result2.aligned_image, roi=result2.roi_coords)
#                     cv2.imwrite(os.path.join(output_dir, f"method2_{det_type}_vis.jpg"), vis)

#                     # 记录结果
#                     method2_results[det_type] = {
#                         'score': result2.similarity_score,
#                         'result': result2
#                     }

#                     print(f"已保存方法2的对齐结果和差异图到 {output_dir} 目录")
#                 else:
#                     print(f"方法2失败: {result2.message}")
#                     method2_results[det_type] = None
#             else:
#                 print(f"方法2失败: 无法计算单应性矩阵")
#                 method2_results[det_type] = None

#         except Exception as e:
#             print(f"使用 {det_type} 时出错: {str(e)}")
#             method1_results[det_type] = None
#             method2_results[det_type] = None

#     # 输出所有结果的比较
#     print("\n\n====== 结果比较 ======")

#     # 比较方法1的结果
#     print("\n方法1 (传统方式) 结果比较:")
#     best_method1_score = -1
#     best_method1_detector = None

#     for det_type, result_info in method1_results.items():
#         if result_info is not None:
#             score = result_info['score']
#             print(f"  {det_type.upper()}: 得分 = {score:.4f}")

#             if score > best_method1_score:
#                 best_method1_score = score
#                 best_method1_detector = det_type
#         else:
#             print(f"  {det_type.upper()}: 失败")

#     if best_method1_detector:
#         print(f"\n方法1最佳检测器: {best_method1_detector.upper()}, 得分 = {best_method1_score:.4f}")
#     else:
#         print("\n方法1所有检测器均失败")

#     # 比较方法2的结果
#     print("\n方法2 (分离方式) 结果比较:")
#     best_method2_score = -1
#     best_method2_detector = None

#     for det_type, result_info in method2_results.items():
#         if result_info is not None:
#             score = result_info['score']
#             print(f"  {det_type.upper()}: 得分 = {score:.4f}")

#             if score > best_method2_score:
#                 best_method2_score = score
#                 best_method2_detector = det_type
#         else:
#             print(f"  {det_type.upper()}: 失败")

#     if best_method2_detector:
#         print(f"\n方法2最佳检测器: {best_method2_detector.upper()}, 得分 = {best_method2_score:.4f}")
#     else:
#         print("\n方法2所有检测器均失败")

#     # 综合比较
#     print("\n综合最佳结果:")
#     if best_method1_score > best_method2_score:
#         print(f"方法1 + {best_method1_detector.upper()} (得分 = {best_method1_score:.4f})")
#         best_result = method1_results[best_method1_detector]['result']
#         best_method = f"method1_{best_method1_detector}"
#     elif best_method2_score > -1:
#         print(f"方法2 + {best_method2_detector.upper()} (得分 = {best_method2_score:.4f})")
#         best_result = method2_results[best_method2_detector]['result']
#         best_method = f"method2_{best_method2_detector}"
#     else:
#         print("所有方法均失败")
#         exit(0)

#     # 保存综合最佳结果
#     template_filename = Path(template_path).stem
#     target_filename = Path(target_path).stem

#     best_vis = aligner.visualize_alignment(template_color, best_result.aligned_image)
#     cv2.imwrite(os.path.join(output_dir, f"best_alignment_{template_filename}_{target_filename}.jpg"), best_vis)

#     best_diff = aligner.calculate_difference_map(best_result.roi_template, best_result.roi_aligned)
#     cv2.imwrite(os.path.join(output_dir, f"best_diff_{template_filename}_{target_filename}.jpg"), best_diff)

#     print(f"\n已保存最佳对齐结果和差异图到 {output_dir} 目录")
#     print(f"最佳方法: {best_method}")
