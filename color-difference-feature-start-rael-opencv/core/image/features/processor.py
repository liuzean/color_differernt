import logging
import traceback
from dataclasses import dataclass

import numpy as np

# 导入自定义的特征检测和匹配模块
from detector import DetectorFactory, FeatureDetector
from matcher import MatcherFactory

# 配置日志记录
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FeatureProcessor")


@dataclass
class FeatureProcessResult:
    """特征处理结果数据类"""

    success: bool
    message: str = ""
    keypoints1: list = None
    keypoints2: list = None
    descriptors1: np.ndarray = None
    descriptors2: np.ndarray = None
    matches: list = None
    homography: np.ndarray = None
    inliers_mask: list = None
    similarity_score: float = 0.0
    result_image: np.ndarray = None


class FeatureProcessor:
    """
    特征处理协调器
    - 集成特征检测和匹配
    - 提供高级API
    - 错误处理与容错机制
    """

    def __init__(
        self,
        detector_type: str = "sift",
        matcher_type: str = None,
        detector_params: dict = None,
        matcher_params: dict = None,
    ):
        """
        初始化特征处理器

        参数:
            detector_type: 特征检测器类型 ('sift', 'surf', 'orb')
            matcher_type: 特征匹配器类型 (None: 自动选择, 'flann', 'bf')
            detector_params: 传递给检测器的参数
            matcher_params: 传递给匹配器的参数
        """
        try:
            # 初始化检测器
            self.detector_type = detector_type
            self.detector_params = detector_params or {}
            self.detector = self._create_detector(detector_type, self.detector_params)

            # 初始化匹配器 (如果未指定，根据检测器类型自动选择)
            if matcher_type is None:
                self.matcher = MatcherFactory.create_for_detector(
                    detector_type, **(matcher_params or {})
                )
                self.matcher_type = self.matcher.get_name()
            else:
                self.matcher_type = matcher_type
                self.matcher_params = matcher_params or {}
                self.matcher = MatcherFactory.create(
                    matcher_type, **(self.matcher_params)
                )

            logger.info(
                f"特征处理器初始化成功: 检测器={detector_type}, 匹配器={self.matcher_type}"
            )

        except Exception as e:
            error_msg = f"特征处理器初始化失败: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            raise RuntimeError(error_msg)

    def _create_detector(self, detector_type: str, params: dict) -> FeatureDetector:
        """
        创建特征检测器

        参数:
            detector_type: 检测器类型
            params: 检测器参数

        返回:
            FeatureDetector: 创建的特征检测器
        """
        detector = DetectorFactory.create(detector_type, **params)
        if detector is None:
            available_detectors = DetectorFactory.list_available_detectors()
            error_msg = (
                f"无效的检测器类型: {detector_type}, 可用类型: {available_detectors}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        return detector

    def change_detector(self, detector_type: str, params: dict = None) -> bool:
        """
        更改特征检测器

        参数:
            detector_type: 新的检测器类型
            params: 新的检测器参数

        返回:
            bool: 操作是否成功
        """
        try:
            self.detector_type = detector_type
            self.detector_params = params or {}
            self.detector = self._create_detector(detector_type, self.detector_params)

            # 自动更新匹配器以适配新检测器
            self.matcher = MatcherFactory.create_for_detector(
                detector_type, **self.matcher_params
            )
            self.matcher_type = self.matcher.get_name()

            logger.info(f"检测器已更改为: {detector_type}")
            return True
        except Exception as e:
            error_msg = f"更改检测器失败: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            return False

    def change_matcher(self, matcher_type: str, params: dict = None) -> bool:
        """
        更改特征匹配器

        参数:
            matcher_type: 新的匹配器类型
            params: 新的匹配器参数

        返回:
            bool: 操作是否成功
        """
        try:
            self.matcher_type = matcher_type
            self.matcher_params = params or {}
            self.matcher = MatcherFactory.create(matcher_type, **self.matcher_params)

            if self.matcher is None:
                available_matchers = MatcherFactory.list_available_matchers()
                error_msg = (
                    f"无效的匹配器类型: {matcher_type}, 可用类型: {available_matchers}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info(f"匹配器已更改为: {matcher_type}")
            return True
        except Exception as e:
            error_msg = f"更改匹配器失败: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            return False

    def process_images(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        ratio_test: bool = True,
        ratio_threshold: float = 0.75,
        compute_homography: bool = True,
        ransac_threshold: float = 5.0,
        min_matches: int = 10,
        max_matches: int = 100,
        draw_result: bool = True,
    ) -> FeatureProcessResult:
        """
        处理两张图像的高级API：检测特征点、匹配特征、计算变换矩阵

        参数:
            img1: 第一张图像
            img2: 第二张图像
            ratio_test: 是否使用比率测试过滤匹配点
            ratio_threshold: 比率测试的阈值
            compute_homography: 是否计算单应性矩阵
            ransac_threshold: RANSAC算法阈值
            min_matches: 计算相似度的最小匹配点数量
            max_matches: 可视化时显示的最大匹配点数量
            draw_result: 是否生成可视化结果图像

        返回:
            FeatureProcessResult: 处理结果对象
        """
        result = FeatureProcessResult(success=False)

        # 参数验证
        if img1 is None or img2 is None:
            result.message = "输入图像无效"
            logger.error(result.message)
            return result

        try:
            # 1. 检测第一张图像的特征点
            kp1, desc1 = self.detector.detect(img1)
            if not kp1 or len(kp1) == 0:
                result.message = "图像1中未检测到特征点"
                logger.warning(result.message)
                return result

            # 2. 检测第二张图像的特征点
            kp2, desc2 = self.detector.detect(img2)
            if not kp2 or len(kp2) == 0:
                result.message = "图像2中未检测到特征点"
                logger.warning(result.message)
                return result

            logger.info(
                f"图像1检测到 {len(kp1)} 个特征点, 图像2检测到 {len(kp2)} 个特征点"
            )

            # 3. 匹配特征点
            if ratio_test:
                # 使用KNN匹配 (k=2) 然后应用比率测试
                raw_matches = self.matcher.match(desc1, desc2, k=2)
                matches = self.matcher.filter_matches(
                    raw_matches, ratio=ratio_threshold
                )
            else:
                # 简单匹配
                matches = self.matcher.match(desc1, desc2, k=1)

            if not matches or len(matches) == 0:
                result.message = "未找到有效匹配点"
                logger.warning(result.message)
                return result

            logger.info(f"找到 {len(matches)} 个匹配点")

            # 4. 计算单应性矩阵 (如果需要)
            homography = None
            mask = None

            if compute_homography and len(matches) >= 4:
                homography, mask = self.matcher.compute_homography(
                    kp1, kp2, matches, min_matches=4, ransac_thresh=ransac_threshold
                )

                if homography is not None:
                    inliers_count = sum(mask) if mask else 0
                    logger.info(f"计算单应性矩阵成功, 找到 {inliers_count} 个内点")

            # 5. 计算相似度分数
            similarity_score = self._compute_similarity_score(kp1, kp2, matches, mask)

            # 6. 可视化结果 (如果需要)
            result_image = None
            if draw_result:
                # 限制绘制的匹配点数量
                matches_to_draw = matches[: min(len(matches), max_matches)]

                # 修复: 确保mask与matches_to_draw大小一致
                if mask is not None:
                    # 只取与matches_to_draw对应的mask
                    draw_mask = mask[: len(matches_to_draw)]
                else:
                    draw_mask = None

                result_image = self.matcher.draw_matches(
                    img1, kp1, img2, kp2, matches_to_draw, draw_mask
                )

            # 7. 填充结果对象
            result.success = True
            result.message = "特征处理成功"
            result.keypoints1 = kp1
            result.keypoints2 = kp2
            result.descriptors1 = desc1
            result.descriptors2 = desc2
            result.matches = matches
            result.homography = homography
            result.inliers_mask = mask
            result.similarity_score = similarity_score
            result.result_image = result_image

            return result

        except Exception as e:
            error_msg = f"处理图像失败: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            result.message = error_msg
            return result

    def _compute_similarity_score(
        self, kp1: list, kp2: list, matches: list, mask: list = None
    ) -> float:
        """
        计算两张图像的相似度评分

        参数:
            kp1: 第一张图像的关键点
            kp2: 第二张图像的关键点
            matches: 匹配点列表
            mask: 内点掩码 (来自RANSAC)

        返回:
            float: 相似度评分(0-1), 1表示完全匹配
        """
        if not matches or len(matches) == 0:
            return 0.0

        # 计算基本匹配比例
        min_keypoints = min(len(kp1), len(kp2))
        if min_keypoints == 0:
            return 0.0

        match_ratio = len(matches) / min_keypoints

        # 如果有内点掩码，考虑内点质量
        if mask is not None:
            inliers_count = sum(mask)
            inlier_ratio = inliers_count / len(matches) if len(matches) > 0 else 0

            # 综合考虑匹配比例和内点质量
            score = 0.4 * match_ratio + 0.6 * inlier_ratio
        else:
            score = match_ratio

        return min(score, 1.0)  # 确保评分在0-1范围内

    def detect_features(self, image: np.ndarray) -> dict:
        """
        检测单个图像的特征点

        参数:
            image: 输入图像

        返回:
            Dict: 包含关键点和描述符的字典
        """
        if image is None:
            logger.error("输入图像无效")
            return {"success": False, "message": "输入图像无效"}

        try:
            keypoints, descriptors = self.detector.detect(image)

            if not keypoints or len(keypoints) == 0:
                return {
                    "success": False,
                    "message": "未检测到特征点",
                    "keypoints": [],
                    "descriptors": None,
                }

            return {
                "success": True,
                "message": f"检测到 {len(keypoints)} 个特征点",
                "keypoints": keypoints,
                "descriptors": descriptors,
            }

        except Exception as e:
            error_msg = f"特征检测失败: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            return {"success": False, "message": error_msg}

    def visualize_features(
        self, image: np.ndarray, keypoints: list = None
    ) -> np.ndarray:
        """
        可视化图像的特征点

        参数:
            image: 输入图像
            keypoints: 已检测的关键点，若为None则重新检测

        返回:
            np.ndarray: 标记了特征点的图像
        """
        if image is None:
            logger.error("输入图像无效")
            return None

        try:
            # 如果未提供关键点，则检测
            if keypoints is None:
                keypoints = self.detector.detect_keypoints(image)

            if not keypoints or len(keypoints) == 0:
                logger.warning("未检测到特征点")
                return image

            # 绘制关键点
            return self.detector.draw_keypoints(image, keypoints)

        except Exception as e:
            error_msg = f"可视化特征点失败: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            return image

    def compare_images(self, img1: np.ndarray, img2: np.ndarray, **kwargs) -> dict:
        """
        比较两张图像的相似度并返回详细结果

        参数:
            img1: 第一张图像
            img2: 第二张图像
            **kwargs: 传递给process_images的其他参数

        返回:
            Dict: 比较结果，包含相似度评分和其他信息
        """
        result = self.process_images(img1, img2, **kwargs)

        comparison_result = {
            "success": result.success,
            "message": result.message,
            "similarity_score": result.similarity_score,
            "matches_count": len(result.matches) if result.matches else 0,
            "keypoints1_count": len(result.keypoints1) if result.keypoints1 else 0,
            "keypoints2_count": len(result.keypoints2) if result.keypoints2 else 0,
            "result_image": result.result_image,
        }

        if result.inliers_mask:
            comparison_result["inliers_count"] = sum(result.inliers_mask)

        return comparison_result

    def get_supported_detectors(self) -> list[str]:
        """
        获取支持的特征检测器列表

        返回:
            List[str]: 特征检测器名称列表
        """
        return DetectorFactory.list_available_detectors()

    def get_supported_matchers(self) -> list[str]:
        """
        获取支持的特征匹配器列表

        返回:
            List[str]: 特征匹配器名称列表
        """
        return MatcherFactory.list_available_matchers()


# # 示例用法
# if __name__ == "__main__":
#     import os

#     # 加载测试图像
#     image1_path = "results/template_20250513_113409/temp/butterfly.png"
#     image2_path = "results/template_20250513_113409/temp/processed_20250513_113419.png"

#     if os.path.exists(image1_path) and os.path.exists(image2_path):
#         img1 = cv2.imread(image1_path)
#         img2 = cv2.imread(image2_path)

#         # 创建特征处理器
#         processor = FeatureProcessor(detector_type='surf', matcher_type='bf')

#         # 处理图像
#         result = processor.process_images(img1, img2, draw_result=True)

#         if result.success:
#             print(f"检测到的特征点: 图像1={len(result.keypoints1)}, 图像2={len(result.keypoints2)}")
#             print(f"匹配点数量: {len(result.matches)}")
#             print(f"相似度评分: {result.similarity_score:.4f}")

#             # 保存结果图像
#             if result.result_image is not None:
#                 cv2.imwrite("feature_matches.jpg", result.result_image)
#                 print("已保存匹配结果图像")
#         else:
#             print(f"处理失败: {result.message}")
#     else:
#         print("测试图像不存在")
