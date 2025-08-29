# filepath: c:\Users\gdut\Desktop\C_D\new_version\color_difference\core\image\features\matcher.py
from abc import ABC, abstractmethod
from typing import Any

import cv2
import numpy as np


class FeatureMatcher(ABC):
    """
    特征匹配器抽象基类
    定义了特征匹配器的通用接口
    """

    def __init__(self, **kwargs):
        """
        初始化特征匹配器

        参数:
            **kwargs: 特征匹配器的配置参数
        """
        self.params = kwargs
        self._matcher = None
        self._init_matcher()

    @abstractmethod
    def _init_matcher(self):
        """
        初始化具体的特征匹配器实现
        由子类实现
        """

    def match(
        self, descriptors1: np.ndarray, descriptors2: np.ndarray, k: int = 2
    ) -> list:
        """
        匹配两组特征描述符

        参数:
            descriptors1: 第一组特征描述符
            descriptors2: 第二组特征描述符
            k: 每个特征点返回的最佳匹配数量

        返回:
            List: 匹配结果列表
        """
        if descriptors1 is None or descriptors2 is None:
            return []

        if len(descriptors1) == 0 or len(descriptors2) == 0:
            return []

        # 检查描述符类型并转换为适当的格式
        # 对于二进制描述符（如ORB），应该使用uint8类型，而对于浮点描述符（如SIFT、SURF），则使用float32
        descriptor_type = self.params.get("descriptor_type", "float")
        # print(f"descriptor_type: {descriptor_type}")
        if descriptor_type.lower() == "binary":
            # 二进制描述符（ORB、BRIEF等）使用uint8
            if descriptors1.dtype != np.uint8:
                descriptors1 = descriptors1.astype(np.uint8)
            if descriptors2.dtype != np.uint8:
                descriptors2 = descriptors2.astype(np.uint8)
        else:
            # 浮点描述符（SIFT、SURF等）使用float32
            if descriptors1.dtype != np.float32:
                descriptors1 = descriptors1.astype(np.float32)
            if descriptors2.dtype != np.float32:
                descriptors2 = descriptors2.astype(np.float32)

        # 匹配描述符
        if k == 1:
            matches = self._matcher.match(descriptors1, descriptors2)
            return matches
        else:
            matches = self._matcher.knnMatch(descriptors1, descriptors2, k=k)
            return matches

    def filter_matches(self, matches: list, ratio: float = 0.75) -> list:
        """
        使用Lowe比率测试过滤匹配点

        参数:
            matches: knnMatch返回的匹配结果列表
            ratio: Lowe比率测试阈值

        返回:
            List: 过滤后的匹配点列表
        """
        # 检查匹配结果是否为KNN匹配(k=2)
        # OpenCV可能返回tuple或list，都应该支持
        if (
            not matches
            or (not isinstance(matches[0], list | tuple))
            or len(matches[0]) != 2
        ):
            return matches

        # 应用Lowe比率测试
        good_matches = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good_matches.append(m)

        return good_matches

    def draw_matches(
        self,
        image1: np.ndarray,
        keypoints1: list[cv2.KeyPoint],
        image2: np.ndarray,
        keypoints2: list[cv2.KeyPoint],
        matches: list,
        mask: list[int] | None = None,
    ) -> np.ndarray:
        """
        绘制两幅图像之间的匹配点

        参数:
            image1: 第一幅图像
            keypoints1: 第一幅图像的关键点
            image2: 第二幅图像
            keypoints2: 第二幅图像的关键点
            matches: 匹配点列表
            mask: 可选的匹配掩码列表

        返回:
            np.ndarray: 包含匹配可视化的图像
        """
        # 创建匹配可视化图像
        matches_img = cv2.drawMatches(
            image1,
            keypoints1,
            image2,
            keypoints2,
            matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchesMask=mask,
        )

        return matches_img

    def compute_homography(
        self,
        keypoints1: list[cv2.KeyPoint],
        keypoints2: list[cv2.KeyPoint],
        matches: list,
        min_matches: int = 4,
        ransac_thresh: float = 5.0,
    ) -> tuple[np.ndarray, list[int]]:
        """
        根据匹配点计算单应性矩阵

        参数:
            keypoints1: 第一幅图像的关键点
            keypoints2: 第二幅图像的关键点
            matches: 匹配点列表
            min_matches: 最小所需匹配点数量
            ransac_thresh: RANSAC算法的阈值

        返回:
            Tuple[np.ndarray, List[int]]: 单应性矩阵和内点掩码
        """
        if len(matches) < min_matches:
            return None, []

        # 提取匹配点的坐标
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(
            -1, 1, 2
        )

        # 使用RANSAC算法计算单应性矩阵
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        mask = mask.ravel().tolist()

        return H, mask

    def get_params(self) -> dict[str, Any]:
        """
        获取匹配器参数

        返回:
            Dict[str, Any]: 参数字典
        """
        return self.params

    def get_name(self) -> str:
        """
        获取匹配器名称

        返回:
            str: 匹配器名称
        """
        return self.__class__.__name__


class FlannMatcher(FeatureMatcher):
    """
    FLANN特征匹配器实现
    快速最近邻搜索库（Fast Library for Approximate Nearest Neighbors）
    """

    def _init_matcher(self):
        """
        初始化FLANN匹配器
        """
        # 从参数中提取需要的配置，如果未提供则使用默认值
        algorithm = self.params.get("algorithm", 1)  # FLANN_INDEX_KDTREE = 1
        trees = self.params.get("trees", 5)
        checks = self.params.get("checks", 50)

        # 根据不同类型的描述符设置不同的索引参数
        # 当使用ORB等二进制描述符时，需要使用不同的算法
        descriptor_type = self.params.get("descriptor_type", "float")
        self.params.get("crossCheck", False)

        if descriptor_type.lower() == "binary":
            # 适用于ORB、BRIEF等二进制描述符
            index_params = {
                "algorithm": 6,  # FLANN_INDEX_LSH = 6
                "table_number": 6,
                "key_size": 12,
                "multi_probe_level": 1,
            }
        else:
            # 适用于SIFT、SURF等浮点描述符
            index_params = {"algorithm": algorithm, "trees": trees}

        search_params = {"checks": checks}

        # 创建FLANN匹配器
        self._matcher = cv2.FlannBasedMatcher(index_params, search_params)


class BFMatcher(FeatureMatcher):
    """
    暴力特征匹配器实现
    Brute-Force Matcher
    """

    def _init_matcher(self):
        """
        初始化暴力匹配器
        """
        # 从参数中提取需要的配置，如果未提供则使用默认值
        norm_type = self.params.get("normType", cv2.NORM_L2)
        cross_check = self.params.get("crossCheck", False)

        # 根据描述符类型选择适当的距离度量方式
        descriptor_type = self.params.get("descriptor_type", "float")
        if descriptor_type.lower() == "binary":
            # 适用于ORB、BRIEF等二进制描述符
            norm_type = cv2.NORM_HAMMING

        # 创建暴力匹配器
        self._matcher = cv2.BFMatcher(norm_type, cross_check)


class MatcherFactory:
    """
    特征匹配器工厂类
    用于创建不同类型的特征匹配器
    """

    @staticmethod
    def create(matcher_type: str, **kwargs) -> FeatureMatcher | None:
        """
        创建指定类型的特征匹配器

        参数:
            matcher_type: 匹配器类型 ('flann', 'bf')
            **kwargs: 传递给匹配器的参数

        返回:
            FeatureMatcher: 创建的特征匹配器，如果类型无效则返回None
        """
        matcher_type = matcher_type.lower()

        if matcher_type == "flann":
            return FlannMatcher(**kwargs)
        elif matcher_type in ["bf", "bruteforce"]:
            return BFMatcher(**kwargs)
        else:
            print(f"错误：不支持的匹配器类型 '{matcher_type}'")
            return None

    @staticmethod
    def create_for_detector(detector_type: str, **kwargs) -> FeatureMatcher:
        """
        根据检测器类型创建适合的匹配器

        参数:
            detector_type: 检测器类型 ('sift', 'surf', 'orb')
            **kwargs: 传递给匹配器的其他参数

        返回:
            FeatureMatcher: 创建的特征匹配器
        """
        detector_type = detector_type.lower()

        if detector_type in ["orb", "brief", "brisk", "freak"]:
            # 二进制描述符，使用BF匹配器 + 汉明距离
            return BFMatcher(descriptor_type="binary", **kwargs)
        else:
            # 浮点描述符 (SIFT, SURF)，使用FLANN匹配器 + L2距离
            return FlannMatcher(descriptor_type="float", **kwargs)

    @staticmethod
    def list_available_matchers() -> list[str]:
        """
        列出所有可用的匹配器类型

        返回:
            List[str]: 可用匹配器类型列表
        """
        return ["flann", "bf"]


# # 示例用法
# if __name__ == "__main__":
#     import cv2
#     from detector import DetectorFactory
#     import os

#     # 加载测试图像
#     image1_path = "results/template_20250513_113409/temp/butterfly.png"
#     image2_path = "results/template_20250513_113409/temp/processed_20250513_113419.png"

#     if not os.path.exists(image1_path) or not os.path.exists(image2_path):
#         print(f"无法找到测试图像")
#     else:
#         image1 = cv2.imread(image1_path)
#         image2 = cv2.imread(image2_path)

#         # 创建特征检测器
#         detector = DetectorFactory.create('sift')

#         if detector:
#             # 检测特征点
#             keypoints1, descriptors1 = detector.detect(image1)
#             keypoints2, descriptors2 = detector.detect(image2)

#             print(f"图像1检测到 {len(keypoints1)} 个关键点")
#             print(f"图像2检测到 {len(keypoints2)} 个关键点")

#             # 创建特征匹配器
#             matcher = MatcherFactory.create_for_detector('sift')

#             # 匹配特征点
#             matches = matcher.match(descriptors1, descriptors2, k=2)

#             # 过滤匹配点
#             good_matches = matcher.filter_matches(matches, ratio=0.75)

#             print(f"找到 {len(good_matches)} 个良好匹配点")

#             # 绘制匹配结果
#             result = matcher.draw_matches(image1, keypoints1, image2, keypoints2, good_matches, None)

#             # 计算单应性变换
#             H, mask = matcher.compute_homography(keypoints1, keypoints2, good_matches)

#             # 保存结果
#             cv2.imwrite("results/template_20250513_113409/temp/matches_original_20250513_113418.png", result)
#             print("已保存匹配结果图像")
