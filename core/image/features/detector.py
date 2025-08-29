# filepath: c:\Users\gdut\Desktop\C_D\new_version\color_difference\core\image\features\detector.py
from abc import ABC, abstractmethod
from typing import Any

import cv2
import numpy as np


class FeatureDetector(ABC):
    """
    特征点检测器抽象基类
    定义了特征检测器的通用接口
    """

    def __init__(self, **kwargs):
        """
        初始化特征检测器

        参数:
            **kwargs: 特征检测器的配置参数
        """
        self.params = kwargs
        self._detector = None
        self._init_detector()

    @abstractmethod
    def _init_detector(self):
        """
        初始化具体的特征检测器实现
        由子类实现
        """

    def detect(self, image: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray]:
        """
        检测图像中的特征点

        参数:
            image: 输入图像 (np.ndarray)

        返回:
            Tuple[List[cv2.KeyPoint], np.ndarray]: 关键点列表和特征描述符
        """
        if image is None:
            return [], None

        # 转换为灰度图像，如果是彩色图像
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 检测关键点和计算描述符
        keypoints, descriptors = self._detector.detectAndCompute(gray, None)

        return keypoints, descriptors

    def detect_keypoints(self, image: np.ndarray) -> list[cv2.KeyPoint]:
        """
        仅检测特征点，不计算描述符

        参数:
            image: 输入图像

        返回:
            List[cv2.KeyPoint]: 检测到的关键点列表
        """
        if image is None:
            return []

        # 转换为灰度图像，如果是彩色图像
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 仅检测关键点
        keypoints = self._detector.detect(gray, None)

        return keypoints

    def compute_descriptors(
        self, image: np.ndarray, keypoints: list[cv2.KeyPoint]
    ) -> np.ndarray:
        """
        计算给定关键点的描述符

        参数:
            image: 输入图像
            keypoints: 关键点列表

        返回:
            np.ndarray: 特征描述符
        """
        if image is None or not keypoints:
            return None

        # 转换为灰度图像，如果是彩色图像
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 计算描述符
        _, descriptors = self._detector.compute(gray, keypoints)

        return descriptors

    def get_params(self) -> dict[str, Any]:
        """
        获取检测器参数

        返回:
            Dict[str, Any]: 参数字典
        """
        return self.params

    def get_name(self) -> str:
        """
        获取检测器名称

        返回:
            str: 检测器名称
        """
        return self.__class__.__name__

    def draw_keypoints(
        self, image: np.ndarray, keypoints: list[cv2.KeyPoint]
    ) -> np.ndarray:
        """
        在图像上绘制特征点

        参数:
            image: 输入图像
            keypoints: 关键点列表

        返回:
            np.ndarray: 带有特征点标记的图像
        """
        return cv2.drawKeypoints(
            image,
            keypoints,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )


class SiftDetector(FeatureDetector):
    """
    SIFT特征检测器实现
    尺度不变特征变换（Scale-Invariant Feature Transform）
    """

    def _init_detector(self):
        """
        初始化SIFT检测器
        """
        # 提取参数，如果未提供则使用默认值
        n_features = self.params.get("nfeatures", 0)
        n_octave_layers = self.params.get("nOctaveLayers", 3)
        contrast_threshold = self.params.get("contrastThreshold", 0.04)
        edge_threshold = self.params.get("edgeThreshold", 10)
        sigma = self.params.get("sigma", 1.6)

        # 创建SIFT检测器 - 使用现代OpenCV API (OpenCV 4.x+)
        try:
            # 使用新版本OpenCV的SIFT (cv2.SIFT_create)
            self._detector = cv2.SIFT_create(
                nfeatures=n_features,
                nOctaveLayers=n_octave_layers,
                contrastThreshold=contrast_threshold,
                edgeThreshold=edge_threshold,
                sigma=sigma,
            )
        except AttributeError:
            print("错误：SIFT在此版本的OpenCV中不可用")
            raise RuntimeError("SIFT detector not available in this OpenCV version")


class SurfDetector(FeatureDetector):
    """
    SURF特征检测器实现
    加速稳健特征（Speeded-Up Robust Features）
    注意：较新版本的OpenCV可能不包含SURF，因为它是专利算法
    """

    def _init_detector(self):
        """
        初始化SURF检测器
        """
        # 如果OpenCV没有编译SURF支持，提供备用方案
        try:
            # 提取参数，如果未提供则使用默认值
            hessian_threshold = self.params.get("hessianThreshold", 100)
            n_octaves = self.params.get("nOctaves", 4)
            n_octave_layers = self.params.get("nOctaveLayers", 3)
            extended = self.params.get("extended", False)
            upright = self.params.get("upright", False)

            # 创建SURF检测器
            self._detector = cv2.xfeatures2d.SURF_create(
                hessianThreshold=hessian_threshold,
                nOctaves=n_octaves,
                nOctaveLayers=n_octave_layers,
                extended=extended,
                upright=upright,
            )
        except (AttributeError, cv2.error):
            print("警告：SURF在此版本的OpenCV中不可用，使用SIFT替代")
            # 回退到SIFT - 使用现代OpenCV API
            try:
                self._detector = cv2.SIFT_create()
            except AttributeError:
                raise RuntimeError(
                    "Neither SURF nor SIFT are available in this OpenCV version"
                )


class OrbDetector(FeatureDetector):
    """
    ORB特征检测器实现
    定向快速旋转简要特征（Oriented FAST and Rotated BRIEF）
    """

    def _init_detector(self):
        """
        初始化ORB检测器
        """
        # 提取参数，如果未提供则使用默认值
        n_features = self.params.get("nfeatures", 500)
        scale_factor = self.params.get("scaleFactor", 1.2)
        n_levels = self.params.get("nlevels", 8)
        edge_threshold = self.params.get("edgeThreshold", 31)
        first_level = self.params.get("firstLevel", 0)
        wta_k = self.params.get("WTA_K", 2)
        score_type = self.params.get("scoreType", cv2.ORB_HARRIS_SCORE)
        patch_size = self.params.get("patchSize", 31)
        fast_threshold = self.params.get("fastThreshold", 20)

        # 创建ORB检测器
        self._detector = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=edge_threshold,
            firstLevel=first_level,
            WTA_K=wta_k,
            scoreType=score_type,
            patchSize=patch_size,
            fastThreshold=fast_threshold,
        )


class DetectorFactory:
    """
    特征检测器工厂类
    用于创建不同类型的特征检测器
    """

    @staticmethod
    def create(detector_type: str, **kwargs) -> FeatureDetector | None:
        """
        创建指定类型的特征检测器

        参数:
            detector_type: 检测器类型 ('sift', 'surf', 'orb')
            **kwargs: 传递给检测器的参数

        返回:
            FeatureDetector: 创建的特征检测器，如果类型无效则返回None
        """
        detector_type = detector_type.lower()

        if detector_type == "sift":
            return SiftDetector(**kwargs)
        elif detector_type == "surf":
            return SurfDetector(**kwargs)
        elif detector_type == "orb":
            return OrbDetector(**kwargs)
        else:
            print(f"错误：不支持的检测器类型 '{detector_type}'")
            return None

    @staticmethod
    def list_available_detectors() -> list[str]:
        """
        列出所有可用的检测器类型

        返回:
            List[str]: 可用检测器类型列表
        """
        detectors = ["sift", "orb"]

        # 检查SURF是否可用
        try:
            cv2.xfeatures2d.SURF_create()
            detectors.append("surf")
        except (AttributeError, cv2.error):
            pass

        return detectors


# 示例用法
if __name__ == "__main__":
    import cv2

    # 加载测试图像
    image_path = "results/template_20250513_113409/temp/butterfly.png"
    image = cv2.imread(image_path)

    if image is None:
        print(f"无法加载图像: {image_path}")
    else:
        # 使用工厂创建检测器
        detector = DetectorFactory.create("surf", nfeatures=100)

        if detector:
            # 检测特征点
            keypoints, descriptors = detector.detect(image)

            # 绘制特征点
            result = detector.draw_keypoints(image, keypoints)

            # 显示结果
            print(f"检测到 {len(keypoints)} 个关键点")
            cv2.imwrite(
                "results/template_20250513_113409/temp/surf_keypoints.png", result
            )
            print("已保存带有关键点的图像")
