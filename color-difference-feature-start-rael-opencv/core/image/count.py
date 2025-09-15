import cv2
import numpy as np
from features.detector import DetectorFactory
from features.matcher import MatcherFactory
from preprocessor import ImagePreprocessor
from sklearn.cluster import DBSCAN


def count_and_draw(
    scene_path,
    tpl_path,
    ratio=0.75,  # Lowe 比率测试阈值
    eps=50,  # 聚类半径（像素）
    min_samples=5,  # 每组最少匹配点
    detector_type="sift",  # 特征检测器类型
    matcher_type="flann",  # 特征匹配器类型
    maxdim=10000,
):  # 图像最大尺寸
    """
    使用特征检测、匹配和聚类统计模板在场景图像中出现的次数

    参数:
        scene_path: 场景图像路径
        tpl_path: 模板图像路径
        ratio: Lowe比率测试阈值
        eps: DBSCAN聚类半径
        min_samples: DBSCAN每组最少样本数
        detector_type: 特征检测器类型
        matcher_type: 特征匹配器类型
        maxdim: 图像最大尺寸

    返回:
        int: 检测到的模板数量
    """
    # 1. 读入图像
    img_color = cv2.imread(scene_path)
    if img_color is None:
        raise FileNotFoundError(f"无法打开场景图像：{scene_path}")

    tpl_color = cv2.imread(tpl_path)
    if tpl_color is None:
        raise FileNotFoundError(f"无法打开模板图像：{tpl_path}")

    # 2. 使用预处理器调整图像大小
    preprocessor = ImagePreprocessor()
    img_color = preprocessor.resize_image(img_color, maxdim=maxdim)
    # cv2.imshow("img_color", img_color)

    # 转灰度图
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    tpl_gray = cv2.cvtColor(tpl_color, cv2.COLOR_BGR2GRAY)

    # 3. 创建特征检测器
    detector = DetectorFactory.create(detector_type)
    if detector is None:
        raise ValueError(f"不支持的特征检测器类型: {detector_type}")

    # 4. 检测特征点和计算描述子
    kp_tpl, des_tpl = detector.detect(tpl_gray)
    kp_img, des_img = detector.detect(img_gray)

    # 如果没有检测到特征点，直接返回零
    if des_tpl is None or des_img is None or len(kp_tpl) == 0 or len(kp_img) == 0:
        print("未检测到足够的特征点。")
        return 0

    # 5. 创建特征匹配器
    descriptor_type = (
        "binary" if detector_type.lower() in ["orb", "brief", "brisk"] else "float"
    )
    matcher = MatcherFactory.create(matcher_type, descriptor_type=descriptor_type)
    if matcher is None:
        raise ValueError(f"不支持的特征匹配器类型: {matcher_type}")

    # 6. 匹配特征点
    matches = matcher.match(des_tpl, des_img, k=2)
    good_matches = matcher.filter_matches(matches, ratio=ratio)

    # 如果没有匹配点，直接返回零
    if not good_matches:
        print("没有检测到任何匹配。")
        return 0

    # 7. 收集所有匹配点在大图中的坐标
    pts = np.float32([kp_img[m.trainIdx].pt for m in good_matches])

    # 8. 用 DBSCAN 聚类，把每一簇视作一个实例
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
    labels = db.labels_
    # 标签值 -1 表示噪声
    num_instances = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"检测到 {num_instances} 个实例。")

    # 9. 可选：在原图上为每个簇画一个包围框
    for inst_id in set(labels):
        if inst_id < 0:
            continue
        cluster_pts = pts[labels == inst_id]
        x_min, y_min = cluster_pts.min(axis=0).astype(int)
        x_max, y_max = cluster_pts.max(axis=0).astype(int)
        # 根据模板大小稍微扩展一点框
        h, w = tpl_gray.shape
        cv2.rectangle(
            img_color, (x_min, y_min), (x_max + w // 2, y_max + h // 2), (0, 255, 0), 2
        )

    # 10. 显示结果
    cv2.imshow("Matches", img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return num_instances


if __name__ == "__main__":
    count_and_draw(
        "C:/Users/gdut/Desktop/C_D/new_version/color_difference/photo/img008.tif",
        "C:/Users/gdut/Desktop/C_D/new_version/color_difference/photo/155-2/kapibala.png",
        ratio=0.75,
        eps=60,
        min_samples=6,
        detector_type="sift",
        matcher_type="flann",
    )
