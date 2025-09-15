"""
高级色块分析模块：对单个或多个色块进行颜色特征提取，包括RGB、CMYK、HSV、主要颜色占比等。
    colorbar_segment: OpenCV格式的色板图像片段
    Returns:
        返回颜色分析结果字典。
调用外部函数srgb_to_cmyk_array进行高精度RGB到CMYK转换（如果可用）。路径：from ..color.icc_trans import srgb_to_cmyk_array

"""

import colorsys
import cv2
import numpy as np
from sklearn.cluster import KMeans

# 尝试导入基于ICC的sRGB转CMYK转换函数
try:
    from ..color.icc_trans import srgb_to_cmyk_array
except ImportError:
    srgb_to_cmyk_array = None


def rgb_to_cmyk_icc(rgb_tuple: tuple[int, int, int]) -> tuple[int, int, int, int]:
    """
    将RGB颜色转换为CMYK，优先使用ICC配置文件进行高精度转换。
    如果ICC方法不可用，则使用简单算法。
    """
    if srgb_to_cmyk_array is None:
        # 如果ICC方法不可用，使用简单RGB→CMYK计算
        return rgb_to_cmyk_simple(rgb_tuple)

    try:
        # 将RGB元组转换为BGR格式，并创建小图用于ICC转换
        r, g, b = rgb_tuple
        rgb_array = np.array([[[b, g, r]]], dtype=np.uint8)  # OpenCV使用BGR顺序

        # 使用ICC转换函数
        cmyk_array, _ = srgb_to_cmyk_array(rgb_array)

        # 从ICC结果提取CMYK值（0-255），转换为百分比
        if cmyk_array.size >= 4:
            c, m, y, k = cmyk_array[0, 0, :]
            return (
                int(c * 100 / 255),
                int(m * 100 / 255),
                int(y * 100 / 255),
                int(k * 100 / 255),
            )
        else:
            return rgb_to_cmyk_simple(rgb_tuple)

    except Exception:
        # 如果ICC转换失败，回退到简单算法
        return rgb_to_cmyk_simple(rgb_tuple)


def rgb_to_cmyk_simple(rgb_tuple: tuple[int, int, int]) -> tuple[int, int, int, int]:
    """
    简单的RGB到CMYK转换（备用方案）。
    转换范围：CMYK 0-100%
    """
    r, g, b = [x / 255.0 for x in rgb_tuple]

    # 计算黑色分量
    k = 1 - max(r, g, b)

    if k == 1:
        return (0, 0, 0, 100)

    # 计算CMY
    c = (1 - r - k) / (1 - k)
    m = (1 - g - k) / (1 - k)
    y = (1 - b - k) / (1 - k)

    return (int(c * 100), int(m * 100), int(y * 100), int(k * 100))


def shrink_image(
    image: np.ndarray, target_size: tuple[int, int] = (50, 50)
) -> np.ndarray:
    """
    将输入图像缩小到指定尺寸，用于后续颜色分析。
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def get_dominant_colors(
    image: np.ndarray, k: int = 5, image_processing_size: tuple[int, int] = (25, 25)
) -> list[tuple[tuple[int, int, int], float]]:
    """
    提取图像的主要颜色（K-Means聚类）。
    返回结果：[(RGB颜色, 占比), ...]
    """
    # 缩小图像以加速聚类
    image_resized = cv2.resize(
        image, image_processing_size, interpolation=cv2.INTER_AREA
    )

    # 转换为RGB
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    # 转换为像素点列表
    pixels = image_rgb.reshape((-1, 3))

    # 使用KMeans聚类提取k种颜色
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)

    # 获取聚类中心（颜色）和标签
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_

    # 统计每个聚类的像素数
    (unique, counts) = np.unique(labels, return_counts=True)

    # 计算颜色占比
    total_pixels = len(pixels)
    color_percentages = []
    for i, count in enumerate(counts):
        percentage = count / total_pixels
        rgb_color = tuple(colors[unique[i]].astype(int))
        color_percentages.append((rgb_color, percentage))

    # 按占比从大到小排序
    color_percentages.sort(key=lambda x: x[1], reverse=True)

    return color_percentages


def analyze_single_block_color(
    block_image: np.ndarray,
    shrink_size: tuple[int, int] = (30, 30),
    colorbar_id: int = None,
    block_id: int = None,
) -> dict:
    """
    分析单个色块，输出全面的颜色特征（RGB、CMYK、HSV、主要颜色、均匀性等）。
    """
    if block_image.size == 0:
        return {"error": "色块图像为空"}

    # 缩小图像
    shrunken = shrink_image(block_image, shrink_size)

    # 转换为RGB
    rgb_image = cv2.cvtColor(shrunken, cv2.COLOR_BGR2RGB)

    # 提取主色（前三个）
    dominant_colors = get_dominant_colors(
        block_image, k=3, image_processing_size=shrink_size
    )

    # 计算平均颜色
    avg_color_bgr = np.mean(shrunken.reshape(-1, 3), axis=0)
    avg_color_rgb = avg_color_bgr[::-1]

    # 转换为HSV
    avg_color_hsv = colorsys.rgb_to_hsv(
        avg_color_rgb[0] / 255, avg_color_rgb[1] / 255, avg_color_rgb[2] / 255
    )
    avg_color_hsv = (
        int(avg_color_hsv[0] * 360),
        int(avg_color_hsv[1] * 100),
        int(avg_color_hsv[2] * 100),
    )

    # 计算颜色方差与标准差（判断颜色是否均匀）
    color_variance = np.var(rgb_image.reshape(-1, 3), axis=0)
    color_std = np.std(rgb_image.reshape(-1, 3), axis=0)

    # 判断是否为纯色
    is_solid = np.all(color_std < 20)

    # 取最主要颜色
    primary_color = dominant_colors[0][0] if dominant_colors else (0, 0, 0)
    primary_percentage = dominant_colors[0][1] if dominant_colors else 0.0

    # 转换为CMYK
    primary_color_cmyk = rgb_to_cmyk_icc(primary_color)
    avg_color_rgb_int = tuple(avg_color_rgb.astype(int))
    avg_color_cmyk = rgb_to_cmyk_icc(avg_color_rgb_int)

    return {
        "colorbar_id": colorbar_id,
        "block_id": block_id,
        "primary_color_rgb": primary_color,
        "primary_color_cmyk": primary_color_cmyk,
        "primary_color_percentage": primary_percentage,
        "average_color_rgb": avg_color_rgb_int,
        "average_color_cmyk": avg_color_cmyk,
        "average_color_hsv": avg_color_hsv,
        "dominant_colors": dominant_colors[:3],
        "color_variance": color_variance.tolist(),
        "color_std": color_std.tolist(),
        "is_solid_color": is_solid,
        "block_size": block_image.shape[:2],
        "shrunken_size": shrunken.shape[:2],
        "total_pixels": block_image.shape[0] * block_image.shape[1],
    }


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    """将RGB颜色转换为Hex颜色值（#RRGGBB）。"""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def format_color_analysis_report(analysis: dict, block_identifier: str = None) -> str:
    """
    格式化颜色分析结果，输出报告字符串。
    """
    if "error" in analysis:
        return f"色块 {block_identifier}: {analysis['error']}"

    if block_identifier is None:
        colorbar_id = analysis.get("colorbar_id", "?")
        block_id = analysis.get("block_id", "?")
        block_identifier = f"{colorbar_id}.{block_id}"

    report = f"🎨 色块 {block_identifier} 分析结果:\n"

    # 主色
    primary_rgb = analysis["primary_color_rgb"]
    primary_cmyk = analysis["primary_color_cmyk"]
    primary_hex = rgb_to_hex(primary_rgb)
    report += f"  • 主色: RGB{primary_rgb} ({primary_hex}) - {analysis['primary_color_percentage']:.1%}\n"
    report += f"    └─ CMYK: C={primary_cmyk[0]}% M={primary_cmyk[1]}% Y={primary_cmyk[2]}% K={primary_cmyk[3]}%\n"

    # 平均色
    avg_rgb = analysis["average_color_rgb"]
    avg_cmyk = analysis["average_color_cmyk"]
    avg_hex = rgb_to_hex(avg_rgb)
    avg_hsv = analysis["average_color_hsv"]
    report += f"  • 平均色: RGB{avg_rgb} ({avg_hex})\n"
    report += f"    └─ CMYK: C={avg_cmyk[0]}% M={avg_cmyk[1]}% Y={avg_cmyk[2]}% K={avg_cmyk[3]}%\n"
    report += f"    └─ HSV: H={avg_hsv[0]}° S={avg_hsv[1]}% V={avg_hsv[2]}%\n"

    # 颜色类型
    if analysis["is_solid_color"]:
        report += "  • 类型: 纯色\n"
    else:
        report += "  • 类型: 渐变/混合\n"

    # 主导颜色列表
    report += "  • 主导颜色列表:\n"
    for i, (color, percentage) in enumerate(analysis["dominant_colors"][:2]):
        hex_color = rgb_to_hex(color)
        cmyk_color = rgb_to_cmyk_icc(color)
        report += f"    {i + 1}. RGB{color} ({hex_color}) - {percentage:.1%}\n"
        report += f"       └─ CMYK: C={cmyk_color[0]}% M={cmyk_color[1]}% Y={cmyk_color[2]}% K={cmyk_color[3]}%\n"

    # 图像尺寸
    report += f"  • 原始尺寸: {analysis['block_size']}\n"
    report += f"  • 分析尺寸: {analysis['shrunken_size']}\n"

    return report


def analyze_colorbar_blocks(
    colorbar_blocks: list[np.ndarray],
    shrink_size: tuple[int, int] = (30, 30),
    colorbar_id: int = None,
) -> list[dict]:
    """
    批量分析一个颜色条中的多个色块。
    """
    analyses = []
    for i, block in enumerate(colorbar_blocks):
        analysis = analyze_single_block_color(
            block, shrink_size, colorbar_id=colorbar_id, block_id=i + 1
        )
        analyses.append(analysis)
    return analyses
