"""
重新设计的基于纯色的色板分析流水线

本模块提供专注于纯色匹配标准真值的色板分析系统，重点包括：
- 纯主导色提取（非平均值）
- 与标准真值直接颜色匹配，包含CMYK值
- 清晰的Delta E报告和准确性评估
- 简化、专注的颜色准确性测试工作流程
"""

import cv2
import numpy as np
from PIL import Image

from ..color.ground_truth_checker import ground_truth_checker
# 恢复颜色条检测的导入
from .yolo_show import detect_colorbars_yolo, load_yolo_model
# 导入新的YOLO色块检测器
from .yolo_block_detection import detect_blocks_with_yolo, load_yolo_block_model


def extract_pure_color_from_block(
    color_block: np.ndarray,
    purity_threshold: float = 0.8,
    sample_size: tuple[int, int] = (20, 20),
) -> tuple[tuple[int, int, int], float]:
    """
    从颜色块中提取最纯净/主导的颜色。

    该函数专注于找到最具代表性的单一颜色，而不是计算平均值，
    这对纯色分析至关重要。

    Args:
        color_block: 单个颜色块图像（BGR格式）
        purity_threshold: 颜色接受的最小纯度级别（0-1）
        sample_size: 分析的采样尺寸

    Returns:
        颜色元组：(RGB颜色, 纯度分数)
        purity_score: 颜色的"纯度"（1.0 = 完全均匀）
    """
    if color_block.size == 0:
        return (0, 0, 0), 0.0

    # 调整尺寸以进行一致性分析
    if color_block.shape[0] > sample_size[0] or color_block.shape[1] > sample_size[1]:
        resized = cv2.resize(color_block, sample_size, interpolation=cv2.INTER_AREA)
    else:
        resized = color_block

    # 转换为RGB
    rgb_block = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # 提取中心区域进行纯色采样
    h, w = rgb_block.shape[:2]
    center_h, center_w = h // 2, w // 2

    # 从块的中心50%区域采样，避免边缘伖影
    margin_h, margin_w = h // 4, w // 4
    center_region = rgb_block[
        center_h - margin_h : center_h + margin_h,
        center_w - margin_w : center_w + margin_w,
    ]

    if center_region.size == 0:
        center_region = rgb_block

    # 计算颜色统计信息以评估纯度
    pixels = center_region.reshape(-1, 3)

    if len(pixels) == 0:
        return (0, 0, 0), 0.0

    # 使用中位数找到最常见的颜色（比均值更鲁棒）
    median_color = np.median(pixels, axis=0).astype(int)

    # 计算颜色纯度（均匀性）
    # 较低的标准差 = 较高的纯度
    color_std = np.std(pixels, axis=0)
    max_std = np.max(color_std)

    # 纯度分数：颜色变化的倒数
    # max_std为0 = 完美纯度（1.0）
    # max_std为50+ = 低纯度（接近0）
    purity_score = max(0.0, 1.0 - (max_std / 50.0))

    # 使用最频繁的颜色（众数）以更好地检测纯色
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    most_frequent_idx = np.argmax(counts)
    dominant_color = unique_colors[most_frequent_idx]

    # 如果颜色足够纯净，使用主导色，否则使用中位数
    if purity_score >= purity_threshold:
        pure_color = tuple(dominant_color)
    else:
        pure_color = tuple(median_color)

    return pure_color, purity_score


def analyze_pure_color_block(
    color_block: np.ndarray,
    block_id: int = None,
    colorbar_id: int = None,
    purity_threshold: float = 0.8,
) -> dict:
    """
    分析单个颜色块，专注于纯色提取和标准真值匹配。

    Args:
        color_block: 单个颜色块图像（BGR格式）
        block_id: 该块的ID
        colorbar_id: 父色板的ID
        purity_threshold: 颜色接受的最小纯度分数

    Returns:
        包含纯色分析和标准真值比较的字典
    """
    if color_block.size == 0:
        return {"error": "Empty color block"}

    # 提取纯色
    pure_rgb, purity_score = extract_pure_color_from_block(color_block, purity_threshold)

    # 使用现有颜色系统将纯色转换为CMYK
    from .color_analysis import rgb_to_cmyk_icc

    pure_cmyk = rgb_to_cmyk_icc(pure_rgb)

    # 查找最接近的标准真值颜色并计算Delta E
    closest_gt_color, delta_e = ground_truth_checker.find_closest_color(pure_rgb)

    # 创建分析结果
    analysis = {
        "block_id": block_id,
        "colorbar_id": colorbar_id,
        "pure_color_rgb": pure_rgb,
        "pure_color_cmyk": pure_cmyk,
        "purity_score": purity_score,
        "color_quality": _get_color_quality(purity_score),
        "ground_truth_match": {
            "closest_color": {
                "id": closest_gt_color.id,
                "name": closest_gt_color.name,
                "cmyk": closest_gt_color.cmyk,
                "rgb": closest_gt_color.rgb,
            }
            if closest_gt_color
            else None,
            "delta_e": delta_e,
            "accuracy_level": ground_truth_checker._get_accuracy_level(delta_e),
            "is_acceptable": delta_e < 3.0,  # Delta E < 3.0 为可接受
            "is_excellent": delta_e < 1.0,   # Delta E < 1.0 为优秀
        },
        "block_size": color_block.shape[:2],
    }

    return analysis


def _get_color_quality(purity_score: float) -> str:
    """
    根据纯度分数获取颜色质量描述。

    Args:
        purity_score: 颜色纯度分数（0-1）

    Returns:
        质量等级字符串
    """
    if purity_score >= 0.9:
        return "Excellent"    # 优秀
    elif purity_score >= 0.8:
        return "Very Good"    # 很好
    elif purity_score >= 0.7:
        return "Good"         # 好
    elif purity_score >= 0.6:
        return "Fair"         # 一般
    elif purity_score >= 0.5:
        return "Poor"         # 差
    else:
        return "Very Poor"    # 很差


def analyze_colorbar_pure_colors(
    colorbar_blocks: list[np.ndarray],
    colorbar_id: int = None,
    purity_threshold: float = 0.8,  # 将纯度阈值传递到这里
) -> list[dict]:
    """
    分析色板中的多个颜色块，专注于纯色。

    Args:
        colorbar_blocks: 单个颜色块图像列表（BGR格式）
        colorbar_id: 父色板的ID
        purity_threshold: 颜色接受的最小纯度分数

    Returns:
        纯色分析字典列表
    """
    analyses = []

    # 逐个分析每个颜色块
    for i, block in enumerate(colorbar_blocks):
        analysis = analyze_pure_color_block(
            block,
            block_id=i + 1,
            colorbar_id=colorbar_id,
            purity_threshold=purity_threshold,  # 传递纯度阈值
        )
        analyses.append(analysis)

    return analyses


def pure_colorbar_analysis_pipeline(
    pil_image: Image.Image,
    # YOLO颜色条检测参数
    confidence_threshold: float = 0.5,
    box_expansion: int = 10,
    model_path: str = None,
    # 新增：YOLO色块检测参数
    yolo_block_confidence: float = 0.5,
    # 色块过滤参数
    block_area_threshold: int = 50,
    # 纯色分析参数
    purity_threshold: float = 0.8,
    **kwargs,  # 接受并忽略任何其他参数
) -> dict:
    """
    完整的基于纯色的色板分析流水线。

    该流水线专注于：
    1. 使用YOLO检测色板 (best0710.pt)
    2. 对每个色板，使用YOLO检测单个颜色块 (best.pt)
    3. 寻找纯色/主导色（非平均值）
    4. 与标准真值匹配，提供精确的CMYK和Delta E报告

    Args:
        pil_image: 输入PIL图像
        confidence_threshold: YOLO颜色条置信度阈值
        box_expansion: YOLO框扩展像素
        model_path: YOLO颜色条模型路径
        yolo_block_confidence: YOLO色块检测置信度阈值
        block_area_threshold: 色板内块的最小面积
        purity_threshold: 颜色接受的最小纯度分数

    Returns:
        包含完整纯色分析结果的字典
    """
    if pil_image is None:
        return {"error": "No image provided"}

    try:
        # 步骤1：YOLO色板检测
        print("Step 1: Detecting colorbars with YOLO (best0710.pt)...")
        model = load_yolo_model(model_path)

        # 将PIL转换为OpenCV
        opencv_image = np.array(pil_image.convert("RGB"))
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

        # 运行YOLO颜色条检测
        (
            annotated_image,
            colorbar_boxes,
            confidences,
            colorbar_segments,
        ) = detect_colorbars_yolo(
            opencv_image,
            model,
            box_expansion=box_expansion,
            confidence_threshold=confidence_threshold,
        )

        if len(colorbar_segments) == 0:
            return {
                "error": "No colorbars detected",
                "annotated_image": Image.fromarray(
                    cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                ),
                "step_completed": 1,
            }

        # 步骤2：初始化YOLO色块检测器 (best.pt)
        print("Step 2: Initializing YOLO block detector (best.pt)...")
        try:
            block_model = load_yolo_block_model()
        except (FileNotFoundError, RuntimeError) as e:
            return {"error": str(e)}

        colorbar_results = []

        for i, (segment, confidence, box) in enumerate(
            zip(colorbar_segments, confidences, colorbar_boxes, strict=False)
        ):
            colorbar_id = i + 1
            print(f"  Processing colorbar {colorbar_id}/{len(colorbar_segments)}...")

            # 步骤2.1: 使用YOLO检测色块 (替换旧逻辑)
            (
                segmented_colorbar,
                color_blocks,
                block_count,
            ) = detect_blocks_with_yolo(
                segment,
                block_model,
                confidence_threshold=yolo_block_confidence,
                min_area=block_area_threshold,
            )

            # 步骤3：每个块的纯色分析
            print(f"  Analyzing {block_count} pure colors in colorbar {colorbar_id}...")
            pure_color_analyses = []

            if block_count > 0:
                pure_color_analyses = analyze_colorbar_pure_colors(
                    color_blocks,
                    colorbar_id=colorbar_id,
                    purity_threshold=purity_threshold,
                )

            # 将片段转换为PIL以便更好的界面集成
            colorbar_result = {
                "colorbar_id": colorbar_id,
                "confidence": confidence,
                "bounding_box": box,
                "original_segment_pil": Image.fromarray(
                    cv2.cvtColor(segment, cv2.COLOR_BGR2RGB)
                ),
                "segmented_colorbar_pil": Image.fromarray(
                    cv2.cvtColor(segmented_colorbar, cv2.COLOR_BGR2RGB)
                ),
                "color_blocks": color_blocks,
                "block_count": block_count,
                "pure_color_analyses": pure_color_analyses,
            }

            colorbar_results.append(colorbar_result)

        # 计算总体统计信息
        total_blocks = sum(result["block_count"] for result in colorbar_results)
        all_delta_e_values = []
        excellent_count = 0
        acceptable_count = 0
        high_purity_count = 0

        for result in colorbar_results:
            for analysis in result["pure_color_analyses"]:
                if "error" not in analysis:
                    gt_match = analysis["ground_truth_match"]
                    delta_e = gt_match["delta_e"]
                    all_delta_e_values.append(delta_e)

                    if gt_match["is_excellent"]:
                        excellent_count += 1
                    if gt_match["is_acceptable"]:
                        acceptable_count += 1

                    if analysis["purity_score"] >= 0.8:
                        high_purity_count += 1

        # 计算准确性统计信息
        accuracy_stats = {}
        if all_delta_e_values:
            import statistics

            total_analyzed = len(all_delta_e_values)
            accuracy_stats = {
                "average_delta_e": statistics.mean(all_delta_e_values),
                "median_delta_e": statistics.median(all_delta_e_values),
                "max_delta_e": max(all_delta_e_values),
                "min_delta_e": min(all_delta_e_values),
                "excellent_colors": excellent_count,
                "acceptable_colors": acceptable_count,
                "high_purity_colors": high_purity_count,
                "total_analyzed": total_analyzed,
                "excellent_percentage": (excellent_count / total_analyzed) * 100 if total_analyzed > 0 else 0,
                "acceptable_percentage": (acceptable_count / total_analyzed) * 100 if total_analyzed > 0 else 0,
                "high_purity_percentage": (high_purity_count / total_analyzed) * 100 if total_analyzed > 0 else 0,
            }

        return {
            "success": True,
            "analysis_type": "direct_yolo_block_detection",
            "annotated_image": annotated_image,
            "colorbar_count": len(colorbar_results),
            "colorbar_results": colorbar_results,
            "total_blocks": total_blocks,
            "accuracy_statistics": accuracy_stats,
            "step_completed": 3,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": f"Error in pure colorbar analysis pipeline: {str(e)}",
            "step_completed": 0,
        }


def pure_colorbar_analysis_for_gradio(
    pil_image: Image.Image,
    confidence_threshold: float = 0.5,
    box_expansion: int = 10,
    # 新增YOLO色块检测参数
    yolo_block_confidence: float = 0.5,
    # 色块过滤参数
    block_area_threshold: int = 50,
    purity_threshold: float = 0.8,
    **kwargs,  # 接受并忽略任何其他参数
) -> tuple[Image.Image, list[dict], str, int]:
    """
    为Gradio界面优化的纯色色板分析流水线包装器。

    Returns:
        返回元组：(
            标注图像,
            包含纯色的色板数据,
            分析报告,
            找到的总块数
        )
    """
    # 运行完整的纯色分析流水线
    result = pure_colorbar_analysis_pipeline(
        pil_image,
        confidence_threshold=confidence_threshold,
        box_expansion=box_expansion,
        yolo_block_confidence=yolo_block_confidence,
        block_area_threshold=block_area_threshold,
        purity_threshold=purity_threshold,
    )

    if "error" in result:
        error_img = pil_image if pil_image else None
        return error_img, [], f"❌ {result['error']}", 0

    if not result.get("success", False):
        return pil_image, [], "❌ Pure color analysis failed", 0

    # 将标注图像转换为PIL
    annotated_pil = Image.fromarray(
        cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB)
    )

    # 准备包含纯色分析的色板数据
    colorbar_data = result.get("colorbar_results", [])

    # 构建综合分析报告
    report = "🎯 YOLO Direct Block Analysis Results\n"
    report += "=" * 55 + "\n\n"
    report += "📊 Summary:\n"
    report += f"  • Total color blocks found: {result['total_blocks']}\n"

    # 添加准确性统计信息
    accuracy_stats = result.get("accuracy_statistics", {})
    if accuracy_stats:
        report += f"  • Average ΔE: {accuracy_stats['average_delta_e']:.2f}\n"
        report += f"  • ΔE Range: {accuracy_stats['min_delta_e']:.2f} - {accuracy_stats['max_delta_e']:.2f}\n"
        report += f"  • Excellent colors (ΔE < 1.0): {accuracy_stats['excellent_colors']}/{accuracy_stats['total_analyzed']} ({accuracy_stats['excellent_percentage']:.1f}%)\n"
        report += f"  • Acceptable colors (ΔE < 3.0): {accuracy_stats['acceptable_colors']}/{accuracy_stats['total_analyzed']} ({accuracy_stats['acceptable_percentage']:.1f}%)\n"
        report += f"  • High purity colors: {accuracy_stats['high_purity_colors']}/{accuracy_stats['total_analyzed']} ({accuracy_stats['high_purity_percentage']:.1f}%)\n"

    report += "\n"

    for colorbar_result in result["colorbar_results"]:
        block_count = colorbar_result["block_count"]
        report += f"🎨 Detected Blocks (Total: {block_count}):\n"

        if block_count > 0:
            report += "  • Pure colors with CMYK values and delta E:\n"

            # 为每个块添加详细的纯色分析
            for analysis in colorbar_result["pure_color_analyses"]:
                if "error" not in analysis:
                    block_id = analysis.get("block_id", "?")
                    pure_rgb = analysis["pure_color_rgb"]
                    pure_cmyk = analysis["pure_color_cmyk"]
                    purity_score = analysis["purity_score"]
                    color_quality = analysis["color_quality"]
                    pure_hex = f"#{pure_rgb[0]:02x}{pure_rgb[1]:02x}{pure_rgb[2]:02x}"

                    report += f"    {block_id}: {pure_hex} "
                    report += f"(C={pure_cmyk[0]}% M={pure_cmyk[1]}% Y={pure_cmyk[2]}% K={pure_cmyk[3]}%)"

                    # 添加纯度信息
                    report += f" | Purity: {purity_score:.2f} ({color_quality})"

                    # 添加标准真值比较
                    gt_match = analysis["ground_truth_match"]
                    if gt_match["closest_color"]:
                        delta_e = gt_match["delta_e"]
                        accuracy_level = gt_match["accuracy_level"]
                        gt_color = gt_match["closest_color"]

                        report += f" | ΔE: {delta_e:.2f} ({accuracy_level})"
                        report += f" vs {gt_color['name']}"

                        # 添加状态指示器
                        if gt_match["is_excellent"]:
                            report += " ✅"
                        elif gt_match["is_acceptable"]:
                            report += " ⚠️"
                        else:
                            report += " ❌"

                    report += "\n"

        report += "\n"

    return (annotated_pil, colorbar_data, report, result["total_blocks"])
