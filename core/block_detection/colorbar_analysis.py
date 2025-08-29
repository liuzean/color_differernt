"""
智能色板分析流水线
高级三步骤工作流程，用于色板检测和颜色分析：
1. YOLO检测色板区域
2. 在每个色板内进行块检测
3. 使用CMYK转换和Delta E计算进行单个颜色块分析

功能特点：
- 包含原始分割的色板图像
- 密集、有组织的结果显示
- 使用ICC配置文件进行CMYK颜色转换
- 与标准真值颜色比较，包含Delta E计算
"""

import cv2
import numpy as np
from PIL import Image

from ..color.ground_truth_checker import ground_truth_checker
from .blocks_detect import detect_blocks
from .color_analysis import analyze_colorbar_blocks
from .yolo_show import detect_colorbars_yolo, load_yolo_model


def extract_blocks_from_colorbar(
    colorbar_segment: np.ndarray,
    area_threshold: int = 50,
    aspect_ratio_threshold: float = 0.3,
    min_square_size: int = 5,
) -> tuple[np.ndarray, list[np.ndarray], int]:
    """
    专门对色板片段应用块检测以提取单个颜色色块。

    Args:
        colorbar_segment: 色板图像片段（BGR格式）
        area_threshold: 检测块的最小面积
        aspect_ratio_threshold: 块的最小长宽比
        min_square_size: 检测块的最小宽度和高度（像素）

    Returns:
        返回元组：(标注色板, 颜色块列表, 块数量)
    """
    if colorbar_segment.size == 0:
        return colorbar_segment, [], 0

    # 使用现有的块检测函数，调整参数适用于色板
    result_image, block_images, block_count = detect_blocks(
        colorbar_segment,
        output_dir=None,  # 不保存文件
        area_threshold=area_threshold,
        aspect_ratio_threshold=aspect_ratio_threshold,
        min_square_size=min_square_size,
        return_individual_blocks=True,
    )

    return result_image, block_images, block_count


def enhance_with_ground_truth_comparison(block_analyses: list[dict]) -> list[dict]:
    """
    通过标准真值颜色比较和Delta E计算增强块分析。

    Args:
        block_analyses: 块分析字典列表

    Returns:
        增强的包含标准真值比较数据的块分析
    """
    enhanced_analyses = []

    for analysis in block_analyses:
        # 如果分析中有错误，跳过
        if "error" in analysis:
            enhanced_analyses.append(analysis)
            continue

        # 提取主要RGB颜色
        primary_rgb = analysis.get("primary_color_rgb")
        if not primary_rgb:
            enhanced_analyses.append(analysis)
            continue

        # 查找最接近的标准真值颜色并计算Delta E
        closest_gt_color, delta_e = ground_truth_checker.find_closest_color(primary_rgb)

        # 使用标准真值比较增强分析
        enhanced_analysis = analysis.copy()
        enhanced_analysis.update(
            {
                "ground_truth_comparison": {
                    "closest_color": {
                        "id": closest_gt_color.id,
                        "name": closest_gt_color.name,
                        "cmyk": closest_gt_color.cmyk,
                        "rgb": closest_gt_color.rgb,
                        "lab": closest_gt_color.lab,
                    }
                    if closest_gt_color
                    else None,
                    "delta_e": delta_e,
                    "accuracy_level": ground_truth_checker._get_accuracy_level(delta_e),
                    "is_acceptable": delta_e
                    < 3.0,  # Delta E < 3 是可接受颜色的阈值
                }
            }
        )

        enhanced_analyses.append(enhanced_analysis)

    return enhanced_analyses


def colorbar_analysis_pipeline(
    pil_image: Image.Image,
    # YOLO参数
    confidence_threshold: float = 0.5,
    box_expansion: int = 10,
    model_path: str = None,
    # 块检测参数
    block_area_threshold: int = 50,
    block_aspect_ratio: float = 0.3,
    min_square_size: int = 5,
    # 颜色分析参数
    shrink_size: tuple[int, int] = (30, 30),
) -> dict:
    """
    完整的智能色板分析流水线。

    Args:
        pil_image: 输入PIL图像
        confidence_threshold: YOLO置信度阈值
        box_expansion: YOLO框扩展像素
        model_path: YOLO模型路径
        block_area_threshold: 色板内块的最小面积
        block_aspect_ratio: 块的最小长宽比
        min_square_size: 检测块的最小宽度和高度（像素）
        shrink_size: 为颜色分析缩小块的尺寸

    Returns:
        包含完整分析结果的字典，包括原始色板片段
    """
    if pil_image is None:
        return {"error": "No image provided"}

    try:
        # 步骤1：YOLO色板检测
        print("Step 1: Detecting colorbars with YOLO...")
        model = load_yolo_model(model_path)

        # 将PIL转换为OpenCV
        opencv_image = np.array(pil_image)
        if len(opencv_image.shape) == 3:
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

        # 运行YOLO检测
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

        # 步骤2：在每个色板内进行块检测
        print(
            f"Step 2: Detecting blocks within {len(colorbar_segments)} colorbar(s)..."
        )
        colorbar_results = []

        for i, (segment, confidence, box) in enumerate(
            zip(colorbar_segments, confidences, colorbar_boxes, strict=False)
        ):
            colorbar_id = i + 1
            print(f"  Processing colorbar {colorbar_id}/{len(colorbar_segments)}...")

            # 从此色板提取块
            (
                segmented_colorbar,
                color_blocks,
                block_count,
            ) = extract_blocks_from_colorbar(
                segment,
                area_threshold=block_area_threshold,
                aspect_ratio_threshold=block_aspect_ratio,
                min_square_size=min_square_size,
            )

            # 步骤3：分析每个块中的颜色
            print(
                f"  Step 3: Analyzing {block_count} color blocks in colorbar {colorbar_id}..."
            )
            block_analyses = []

            if block_count > 0:
                block_analyses = analyze_colorbar_blocks(
                    color_blocks, shrink_size, colorbar_id=colorbar_id
                )

                # 步骤4：计算与标准真值颜色的Delta E
                print("  Step 4: Calculating delta E against ground truth colors...")
                block_analyses = enhance_with_ground_truth_comparison(block_analyses)

            # 将片段转换为PIL以便更好的界面集成
            original_segment_pil = None
            segmented_colorbar_pil = None

            if segment.size > 0:
                original_segment_pil = Image.fromarray(
                    cv2.cvtColor(segment, cv2.COLOR_BGR2RGB)
                )
            if segmented_colorbar.size > 0:
                segmented_colorbar_pil = Image.fromarray(
                    cv2.cvtColor(segmented_colorbar, cv2.COLOR_BGR2RGB)
                )

            colorbar_result = {
                "colorbar_id": colorbar_id,
                "confidence": confidence,
                "bounding_box": box,
                "original_segment": segment,  # 原始OpenCV格式
                "original_segment_pil": original_segment_pil,  # 用于显示的PIL格式
                "segmented_colorbar": segmented_colorbar,  # OpenCV格式
                "segmented_colorbar_pil": segmented_colorbar_pil,  # 用于显示的PIL格式
                "color_blocks": color_blocks,
                "block_count": block_count,
                "block_analyses": block_analyses,
            }

            colorbar_results.append(colorbar_result)

        return {
            "success": True,
            "annotated_image": annotated_image,
            "colorbar_count": len(colorbar_segments),
            "colorbar_results": colorbar_results,
            "total_blocks": sum(result["block_count"] for result in colorbar_results),
            "step_completed": 3,
        }

    except Exception as e:
        return {
            "error": f"Error in colorbar analysis pipeline: {str(e)}",
            "step_completed": 0,
        }


def colorbar_analysis_for_gradio(
    pil_image: Image.Image,
    confidence_threshold: float = 0.5,
    box_expansion: int = 10,
    block_area_threshold: int = 50,
    block_aspect_ratio: float = 0.3,
    min_square_size: int = 5,
    shrink_size: tuple[int, int] = (30, 30),
) -> tuple[Image.Image, list[dict], str, int]:
    """
    为Gradio界面优化的色板分析流水线包装器。

    Returns:
        返回元组：(
            标注图像,
            包含原始片段的色板数据,
            分析报告,
            找到的总块数
        )
    """
    # 运行完整的色板分析流水线
    result = colorbar_analysis_pipeline(
        pil_image,
        confidence_threshold=confidence_threshold,
        box_expansion=box_expansion,
        block_area_threshold=block_area_threshold,
        block_aspect_ratio=block_aspect_ratio,
        min_square_size=min_square_size,
        shrink_size=shrink_size,
    )

    if "error" in result:
        error_img = pil_image if pil_image else None
        return error_img, [], f"❌ {result['error']}", 0

    if not result.get("success", False):
        return pil_image, [], "❌ Analysis failed", 0

    # 将标注图像转换为PIL
    annotated_pil = Image.fromarray(
        cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB)
    )

    # 准备包含原始片段的色板数据
    colorbar_data = []

    # 构建包含Delta E统计信息的综合分析报告
    report = "🎯 Intelligent Colorbar Analysis Results\n"
    report += "=" * 50 + "\n\n"
    report += "📊 Summary:\n"
    report += f"  • Colorbars detected: {result['colorbar_count']}\n"
    report += f"  • Total color blocks: {result['total_blocks']}\n"

    # 计算Delta E统计信息
    all_delta_e_values = []
    acceptable_count = 0
    total_analyzed = 0

    for colorbar_result in result["colorbar_results"]:
        for analysis in colorbar_result["block_analyses"]:
            if "error" not in analysis:
                ground_truth_data = analysis.get("ground_truth_comparison")
                if ground_truth_data:
                    delta_e = ground_truth_data.get("delta_e", 0)
                    all_delta_e_values.append(delta_e)
                    total_analyzed += 1
                    if delta_e < 3.0:
                        acceptable_count += 1

    # 如果有Delta E值，计算统计信息
    if all_delta_e_values:
        import statistics

        avg_delta_e = statistics.mean(all_delta_e_values)
        max_delta_e = max(all_delta_e_values)
        min_delta_e = min(all_delta_e_values)
        accuracy_percentage = (acceptable_count / total_analyzed) * 100

        report += f"  • Average ΔE: {avg_delta_e:.2f}\n"
        report += f"  • ΔE Range: {min_delta_e:.2f} - {max_delta_e:.2f}\n"
        report += f"  • Acceptable colors (ΔE < 3.0): {acceptable_count}/{total_analyzed} ({accuracy_percentage:.1f}%)\n"

    report += "\n"

    # 为每个色板生成详细报告
    for colorbar_result in result["colorbar_results"]:
        colorbar_id = colorbar_result["colorbar_id"]
        confidence = colorbar_result["confidence"]
        block_count = colorbar_result["block_count"]

        # 将颜色块转换为PIL
        color_blocks_pil = []
        for block in colorbar_result["color_blocks"]:
            if block.size > 0:
                block_pil = Image.fromarray(cv2.cvtColor(block, cv2.COLOR_BGR2RGB))
                color_blocks_pil.append(block_pil)

        # 创建包含原始片段的色板数据条目
        colorbar_entry = {
            "colorbar_id": colorbar_id,
            "confidence": confidence,
            "original_colorbar": colorbar_result[
                "original_segment_pil"
            ],  # ✨ 原始片段
            "segmented_colorbar": colorbar_result["segmented_colorbar_pil"],
            "color_blocks": color_blocks_pil,
            "block_count": block_count,
            "block_analyses": colorbar_result["block_analyses"],
        }
        colorbar_data.append(colorbar_entry)

        # 向报告添加密集的色板部分
        report += f"🎨 Colorbar {colorbar_id} (confidence: {confidence:.2f}):\n"
        report += f"  • Color blocks found: {block_count}\n"

        if block_count > 0:
            report += "  • Colors with CMYK values and delta E:\n"

            # 为每个块添加简洁的颜色分析
            for analysis in colorbar_result["block_analyses"]:
                if "error" not in analysis:
                    block_id = analysis.get("block_id", "?")
                    primary_rgb = analysis["primary_color_rgb"]
                    primary_cmyk = analysis["primary_color_cmyk"]
                    primary_hex = (
                        f"#{primary_rgb[0]:02x}{primary_rgb[1]:02x}{primary_rgb[2]:02x}"
                    )

                    report += f"    {colorbar_id}.{block_id}: {primary_hex} "
                    report += f"(C={primary_cmyk[0]}% M={primary_cmyk[1]}% Y={primary_cmyk[2]}% K={primary_cmyk[3]}%)"

                    # 如果可用，添加Delta E信息
                    ground_truth_data = analysis.get("ground_truth_comparison")
                    if ground_truth_data:
                        delta_e = ground_truth_data.get("delta_e", 0)
                        accuracy_level = ground_truth_data.get(
                            "accuracy_level", "Unknown"
                        )
                        closest_color = ground_truth_data.get("closest_color")

                        report += f" → ΔE: {delta_e:.2f} ({accuracy_level})"
                        if closest_color:
                            report += f" vs {closest_color['name']}"

                    report += "\n"

        report += "\n"

    return (annotated_pil, colorbar_data, report, result["total_blocks"])


# 向后兼容性别名
enhanced_colorbar_analysis = colorbar_analysis_pipeline
enhanced_colorbar_analysis_from_pil = colorbar_analysis_for_gradio
