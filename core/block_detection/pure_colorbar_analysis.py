# core/block_detection/pure_colorbar_analysis.py

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
    """
    if color_block.size == 0:
        return (0, 0, 0), 0.0

    if color_block.shape[0] > sample_size[0] or color_block.shape[1] > sample_size[1]:
        resized = cv2.resize(color_block, sample_size, interpolation=cv2.INTER_AREA)
    else:
        resized = color_block

    rgb_block = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    h, w = rgb_block.shape[:2]
    center_h, center_w = h // 2, w // 2
    margin_h, margin_w = h // 4, w // 4
    center_region = rgb_block[
        center_h - margin_h : center_h + margin_h,
        center_w - margin_w : center_w + margin_w,
    ]

    if center_region.size == 0:
        center_region = rgb_block

    pixels = center_region.reshape(-1, 3)
    if len(pixels) == 0:
        return (0, 0, 0), 0.0

    median_color = np.median(pixels, axis=0).astype(int)
    color_std = np.std(pixels, axis=0)
    max_std = np.max(color_std)
    purity_score = max(0.0, 1.0 - (max_std / 50.0))
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    most_frequent_idx = np.argmax(counts)
    dominant_color = unique_colors[most_frequent_idx]

    if purity_score >= purity_threshold:
        pure_color = tuple(dominant_color)
    else:
        pure_color = tuple(median_color)

    return pure_color, purity_score


def _extract_block_color_features(
    color_block: np.ndarray,
    block_id: int = None,
    colorbar_id: int = None,
    purity_threshold: float = 0.8,
) -> dict:
    """
    Extracts color features from a single color block without performing ground truth matching.
    """
    if color_block.size == 0:
        return {"error": "Empty color block"}

    pure_rgb, purity_score = extract_pure_color_from_block(color_block, purity_threshold)
    from .color_analysis import rgb_to_cmyk_icc
    pure_cmyk = rgb_to_cmyk_icc(pure_rgb)

    analysis = {
        "block_id": block_id,
        "colorbar_id": colorbar_id,
        "pure_color_rgb": pure_rgb,
        "pure_color_cmyk": pure_cmyk,
        "purity_score": purity_score,
        "color_quality": _get_color_quality(purity_score),
        "block_size": color_block.shape[:2],
    }
    return analysis


# [恢复] 恢复兼容性函数
def analyze_pure_color_block(
    color_block: np.ndarray,
    block_id: int = None,
    colorbar_id: int = None,
    purity_threshold: float = 0.8,
) -> dict:
    """
    [兼容性保留] Analyzes a single color block, focusing on pure color extraction and ground truth matching.
    # WARNING: This function is deprecated. It performs a single-color match against the default
    # ground truth card ('card_001') only. For multi-card matching, use the new pipeline.
    """
    analysis = _extract_block_color_features(
        color_block, block_id, colorbar_id, purity_threshold
    )
    if "error" in analysis:
        return analysis

    closest_gt_color, delta_e = ground_truth_checker.find_closest_color(
        analysis["pure_color_rgb"]
    )

    analysis["ground_truth_match"] = {
        "closest_color": {
            "id": closest_gt_color.id, "name": closest_gt_color.name,
            "cmyk": closest_gt_color.cmyk, "rgb": closest_gt_color.rgb,
        } if closest_gt_color else None,
        "delta_e": delta_e,
        "accuracy_level": ground_truth_checker._get_accuracy_level(delta_e),
        "is_acceptable": delta_e < 3.0, "is_excellent": delta_e < 1.0,
    }
    return analysis


def _get_color_quality(purity_score: float) -> str:
    """
    根据纯度分数获取颜色质量描述。
    """
    if purity_score >= 0.9: return "Excellent"
    elif purity_score >= 0.8: return "Very Good"
    elif purity_score >= 0.7: return "Good"
    elif purity_score >= 0.6: return "Fair"
    elif purity_score >= 0.5: return "Poor"
    else: return "Very Poor"


def analyze_colorbar_with_best_card_match(
    colorbar_blocks: list[np.ndarray],
    colorbar_id: int = None,
    purity_threshold: float = 0.8,
) -> tuple[list[dict], str | None]:
    """
    [新流程] Analyzes multiple color blocks from a colorbar, finds the best matching
    standard color card, and returns detailed comparisons.
    """
    if not colorbar_blocks:
        return [], None

    block_features, detected_rgb_colors = [], []
    for i, block in enumerate(colorbar_blocks):
        features = _extract_block_color_features(
            block, block_id=i + 1, colorbar_id=colorbar_id, purity_threshold=purity_threshold,
        )
        block_features.append(features)
        if "error" not in features:
            detected_rgb_colors.append(features["pure_color_rgb"])

    if not detected_rgb_colors:
        return block_features, None

    card_match_result = ground_truth_checker.find_best_card_for_colorbar(detected_rgb_colors)
    if not card_match_result:
        return block_features, None

    best_card_id = card_match_result["best_card_id"]
    match_results = card_match_result["results"]

    if best_card_id == "INVALID_DETECTION":
        return block_features, best_card_id

    final_analyses = []
    for i, features in enumerate(block_features):
        if "error" in features:
            final_analyses.append(features)
            continue
        if i < len(match_results):
            match = match_results[i]
            gt_color = match["closest_ground_truth"]
            features["ground_truth_match"] = {
                "closest_color": {
                    "id": gt_color.id, "name": gt_color.name,
                    "cmyk": gt_color.cmyk, "rgb": gt_color.rgb,
                },
                "delta_e": match["delta_e"],
                "accuracy_level": match["accuracy_level"],
                "is_acceptable": match["delta_e"] < 3.0,
                "is_excellent": match["delta_e"] < 1.0,
            }
        final_analyses.append(features)
    return final_analyses, best_card_id


# [恢复] 恢复兼容性函数
def analyze_colorbar_pure_colors(
    colorbar_blocks: list[np.ndarray],
    colorbar_id: int = None,
    purity_threshold: float = 0.8,
) -> list[dict]:
    """
    [兼容性保留] Analyzes multiple color blocks in a colorbar, focusing on pure colors.
    # WARNING: This function is deprecated. It uses the old single-match logic for each block.
    # Use `analyze_colorbar_with_best_card_match` for the new multi-card matching pipeline.
    """
    analyses = []
    for i, block in enumerate(colorbar_blocks):
        analysis = analyze_pure_color_block(
            block, block_id=i + 1, colorbar_id=colorbar_id, purity_threshold=purity_threshold,
        )
        analyses.append(analysis)
    return analyses


def pure_colorbar_analysis_pipeline(
    pil_image: Image.Image,
    confidence_threshold: float = 0.5,
    box_expansion: int = 10,
    model_path: str = None,
    yolo_block_confidence: float = 0.5,
    block_area_threshold: int = 50,
    purity_threshold: float = 0.8,
    **kwargs,
) -> dict:
    """
    完整的基于纯色的色板分析流水线。
    """
    if pil_image is None: return {"error": "No image provided"}
    try:
        print("Step 1: Detecting colorbars with YOLO (best0710.pt)...")
        model = load_yolo_model(model_path)
        opencv_image = np.array(pil_image.convert("RGB"))
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

        # 重新获取完整的YOLO输出
        (annotated_image, colorbar_boxes, confidences, colorbar_segments) = detect_colorbars_yolo(
            opencv_image, model, box_expansion=box_expansion, confidence_threshold=confidence_threshold,
        )
        if len(colorbar_segments) == 0:
            return {
                "error": "No colorbars detected",
                "annotated_image": Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)),
                "step_completed": 1,
            }
        print("Step 2: Initializing YOLO block detector (best.pt)...")
        try:
            block_model = load_yolo_block_model()
        except (FileNotFoundError, RuntimeError) as e:
            return {"error": str(e)}
        
        colorbar_results = []
        # 使用完整的YOLO输出进行迭代
        for i, (segment, confidence, box) in enumerate(zip(colorbar_segments, confidences, colorbar_boxes, strict=False)):
            colorbar_id = i + 1
            print(f"  Processing colorbar {colorbar_id}/{len(colorbar_segments)}...")
            (segmented_colorbar, color_blocks, block_count) = detect_blocks_with_yolo(
                segment, block_model, confidence_threshold=yolo_block_confidence, min_area=block_area_threshold,
            )
            print(f"  Analyzing {block_count} pure colors in colorbar {colorbar_id}...")
            pure_color_analyses, best_match_card_id = [], None
            if block_count > 0:
                (pure_color_analyses, best_match_card_id) = analyze_colorbar_with_best_card_match(
                    color_blocks, colorbar_id=colorbar_id, purity_threshold=purity_threshold,
                )
            colorbar_result = {
                "colorbar_id": colorbar_id, "confidence": confidence, "bounding_box": box,
                "original_segment_pil": Image.fromarray(cv2.cvtColor(segment, cv2.COLOR_BGR2RGB)),
                "segmented_colorbar_pil": Image.fromarray(cv2.cvtColor(segmented_colorbar, cv2.COLOR_BGR2RGB)),
                "color_blocks": color_blocks, "block_count": block_count,
                "pure_color_analyses": pure_color_analyses, "best_match_card_id": best_match_card_id,
            }
            colorbar_results.append(colorbar_result)
            
        total_blocks = sum(result["block_count"] for result in colorbar_results)
        all_delta_e_values = []
        excellent_count, acceptable_count, high_purity_count = 0, 0, 0
        for result in colorbar_results:
            for analysis in result["pure_color_analyses"]:
                if "error" not in analysis and "ground_truth_match" in analysis:
                    gt_match = analysis["ground_truth_match"]
                    delta_e = gt_match["delta_e"]
                    all_delta_e_values.append(delta_e)
                    if gt_match.get("is_excellent", False): excellent_count += 1
                    if gt_match.get("is_acceptable", False): acceptable_count += 1
                    if analysis["purity_score"] >= 0.8: high_purity_count += 1
        
        accuracy_stats = {}
        if all_delta_e_values:
            import statistics
            total_analyzed = len(all_delta_e_values)
            accuracy_stats = {
                "average_delta_e": statistics.mean(all_delta_e_values),
                "median_delta_e": statistics.median(all_delta_e_values),
                "max_delta_e": max(all_delta_e_values), "min_delta_e": min(all_delta_e_values),
                "excellent_colors": excellent_count, "acceptable_colors": acceptable_count,
                "high_purity_colors": high_purity_count, "total_analyzed": total_analyzed,
                "excellent_percentage": (excellent_count / total_analyzed) * 100 if total_analyzed > 0 else 0,
                "acceptable_percentage": (acceptable_count / total_analyzed) * 100 if total_analyzed > 0 else 0,
                "high_purity_percentage": (high_purity_count / total_analyzed) * 100 if total_analyzed > 0 else 0,
            }
        
        return {
            "success": True, "analysis_type": "direct_yolo_block_detection",
            "annotated_image": annotated_image, "colorbar_count": len(colorbar_results),
            "colorbar_results": colorbar_results, "total_blocks": total_blocks,
            "accuracy_statistics": accuracy_stats, "step_completed": 3,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": f"Error in pure colorbar analysis pipeline: {str(e)}", "step_completed": 0}


def pure_colorbar_analysis_for_gradio(
    pil_image: Image.Image,
    confidence_threshold: float = 0.5,
    box_expansion: int = 10,
    yolo_block_confidence: float = 0.5,
    block_area_threshold: int = 50,
    purity_threshold: float = 0.8,
    **kwargs,
) -> tuple[Image.Image, list[dict], str, int]:
    """
    为Gradio界面优化的纯色色板分析流水线包装器。
    """
    result = pure_colorbar_analysis_pipeline(
        pil_image, confidence_threshold=confidence_threshold, box_expansion=box_expansion,
        yolo_block_confidence=yolo_block_confidence, block_area_threshold=block_area_threshold,
        purity_threshold=purity_threshold,
    )
    if "error" in result:
        return (pil_image if pil_image else None), [], f"❌ {result['error']}", 0
    if not result.get("success", False):
        return pil_image, [], "❌ Pure color analysis failed", 0

    annotated_pil = Image.fromarray(cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB))
    colorbar_data = result.get("colorbar_results", [])

    report = "🎯 YOLO Direct Block Analysis Results\n" + "=" * 55 + "\n\n"
    for i, res in enumerate(colorbar_data):
        best_card_id = res.get("best_match_card_id")
        block_count = res.get("block_count", 0)
        if best_card_id == "INVALID_DETECTION":
            report += f"🎨 Colorbar #{i+1} - ⚠️ ERROR: Too many blocks detected ({block_count} > 7). Cannot perform matching.\n"
        elif best_card_id:
            report += f"🎨 Colorbar #{i+1} - Best Match Card: {best_card_id.upper()}\n"
        else:
            # This case might happen if 0 blocks were detected in a colorbar segment
            report += f"🎨 Colorbar #{i+1} - No blocks detected or no match found.\n"


    report += "\n📊 Overall Summary:\n"
    report += f"  • Total color blocks found: {result['total_blocks']}\n"

    stats = result.get("accuracy_statistics", {})
    if stats:
        report += f"  • Average ΔE (against best cards): {stats.get('average_delta_e', 0):.2f}\n"
        report += f"  • ΔE Range: {stats.get('min_delta_e', 0):.2f} - {stats.get('max_delta_e', 0):.2f}\n"
        report += f"  • Excellent colors (ΔE < 1.0): {stats.get('excellent_colors', 0)}/{stats.get('total_analyzed', 0)} ({stats.get('excellent_percentage', 0):.1f}%)\n"
        report += f"  • Acceptable colors (ΔE < 3.0): {stats.get('acceptable_colors', 0)}/{stats.get('total_analyzed', 0)} ({stats.get('acceptable_percentage', 0):.1f}%)\n"
        report += f"  • High purity colors: {stats.get('high_purity_colors', 0)}/{stats.get('total_analyzed', 0)} ({stats.get('high_purity_percentage', 0):.1f}%)\n"
    report += "\n"

    for res in colorbar_data:
        best_card_id = res.get("best_match_card_id", "N/A")
        report += f"🔎 Details for Colorbar (Matched to {best_card_id}):\n"
        if best_card_id == "INVALID_DETECTION":
            report += "    - Skipping block details due to invalid detection.\n\n"
            continue
        if res["block_count"] > 0:
            for analysis in res["pure_color_analyses"]:
                if "error" in analysis: continue
                block_id = analysis.get("block_id", "?")
                rgb = analysis.get("pure_color_rgb", (0,0,0))
                cmyk = analysis.get("pure_color_cmyk", (0,0,0,0))
                purity = analysis.get("purity_score", 0)
                quality = analysis.get("color_quality", "N/A")
                hex_code = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
                report += f"    - Block {block_id}: {hex_code} (C{cmyk[0]} M{cmyk[1]} Y{cmyk[2]} K{cmyk[3]})"
                report += f" | Purity: {purity:.2f} ({quality})"
                if "ground_truth_match" in analysis:
                    gt = analysis["ground_truth_match"]
                    if gt and gt.get("closest_color"):
                        delta_e, level, gt_color = gt["delta_e"], gt["accuracy_level"], gt["closest_color"]
                        report += f" | ΔE: {delta_e:.2f} ({level}) vs {gt_color['name']}"
                        if gt.get("is_excellent", False): report += " ✅"
                        elif gt.get("is_acceptable", False): report += " ⚠️"
                        else: report += " ❌"
                report += "\n"
        report += "\n"
    # The return signature for gradio needs image, data, string, int
    return (annotated_pil, colorbar_data, report, result["total_blocks"])