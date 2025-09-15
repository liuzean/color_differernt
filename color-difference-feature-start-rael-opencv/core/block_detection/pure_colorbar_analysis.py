"""
Redesigned Pure Color-Based Colorbar Analysis Pipeline

This module provides a colorbar analysis system that focuses specifically on
pure color matching against ground truth values, emphasizing:
- Pure dominant color extraction (not averages)
- Direct ground truth color matching with CMYK values
- Clear delta E reporting and accuracy assessment
- Simplified, focused workflow for color accuracy testing
"""

import cv2
import numpy as np
from PIL import Image

from ..color.ground_truth_checker import ground_truth_checker
from .blocks_detect import detect_blocks
from .yolo_show import detect_colorbars_yolo, load_yolo_model


def extract_pure_color_from_block(
    color_block: np.ndarray,
    purity_threshold: float = 0.8,
    sample_size: tuple[int, int] = (20, 20),
) -> tuple[tuple[int, int, int], float]:
    """
    Extract the most pure/dominant color from a color block.

    This function focuses on finding the most representative single color
    rather than computing averages, which is key for pure color analysis.

    Args:
        color_block: Individual color block image (BGR format)
        purity_threshold: Minimum purity level (0-1) for color acceptance
        sample_size: Size to sample for analysis

    Returns:
        Tuple of (RGB_color, purity_score)
        purity_score: How "pure" the color is (1.0 = perfectly uniform)
    """
    if color_block.size == 0:
        return (0, 0, 0), 0.0

    # Resize for consistent analysis
    if color_block.shape[0] > sample_size[0] or color_block.shape[1] > sample_size[1]:
        resized = cv2.resize(color_block, sample_size, interpolation=cv2.INTER_AREA)
    else:
        resized = color_block

    # Convert to RGB
    rgb_block = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Extract center region for pure color sampling
    h, w = rgb_block.shape[:2]
    center_h, center_w = h // 2, w // 2

    # Sample from center 50% of the block to avoid edge artifacts
    margin_h, margin_w = h // 4, w // 4
    center_region = rgb_block[
        center_h - margin_h : center_h + margin_h,
        center_w - margin_w : center_w + margin_w,
    ]

    if center_region.size == 0:
        center_region = rgb_block

    # Calculate color statistics for purity assessment
    pixels = center_region.reshape(-1, 3)

    if len(pixels) == 0:
        return (0, 0, 0), 0.0

    # Find the most common color using median (more robust than mean)
    median_color = np.median(pixels, axis=0).astype(int)

    # Calculate color purity (uniformity)
    # Lower standard deviation = higher purity
    color_std = np.std(pixels, axis=0)
    max_std = np.max(color_std)

    # Purity score: inverse of color variation
    # max_std of 0 = perfect purity (1.0)
    # max_std of 50+ = low purity (approaching 0)
    purity_score = max(0.0, 1.0 - (max_std / 50.0))

    # Use the most frequent color (mode) for better pure color detection
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    most_frequent_idx = np.argmax(counts)
    dominant_color = unique_colors[most_frequent_idx]

    # If the color is pure enough, use the dominant color, otherwise use median
    if purity_score >= purity_threshold:
        pure_color = tuple(dominant_color)
    else:
        pure_color = tuple(median_color)

    return pure_color, purity_score


def analyze_pure_color_block(
    color_block: np.ndarray,
    block_id: int = None,
    colorbar_id: int = None,
) -> dict:
    """
    Analyze a single color block focusing on pure color extraction and ground truth matching.

    Args:
        color_block: Individual color block image (BGR format)
        block_id: ID of this block
        colorbar_id: ID of parent colorbar

    Returns:
        Dictionary with pure color analysis and ground truth comparison
    """
    if color_block.size == 0:
        return {"error": "Empty color block"}

    # Extract pure color
    pure_rgb, purity_score = extract_pure_color_from_block(color_block)

    # Convert pure color to CMYK using the existing color system
    from .color_analysis import rgb_to_cmyk_icc

    pure_cmyk = rgb_to_cmyk_icc(pure_rgb)

    # Find closest ground truth color and calculate delta E
    closest_gt_color, delta_e = ground_truth_checker.find_closest_color(pure_rgb)

    # Create analysis result
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
            "is_acceptable": delta_e < 3.0,
            "is_excellent": delta_e < 1.0,
        },
        "block_size": color_block.shape[:2],
    }

    return analysis


def _get_color_quality(purity_score: float) -> str:
    """
    Get color quality description based on purity score.

    Args:
        purity_score: Color purity score (0-1)

    Returns:
        Quality level string
    """
    if purity_score >= 0.9:
        return "Excellent"
    elif purity_score >= 0.8:
        return "Very Good"
    elif purity_score >= 0.7:
        return "Good"
    elif purity_score >= 0.6:
        return "Fair"
    elif purity_score >= 0.5:
        return "Poor"
    else:
        return "Very Poor"


def analyze_colorbar_pure_colors(
    colorbar_blocks: list[np.ndarray],
    colorbar_id: int = None,
) -> list[dict]:
    """
    Analyze multiple color blocks from a colorbar with pure color focus.

    Args:
        colorbar_blocks: List of individual color block images (BGR format)
        colorbar_id: ID of the parent colorbar

    Returns:
        List of pure color analysis dictionaries
    """
    analyses = []

    for i, block in enumerate(colorbar_blocks):
        analysis = analyze_pure_color_block(
            block, block_id=i + 1, colorbar_id=colorbar_id
        )
        analyses.append(analysis)

    return analyses


def pure_colorbar_analysis_pipeline(
    pil_image: Image.Image,
    # YOLO parameters
    confidence_threshold: float = 0.5,
    box_expansion: int = 10,
    model_path: str = None,
    # Block detection parameters
    block_area_threshold: int = 50,
    block_aspect_ratio: float = 0.3,
    min_square_size: int = 5,
    # Pure color analysis parameters
    purity_threshold: float = 0.8,
) -> dict:
    """
    Complete pure color-based colorbar analysis pipeline.

    This pipeline focuses on:
    1. Detecting colorbars using YOLO
    2. Extracting individual color blocks
    3. Finding pure/dominant colors (not averages)
    4. Matching against ground truth with precise CMYK and delta E reporting

    Args:
        pil_image: Input PIL image
        confidence_threshold: YOLO confidence threshold
        box_expansion: YOLO box expansion pixels
        model_path: Path to YOLO model
        block_area_threshold: Minimum area for blocks within colorbar
        block_aspect_ratio: Minimum aspect ratio for blocks
        min_square_size: Minimum width and height for detected blocks (pixels)
        purity_threshold: Minimum purity score for color acceptance

    Returns:
        Dictionary with complete pure color analysis results
    """
    if pil_image is None:
        return {"error": "No image provided"}

    try:
        # Step 1: YOLO colorbar detection
        print("Step 1: Detecting colorbars with YOLO...")
        model = load_yolo_model(model_path)

        # Convert PIL to OpenCV
        opencv_image = np.array(pil_image)
        if len(opencv_image.shape) == 3:
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)

        # Run YOLO detection
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

        # Step 2: Block detection and pure color analysis
        print(
            f"Step 2: Analyzing pure colors in {len(colorbar_segments)} colorbar(s)..."
        )
        colorbar_results = []

        for i, (segment, confidence, box) in enumerate(
            zip(colorbar_segments, confidences, colorbar_boxes, strict=False)
        ):
            colorbar_id = i + 1
            print(f"  Processing colorbar {colorbar_id}/{len(colorbar_segments)}...")

            # Extract blocks from this colorbar
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

            # Step 3: Pure color analysis for each block
            print(f"  Analyzing {block_count} pure colors in colorbar {colorbar_id}...")
            pure_color_analyses = []

            if block_count > 0:
                pure_color_analyses = analyze_colorbar_pure_colors(
                    color_blocks, colorbar_id=colorbar_id
                )

            # Convert segments to PIL for better interface integration
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
                "original_segment": segment,  # Original OpenCV format
                "original_segment_pil": original_segment_pil,  # PIL format for display
                "segmented_colorbar": segmented_colorbar,  # OpenCV format
                "segmented_colorbar_pil": segmented_colorbar_pil,  # PIL format for display
                "color_blocks": color_blocks,
                "block_count": block_count,
                "pure_color_analyses": pure_color_analyses,
            }

            colorbar_results.append(colorbar_result)

        # Calculate overall statistics
        total_blocks = sum(result["block_count"] for result in colorbar_results)
        all_delta_e_values = []
        excellent_count = 0
        acceptable_count = 0
        high_purity_count = 0

        for colorbar_result in colorbar_results:
            for analysis in colorbar_result["pure_color_analyses"]:
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

        # Calculate accuracy statistics
        accuracy_stats = {}
        if all_delta_e_values:
            import statistics

            accuracy_stats = {
                "average_delta_e": statistics.mean(all_delta_e_values),
                "median_delta_e": statistics.median(all_delta_e_values),
                "max_delta_e": max(all_delta_e_values),
                "min_delta_e": min(all_delta_e_values),
                "excellent_colors": excellent_count,
                "acceptable_colors": acceptable_count,
                "high_purity_colors": high_purity_count,
                "total_analyzed": len(all_delta_e_values),
                "excellent_percentage": (excellent_count / len(all_delta_e_values))
                * 100,
                "acceptable_percentage": (acceptable_count / len(all_delta_e_values))
                * 100,
                "high_purity_percentage": (high_purity_count / len(all_delta_e_values))
                * 100,
            }

        return {
            "success": True,
            "analysis_type": "pure_color_based",
            "annotated_image": annotated_image,
            "colorbar_count": len(colorbar_segments),
            "colorbar_results": colorbar_results,
            "total_blocks": total_blocks,
            "accuracy_statistics": accuracy_stats,
            "step_completed": 3,
        }

    except Exception as e:
        return {
            "error": f"Error in pure colorbar analysis pipeline: {str(e)}",
            "step_completed": 0,
        }


def extract_blocks_from_colorbar(
    colorbar_segment: np.ndarray,
    area_threshold: int = 50,
    aspect_ratio_threshold: float = 0.3,
    min_square_size: int = 5,
) -> tuple[np.ndarray, list[np.ndarray], int]:
    """
    Apply block detection specifically to a colorbar segment to extract individual color patches.

    This is a wrapper around the existing block detection function.

    Args:
        colorbar_segment: Colorbar image segment (BGR format)
        area_threshold: Minimum area for detected blocks
        aspect_ratio_threshold: Minimum aspect ratio for blocks
        min_square_size: Minimum width and height for detected blocks (pixels)

    Returns:
        Tuple of (annotated_colorbar, list_of_color_blocks, block_count)
    """
    if colorbar_segment.size == 0:
        return colorbar_segment, [], 0

    # Use the existing block detection function
    result_image, block_images, block_count = detect_blocks(
        colorbar_segment,
        output_dir=None,  # Don't save files
        area_threshold=area_threshold,
        aspect_ratio_threshold=aspect_ratio_threshold,
        min_square_size=min_square_size,
        return_individual_blocks=True,
    )

    return result_image, block_images, block_count


def pure_colorbar_analysis_for_gradio(
    pil_image: Image.Image,
    confidence_threshold: float = 0.5,
    box_expansion: int = 10,
    block_area_threshold: int = 50,
    block_aspect_ratio: float = 0.3,
    min_square_size: int = 5,
    purity_threshold: float = 0.8,
) -> tuple[Image.Image, list[dict], str, int]:
    """
    Pure colorbar analysis pipeline wrapper optimized for Gradio interface.

    Returns:
        Tuple of (
            annotated_image,
            colorbar_data_with_pure_colors,
            analysis_report,
            total_blocks_found
        )
    """
    result = pure_colorbar_analysis_pipeline(
        pil_image,
        confidence_threshold=confidence_threshold,
        box_expansion=box_expansion,
        block_area_threshold=block_area_threshold,
        block_aspect_ratio=block_aspect_ratio,
        min_square_size=min_square_size,
        purity_threshold=purity_threshold,
    )

    if "error" in result:
        error_img = pil_image if pil_image else None
        return error_img, [], f"❌ {result['error']}", 0

    if not result.get("success", False):
        return pil_image, [], "❌ Pure color analysis failed", 0

    # Convert annotated image to PIL
    annotated_pil = Image.fromarray(
        cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB)
    )

    # Prepare colorbar data with pure color analysis
    colorbar_data = []

    # Build comprehensive analysis report
    report = "🎯 Pure Color-Based Colorbar Analysis Results\n"
    report += "=" * 55 + "\n\n"
    report += "📊 Summary:\n"
    report += f"  • Colorbars detected: {result['colorbar_count']}\n"
    report += f"  • Total color blocks: {result['total_blocks']}\n"

    # Add accuracy statistics
    accuracy_stats = result.get("accuracy_statistics", {})
    if accuracy_stats:
        report += f"  • Average ΔE: {accuracy_stats['average_delta_e']:.2f}\n"
        report += f"  • ΔE Range: {accuracy_stats['min_delta_e']:.2f} - {accuracy_stats['max_delta_e']:.2f}\n"
        report += f"  • Excellent colors (ΔE < 1.0): {accuracy_stats['excellent_colors']}/{accuracy_stats['total_analyzed']} ({accuracy_stats['excellent_percentage']:.1f}%)\n"
        report += f"  • Acceptable colors (ΔE < 3.0): {accuracy_stats['acceptable_colors']}/{accuracy_stats['total_analyzed']} ({accuracy_stats['acceptable_percentage']:.1f}%)\n"
        report += f"  • High purity colors: {accuracy_stats['high_purity_colors']}/{accuracy_stats['total_analyzed']} ({accuracy_stats['high_purity_percentage']:.1f}%)\n"

    report += "\n"

    for colorbar_result in result["colorbar_results"]:
        colorbar_id = colorbar_result["colorbar_id"]
        confidence = colorbar_result["confidence"]
        block_count = colorbar_result["block_count"]

        # Convert color blocks to PIL
        color_blocks_pil = []
        for block in colorbar_result["color_blocks"]:
            if block.size > 0:
                block_pil = Image.fromarray(cv2.cvtColor(block, cv2.COLOR_BGR2RGB))
                color_blocks_pil.append(block_pil)

        # Create colorbar data entry
        colorbar_entry = {
            "colorbar_id": colorbar_id,
            "confidence": confidence,
            "original_colorbar": colorbar_result["original_segment_pil"],
            "segmented_colorbar": colorbar_result["segmented_colorbar_pil"],
            "color_blocks": color_blocks_pil,
            "block_count": block_count,
            "pure_color_analyses": colorbar_result["pure_color_analyses"],
        }
        colorbar_data.append(colorbar_entry)

        # Add detailed colorbar section to report
        report += f"🎨 Colorbar {colorbar_id} (confidence: {confidence:.2f}):\n"
        report += f"  • Pure color blocks found: {block_count}\n"

        if block_count > 0:
            report += "  • Pure colors with CMYK values and delta E:\n"

            # Add detailed pure color analysis for each block
            for analysis in colorbar_result["pure_color_analyses"]:
                if "error" not in analysis:
                    block_id = analysis.get("block_id", "?")
                    pure_rgb = analysis["pure_color_rgb"]
                    pure_cmyk = analysis["pure_color_cmyk"]
                    purity_score = analysis["purity_score"]
                    color_quality = analysis["color_quality"]
                    pure_hex = f"#{pure_rgb[0]:02x}{pure_rgb[1]:02x}{pure_rgb[2]:02x}"

                    report += f"    {colorbar_id}.{block_id}: {pure_hex} "
                    report += f"(C={pure_cmyk[0]}% M={pure_cmyk[1]}% Y={pure_cmyk[2]}% K={pure_cmyk[3]}%)"

                    # Add purity information
                    report += f" | Purity: {purity_score:.2f} ({color_quality})"

                    # Add ground truth comparison
                    gt_match = analysis["ground_truth_match"]
                    if gt_match["closest_color"]:
                        delta_e = gt_match["delta_e"]
                        accuracy_level = gt_match["accuracy_level"]
                        gt_color = gt_match["closest_color"]

                        report += f" | ΔE: {delta_e:.2f} ({accuracy_level})"
                        report += f" vs {gt_color['name']}"

                        # Add status indicator
                        if gt_match["is_excellent"]:
                            report += " ✅"
                        elif gt_match["is_acceptable"]:
                            report += " ⚠️"
                        else:
                            report += " ❌"

                    report += "\n"

        report += "\n"

    return (annotated_pil, colorbar_data, report, result["total_blocks"])
