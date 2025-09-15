"""
Intelligent Colorbar Analysis Pipeline
Advanced three-step workflow for colorbar detection and color analysis:
1. YOLO detects colorbar regions
2. Block detection within each colorbar
3. Individual color block analysis with CMYK conversion and delta E calculation

Features:
- Includes original segmented colorbar images
- Dense, organized result display
- CMYK color conversion with ICC profiles
- Ground-truth color comparison with delta E calculations
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
    Apply block detection specifically to a colorbar segment to extract individual color patches.

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

    # Use the existing block detection function with adjusted parameters for colorbar
    result_image, block_images, block_count = detect_blocks(
        colorbar_segment,
        output_dir=None,  # Don't save files
        area_threshold=area_threshold,
        aspect_ratio_threshold=aspect_ratio_threshold,
        min_square_size=min_square_size,
        return_individual_blocks=True,
    )

    return result_image, block_images, block_count


def enhance_with_ground_truth_comparison(block_analyses: list[dict]) -> list[dict]:
    """
    Enhance block analyses with ground truth color comparison and delta E calculations.

    Args:
        block_analyses: List of block analysis dictionaries

    Returns:
        Enhanced block analyses with ground truth comparison data
    """
    enhanced_analyses = []

    for analysis in block_analyses:
        # Skip if there's an error in the analysis
        if "error" in analysis:
            enhanced_analyses.append(analysis)
            continue

        # Extract primary RGB color
        primary_rgb = analysis.get("primary_color_rgb")
        if not primary_rgb:
            enhanced_analyses.append(analysis)
            continue

        # Find closest ground truth color and calculate delta E
        closest_gt_color, delta_e = ground_truth_checker.find_closest_color(primary_rgb)

        # Enhance the analysis with ground truth comparison
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
                    < 3.0,  # Delta E < 3 is the threshold for acceptable colors
                }
            }
        )

        enhanced_analyses.append(enhanced_analysis)

    return enhanced_analyses


def colorbar_analysis_pipeline(
    pil_image: Image.Image,
    # YOLO parameters
    confidence_threshold: float = 0.5,
    box_expansion: int = 10,
    model_path: str = None,
    # Block detection parameters
    block_area_threshold: int = 50,
    block_aspect_ratio: float = 0.3,
    min_square_size: int = 5,
    # Color analysis parameters
    shrink_size: tuple[int, int] = (30, 30),
) -> dict:
    """
    Complete intelligent colorbar analysis pipeline.

    Args:
        pil_image: Input PIL image
        confidence_threshold: YOLO confidence threshold
        box_expansion: YOLO box expansion pixels
        model_path: Path to YOLO model
        block_area_threshold: Minimum area for blocks within colorbar
        block_aspect_ratio: Minimum aspect ratio for blocks
        min_square_size: Minimum width and height for detected blocks (pixels)
        shrink_size: Size to shrink blocks for color analysis

    Returns:
        Dictionary with complete analysis results including original colorbar segments
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

        # Step 2: Block detection within each colorbar
        print(
            f"Step 2: Detecting blocks within {len(colorbar_segments)} colorbar(s)..."
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

            # Step 3: Analyze colors in each block
            print(
                f"  Step 3: Analyzing {block_count} color blocks in colorbar {colorbar_id}..."
            )
            block_analyses = []

            if block_count > 0:
                block_analyses = analyze_colorbar_blocks(
                    color_blocks, shrink_size, colorbar_id=colorbar_id
                )

                # Step 4: Calculate delta E against ground truth colors
                print("  Step 4: Calculating delta E against ground truth colors...")
                block_analyses = enhance_with_ground_truth_comparison(block_analyses)

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
    Colorbar analysis pipeline wrapper optimized for Gradio interface.

    Returns:
        Tuple of (
            annotated_image,
            colorbar_data_with_original_segments,
            analysis_report,
            total_blocks_found
        )
    """
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

    # Convert annotated image to PIL
    annotated_pil = Image.fromarray(
        cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB)
    )

    # Prepare colorbar data with original segments included
    colorbar_data = []

    # Build comprehensive analysis report with delta E statistics
    report = "🎯 Intelligent Colorbar Analysis Results\n"
    report += "=" * 50 + "\n\n"
    report += "📊 Summary:\n"
    report += f"  • Colorbars detected: {result['colorbar_count']}\n"
    report += f"  • Total color blocks: {result['total_blocks']}\n"

    # Calculate delta E statistics
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

        # Create colorbar data entry with original segments
        colorbar_entry = {
            "colorbar_id": colorbar_id,
            "confidence": confidence,
            "original_colorbar": colorbar_result[
                "original_segment_pil"
            ],  # ✨ ORIGINAL SEGMENT
            "segmented_colorbar": colorbar_result["segmented_colorbar_pil"],
            "color_blocks": color_blocks_pil,
            "block_count": block_count,
            "block_analyses": colorbar_result["block_analyses"],
        }
        colorbar_data.append(colorbar_entry)

        # Add dense colorbar section to report
        report += f"🎨 Colorbar {colorbar_id} (confidence: {confidence:.2f}):\n"
        report += f"  • Color blocks found: {block_count}\n"

        if block_count > 0:
            report += "  • Colors with CMYK values and delta E:\n"

            # Add concise color analysis for each block
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

                    # Add delta E information if available
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


# Backward compatibility aliases
enhanced_colorbar_analysis = colorbar_analysis_pipeline
enhanced_colorbar_analysis_from_pil = colorbar_analysis_for_gradio
