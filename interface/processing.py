import logging
import os
import time

import cv2
import numpy as np

import core.image.alignment as alignment
from core.color.icc_trans import convert_color_space_array, get_available_icc_profiles
from core.color.utils import analyze_color_statistics, calculate_color_difference
from core.image.features.detector import DetectorFactory
from core.image.features.matcher import MatcherFactory
from core.image.masking import MaskingProcessor
from core.image.preprocessor import ImagePreprocessor
from visualization.blockwise import (
    analyze_blocks,
    overlay_block_boundaries,
    visualize_block_heatmap,
)
from visualization.charts import (
    create_composite_analysis,
    histogram_delta_e,
    stats_box_chart,
)
from visualization.covermap import calculate_difference_map
from visualization.heatmap import (
    create_heatmap_with_colorbar,
    generate_heatmap,
    highlight_regions,
    overlay_heatmap,
)


# Image processing and color difference analysis function
def process_images(template_path, target_path, params):
    """
    Process two images for alignment and color difference analysis

    Args:
        template_path: Template image path
        target_path: Target image path
        params: Processing parameters dictionary

    Returns:
        Processing results and visualization images
    """
    logger = logging.getLogger("process_images")

    logger.info(
        f"Starting image processing: template={template_path}, target={target_path}"
    )

    output_dir = params["visualization"]["output_dir"]
    # Ensure output_dir is a string
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load images
    logger.info(f"Loading template image: {template_path}")
    template_img = cv2.imread(template_path)

    logger.info(f"Loading target image: {target_path}")
    target_img = cv2.imread(target_path)

    # Check if images loaded successfully
    if template_img is None:
        logger.error(f"Failed to load template image: {template_path}")
        return None, f"Failed to load template image: {template_path}"
    if target_img is None:
        logger.error(f"Failed to load target image: {target_path}")
        return None, f"Failed to load target image: {target_path}"

    logger.info(
        f"Template image size: {template_img.shape}, Target image size: {target_img.shape}"
    )

    # Extract alpha channel mask
    logger.info("Attempting to extract alpha channel mask")
    alpha_mask = MaskingProcessor.extract_alpha_mask(template_path, None)

    if alpha_mask is not None:
        logger.info(
            f"Successfully extracted alpha channel mask, size: {alpha_mask.shape}"
        )
    else:
        logger.warning(
            "Unable to extract alpha channel mask, will continue processing without background removal"
        )

    # Preprocess images
    preprocessor = ImagePreprocessor()
    preprocessor.max_dimension = params["preprocessing"]["resize_max_dimension"]
    preprocessor.enhance_contrast = params["preprocessing"]["enhance_contrast"]
    preprocessor.clahe_clip_limit = params["preprocessing"]["clahe_clip_limit"]
    preprocessor.clahe_grid_size = tuple(params["preprocessing"]["clahe_grid_size"])

    template_color = template_img
    target_color = preprocessor.resize_image(target_img)

    # Determine detector types to use
    compare_all = params["visualization"].get("compare_all_methods", False)
    detector_types = (
        ["surf", "sift", "orb"] if compare_all else [params["detector"]["type"]]
    )

    results_by_method = {}

    for detector_type in detector_types:
        logger.info(f"Processing images with {detector_type} algorithm")
        # Create feature detector and matcher
        detector = DetectorFactory.create(
            detector_type, nfeatures=params["detector"]["nfeatures"]
        )
        matcher = MatcherFactory.create_for_detector(
            detector_type, crossCheck=params["matcher"]["crossCheck"]
        )

        # Create image aligner
        aligner = alignment.ImageAlignment(detector=detector, matcher=matcher)

        # Perform image alignment
        logger.info("Executing image alignment")
        result = aligner.align_images(
            template_color,
            target_color,
            ratio_test=params["matcher"]["ratio_test"],
            ratio_threshold=params["matcher"]["ratio_threshold"],
            ransac_reproj_threshold=params["alignment"]["ransac_reproj_threshold"],
            min_matches=params["alignment"]["min_matches"],
        )

        # Remove background if needed
        if params["color_difference"]["mask_background"] and alpha_mask is not None:
            logger.info("Removing background")
            try:
                # Only remove background if alignment succeeded and image is not None
                if result.success and result.aligned_image is not None:
                    result.aligned_image = MaskingProcessor.remove_background(
                        result.aligned_image, alpha_mask
                    )
                if result.success and result.roi_aligned is not None:
                    result.roi_aligned = MaskingProcessor.remove_background(
                        result.roi_aligned, alpha_mask
                    )
                logger.info("Background removal successful")
            except Exception as e:
                logger.error(f"Error removing background: {str(e)}")
                import traceback

                logger.error(traceback.format_exc())
        else:
            logger.info("Skipping background removal")

        if not result.success:
            if compare_all:
                logger.warning(
                    f"{detector_type} algorithm processing failed: {result.message}"
                )
                continue
            else:
                logger.error(
                    f"{detector_type} algorithm processing failed: {result.message}"
                )
                return None, result.message

        # If refinement alignment is enabled
        if params["alignment"]["refinement"]["enabled"] and result.success:
            # Create detector and matcher for fine alignment
            refined_detector = DetectorFactory.create(
                detector_type, nfeatures=params["alignment"]["refinement"]["nfeatures"]
            )
            refined_matcher = MatcherFactory.create_for_detector(
                detector_type,
                crossCheck=params["alignment"]["refinement"]["crossCheck"],
            )
            refined_aligner = alignment.ImageAlignment(
                detector=refined_detector, matcher=refined_matcher
            )

            # Perform refinement alignment
            refined_result = refined_aligner.align_images(
                result.roi_template,
                result.roi_aligned,
                ratio_test=params["matcher"]["ratio_test"],
                ratio_threshold=params["alignment"]["refinement"]["ratio_threshold"],
                ransac_reproj_threshold=params["alignment"]["refinement"][
                    "ransac_reproj_threshold"
                ],
                min_matches=params["alignment"]["refinement"]["min_matches"],
            )

            if refined_result.success:
                result = refined_result

        # Calculate color difference (using CIEDE2000)
        avg_delta_e, delta_e_map = calculate_color_difference(
            result.roi_template, result.roi_aligned
        )

        # Calculate color difference statistics
        stats = analyze_color_statistics(delta_e_map)

        # Generate visualization results
        threshold = params["color_difference"]["threshold"]

        # Difference map
        diff_map = calculate_difference_map(result.roi_template, result.roi_aligned)

        # Color difference heatmap
        heatmap = generate_heatmap(delta_e_map)

        # Heatmap with colorbar
        heatmap_colorbar = create_heatmap_with_colorbar(
            delta_e_map, title=f"{detector_type.upper()} Color Difference Heatmap"
        )

        # Overlayed heatmap
        overlayed_heatmap = overlay_heatmap(result.roi_template, heatmap)

        # High color difference regions
        highlighted = highlight_regions(delta_e_map, threshold, result.roi_template)

        # Color difference histogram
        histogram = histogram_delta_e(
            delta_e_map,
            title=f"{detector_type.upper()} Color Difference Distribution",
            threshold=threshold,
        )

        # Statistics chart
        stats_chart = stats_box_chart(
            stats, title=f"{detector_type.upper()} Color Difference Statistics"
        )

        # Composite analysis
        composite = create_composite_analysis(
            delta_e_map,
            stats,
            image=result.roi_template,
            title=f"{detector_type.upper()} Color Difference Analysis",
        )

        # Save results
        if params["visualization"]["save_intermediate"]:
            prefix = f"{detector_type}_" if compare_all else ""
            cv2.imwrite(
                os.path.join(output_dir, f"{prefix}aligned.jpg"), result.aligned_image
            )
            cv2.imwrite(os.path.join(output_dir, f"{prefix}diff_map.jpg"), diff_map)
            cv2.imwrite(os.path.join(output_dir, f"{prefix}heatmap.jpg"), heatmap)
            cv2.imwrite(
                os.path.join(output_dir, f"{prefix}heatmap_colorbar.jpg"),
                heatmap_colorbar,
            )
            cv2.imwrite(
                os.path.join(output_dir, f"{prefix}overlayed_heatmap.jpg"),
                overlayed_heatmap,
            )
            cv2.imwrite(
                os.path.join(output_dir, f"{prefix}highlighted.jpg"), highlighted
            )
            cv2.imwrite(os.path.join(output_dir, f"{prefix}histogram.jpg"), histogram)
            cv2.imwrite(
                os.path.join(output_dir, f"{prefix}stats_chart.jpg"), stats_chart
            )
            cv2.imwrite(os.path.join(output_dir, f"{prefix}composite.jpg"), composite)

            # Blockwise analysis
            if params.get("blockwise", {}).get("enabled", True):
                block_size_h = params.get("blockwise", {}).get("block_size_h", 32)
                block_size_w = params.get("blockwise", {}).get("block_size_w", 32)
                use_mask = params.get("blockwise", {}).get("use_mask", True)

                # Create block heatmap
                mask_for_block = (
                    alpha_mask if use_mask and alpha_mask is not None else None
                )
                block_heatmap = visualize_block_heatmap(
                    delta_e_map,
                    block_size=(block_size_h, block_size_w),
                    mask=mask_for_block,
                    title=f"{detector_type.upper()} Block Color Difference Heatmap",
                )

                # Calculate block statistics
                blocks_info = analyze_blocks(
                    delta_e_map,
                    block_size=(block_size_h, block_size_w),
                    mask=mask_for_block,
                )

                # Overlay block boundaries on original image
                overlay_blocks = overlay_block_boundaries(
                    result.roi_template,
                    blocks_info,
                    threshold=params["color_difference"]["threshold"] * 0.8,
                )

                # Save blockwise analysis results
                cv2.imwrite(
                    os.path.join(output_dir, f"{prefix}block_heatmap.jpg"),
                    block_heatmap,
                )
                cv2.imwrite(
                    os.path.join(output_dir, f"{prefix}overlay_blocks.jpg"),
                    overlay_blocks,
                )

        # Store results
        results_by_method[detector_type] = {
            "aligned_image": cv2.cvtColor(result.aligned_image, cv2.COLOR_BGR2RGB),
            "diff_map": cv2.cvtColor(diff_map, cv2.COLOR_BGR2RGB),
            "heatmap": cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB),
            "heatmap_colorbar": cv2.cvtColor(heatmap_colorbar, cv2.COLOR_BGR2RGB),
            "overlayed_heatmap": cv2.cvtColor(overlayed_heatmap, cv2.COLOR_BGR2RGB),
            "highlighted": cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB),
            "histogram": cv2.cvtColor(histogram, cv2.COLOR_BGR2RGB),
            "stats_chart": cv2.cvtColor(stats_chart, cv2.COLOR_BGR2RGB),
            "composite": cv2.cvtColor(composite, cv2.COLOR_BGR2RGB),
            "stats": stats,
            "avg_delta_e": avg_delta_e,
            "similarity_score": result.similarity_score,
        }

        # Add blockwise analysis results
        if params.get("blockwise", {}).get("enabled", True):
            block_size_h = params.get("blockwise", {}).get("block_size_h", 32)
            block_size_w = params.get("blockwise", {}).get("block_size_w", 32)
            use_mask = params.get("blockwise", {}).get("use_mask", True)

            # Create block heatmap
            mask_for_block = alpha_mask if use_mask and alpha_mask is not None else None
            block_heatmap = visualize_block_heatmap(
                delta_e_map,
                block_size=(block_size_h, block_size_w),
                mask=mask_for_block,
                title=f"{detector_type.upper()} Block Color Difference Heatmap",
            )

            # Calculate block statistics
            blocks_info = analyze_blocks(
                delta_e_map,
                block_size=(block_size_h, block_size_w),
                mask=mask_for_block,
            )

            # Overlay block boundaries on original image
            overlay_blocks = overlay_block_boundaries(
                result.roi_template,
                blocks_info,
                threshold=params["color_difference"]["threshold"] * 0.8,
            )

            # Add to results dictionary
            results_by_method[detector_type]["block_heatmap"] = (
                cv2.cvtColor(block_heatmap, cv2.COLOR_BGR2RGB)
                if block_heatmap is not None
                else None
            )
            results_by_method[detector_type]["overlay_blocks"] = (
                cv2.cvtColor(overlay_blocks, cv2.COLOR_BGR2RGB)
                if overlay_blocks is not None
                else None
            )
            results_by_method[detector_type]["blocks_info"] = blocks_info

    # If comparing all methods, create comparison views
    if compare_all and len(results_by_method) > 0:
        # Create method comparison charts
        comparison_results = create_method_comparison(results_by_method)

        # Primary method results
        primary_method = params["detector"]["type"]
        if primary_method in results_by_method:
            primary_results = results_by_method[primary_method]
        else:
            # If primary method failed, use first successful method
            primary_results = list(results_by_method.values())[0]

        # Merge results
        return_results = primary_results.copy()
        return_results.update(
            {
                "comparison_aligned": comparison_results["aligned_comparison"],
                "comparison_heatmap": comparison_results["heatmap_comparison"],
                "comparison_stats": comparison_results["stats_comparison"],
                "all_results": results_by_method,
            }
        )

        return (
            return_results,
            f"Completed analysis with {len(results_by_method)} methods",
        )
    elif len(results_by_method) > 0:
        # Single method processing
        method = list(results_by_method.keys())[0]
        return (
            results_by_method[method],
            f"{method.upper()} method processing completed",
        )
    else:
        return None, "All methods failed to process"


# Create method comparison views
def create_method_comparison(results_by_method):
    """Create comparison views for different methods"""
    methods = list(results_by_method.keys())

    # Aligned images comparison
    aligned_images = [results_by_method[m]["aligned_image"] for m in methods]
    aligned_titles = [f"{m.upper()} Alignment Result" for m in methods]
    aligned_comparison = create_side_by_side_comparison(
        aligned_images, aligned_titles, "Alignment Results Comparison"
    )

    # Heatmap comparison
    heatmaps = [results_by_method[m]["heatmap"] for m in methods]
    heatmap_titles = [f"{m.upper()} Color Difference Heatmap" for m in methods]
    heatmap_comparison = create_side_by_side_comparison(
        heatmaps, heatmap_titles, "Color Difference Heatmaps Comparison"
    )

    # Statistics comparison
    stats_data = {m: results_by_method[m]["stats"] for m in methods}
    avg_delta_e = {m: results_by_method[m]["avg_delta_e"] for m in methods}
    sim_scores = {m: results_by_method[m]["similarity_score"] for m in methods}
    stats_comparison = create_stats_comparison(stats_data, avg_delta_e, sim_scores)

    return {
        "aligned_comparison": aligned_comparison,
        "heatmap_comparison": heatmap_comparison,
        "stats_comparison": stats_comparison,
    }


def create_side_by_side_comparison(images, titles, main_title="Comparison View"):
    """Create side-by-side comparison images"""
    n = len(images)
    if n == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    # Determine dimensions of each image
    heights = [img.shape[0] for img in images]
    widths = [img.shape[1] for img in images]
    max_height = max(heights)
    total_width = sum(widths)

    # Create blank canvas
    comparison = np.ones((max_height + 60, total_width, 3), dtype=np.uint8) * 255

    # Draw main title
    font = cv2.FONT_HERSHEY_SIMPLEX
    main_title_position = (int(total_width / 2 - 200), 30)
    cv2.putText(comparison, main_title, main_title_position, font, 1, (0, 0, 0), 2)

    # Place images and titles
    x_offset = 0
    for _i, (img, title) in enumerate(zip(images, titles, strict=False)):
        # Resize image while maintaining aspect ratio
        if img.shape[0] != max_height:
            scale = max_height / img.shape[0]
            new_width = int(img.shape[1] * scale)
            img = cv2.resize(img, (new_width, max_height))

        # Place image
        comparison[60 : 60 + img.shape[0], x_offset : x_offset + img.shape[1]] = img

        # Place title
        title_position = (x_offset + int(img.shape[1] / 2) - 100, 50)
        cv2.putText(comparison, title, title_position, font, 0.7, (0, 0, 0), 1)

        x_offset += img.shape[1]

    return comparison


def create_stats_comparison(stats_by_method, avg_delta_e, similarity_scores):
    """Create statistical data comparison chart"""
    # Create a table-style comparison image
    methods = list(stats_by_method.keys())
    n_methods = len(methods)

    # Table headers
    headers = [
        "Method",
        "Avg Delta E",
        "Similarity Score",
        "Max Delta E",
        "Std Dev",
        "Median",
        "Above Threshold %",
    ]
    rows = []

    for method in methods:
        stats = stats_by_method[method]
        rows.append(
            [
                method.upper(),
                f"{avg_delta_e[method]:.2f}",
                f"{similarity_scores[method]:.2f}",
                f"{stats.get('max', 0):.2f}",
                f"{stats.get('std_dev', 0):.2f}",
                f"{stats.get('median', 0):.2f}",
                f"{stats.get('above_threshold_percent', 0):.2f}%",
            ]
        )

    # Calculate maximum width for each column
    col_widths = [
        max(len(h), max(len(row[i]) for row in rows)) for i, h in enumerate(headers)
    ]

    # Create image
    cell_height = 40
    cell_padding = 10
    table_width = sum(w * 10 + cell_padding * 2 for w in col_widths)
    table_height = cell_height * (n_methods + 1)  # +1 for header

    # Create white background
    table_img = np.ones((table_height, table_width, 3), dtype=np.uint8) * 255

    # Draw table lines
    for i in range(n_methods + 2):  # Horizontal lines (+2 for top and bottom)
        y = i * cell_height
        cv2.line(table_img, (0, y), (table_width, y), (0, 0, 0), 1)

    x_offset = 0
    for w in col_widths:
        width = w * 10 + cell_padding * 2
        cv2.line(table_img, (x_offset, 0), (x_offset, table_height), (0, 0, 0), 1)
        x_offset += width
    cv2.line(table_img, (table_width, 0), (table_width, table_height), (0, 0, 0), 1)

    # Fill table header
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    x_offset = 0
    for i, header in enumerate(headers):
        width = col_widths[i] * 10 + cell_padding * 2
        x = x_offset + cell_padding
        y = cell_height - cell_padding
        cv2.putText(table_img, header, (x, y), font, font_scale, (0, 0, 0), 1)
        x_offset += width

    # Fill data rows
    for row_idx, row in enumerate(rows):
        x_offset = 0
        y = (row_idx + 1) * cell_height + cell_height - cell_padding
        for i, cell in enumerate(row):
            width = col_widths[i] * 10 + cell_padding * 2
            x = x_offset + cell_padding

            # Use different colors for different columns
            if i == 0:  # Method name column
                color = (0, 0, 128)  # Dark blue
            elif i == 1:  # Average delta E column
                # Set color based on delta E value (green-yellow-red)
                # Low delta E = green, high delta E = red
                delta_e_val = float(avg_delta_e[methods[row_idx]])
                if delta_e_val < 3:  # Low delta E
                    color = (0, 128, 0)  # Green
                elif delta_e_val < 6:  # Medium delta E
                    color = (0, 128, 128)  # Yellow
                else:  # High delta E
                    color = (0, 0, 128)  # Red
            else:
                color = (0, 0, 0)  # Black

            cv2.putText(table_img, cell, (x, y), font, font_scale, color, 1)
            x_offset += width

    # Add title
    title_img = np.ones((50, table_width, 3), dtype=np.uint8) * 255
    cv2.putText(
        title_img,
        "Method Comparison Results",
        (int(table_width / 2) - 100, 30),
        font,
        0.9,
        (0, 0, 0),
        2,
    )

    # Merge title and table
    result_img = np.vstack((title_img, table_img))

    return result_img


# ICC Color Space Conversion
def apply_icc_conversion(
    image_array: np.ndarray, icc_params: dict, output_dir: str = "alignment_results"
) -> dict:
    """
    Apply ICC color space conversion to an image

    Args:
        image_array: Input image array (BGR format from cv2)
        icc_params: ICC conversion parameters
        output_dir: Output directory for saving converted images

    Returns:
        dict: Conversion results including converted image and metadata
    """
    if not icc_params.get("enabled", False) or not icc_params.get(
        "apply_conversion", False
    ):
        return {
            "converted_image": None,
            "original_image": cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB),
            "comparison": None,
            "info": {"status": "ICC conversion disabled"},
        }

    try:
        # Get ICC profile paths
        available_profiles = get_available_icc_profiles()
        srgb_profile_name = icc_params.get("srgb_profile", "sRGB IEC61966-21.icc")
        cmyk_profile_name = icc_params.get("cmyk_profile", "JapanColor2001Coated.icc")

        srgb_profile_path = available_profiles.get(srgb_profile_name)
        cmyk_profile_path = available_profiles.get(cmyk_profile_name)

        if not srgb_profile_path or not cmyk_profile_path:
            return {
                "converted_image": None,
                "original_image": cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB),
                "comparison": None,
                "info": {
                    "status": "Error: ICC profiles not found",
                    "profiles": available_profiles,
                },
            }

        # Determine conversion direction
        conversion_direction = icc_params.get("conversion_direction", "srgb_to_cmyk")

        # Prepare output paths
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time()) if "time" in globals() else 0

        if conversion_direction == "srgb_to_cmyk":
            output_path = (
                os.path.join(output_dir, f"icc_srgb_to_cmyk_{timestamp}.tiff")
                if icc_params.get("save_converted", True)
                else None
            )
            converted_array, converted_pil = convert_color_space_array(
                image_array,
                output_image_path=output_path,
                srgb=srgb_profile_path,
                cmyk=cmyk_profile_path,
                to_cmyk=True,
            )
        else:  # cmyk_to_srgb
            output_path = (
                os.path.join(output_dir, f"icc_cmyk_to_srgb_{timestamp}.png")
                if icc_params.get("save_converted", True)
                else None
            )
            converted_array, converted_pil = convert_color_space_array(
                image_array,
                output_image_path=output_path,
                srgb=srgb_profile_path,
                cmyk=cmyk_profile_path,
                to_cmyk=False,
            )

        # Convert to RGB for display
        original_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

        # Handle different output formats
        if conversion_direction == "srgb_to_cmyk":
            # CMYK output - convert back to RGB for display
            if converted_array.shape[2] == 4:  # CMYK
                # Create a simple CMYK to RGB conversion for display
                converted_display = np.array(converted_pil.convert("RGB"))
            else:
                converted_display = converted_array
        else:
            # RGB output
            converted_display = converted_array

        # Create side-by-side comparison
        if original_rgb.shape[:2] != converted_display.shape[:2]:
            # Resize to match dimensions
            h, w = (
                min(original_rgb.shape[0], converted_display.shape[0]),
                min(original_rgb.shape[1], converted_display.shape[1]),
            )
            original_rgb = cv2.resize(original_rgb, (w, h))
            converted_display = cv2.resize(converted_display, (w, h))

        comparison = np.hstack([original_rgb, converted_display])

        # Add labels to comparison
        font = cv2.FONT_HERSHEY_SIMPLEX
        comparison_labeled = comparison.copy()
        cv2.putText(
            comparison_labeled, "Original", (10, 30), font, 1, (255, 255, 255), 2
        )
        cv2.putText(
            comparison_labeled,
            "ICC Converted",
            (w + 10, 30),
            font,
            1,
            (255, 255, 255),
            2,
        )

        return {
            "converted_image": converted_display,
            "original_image": original_rgb,
            "comparison": comparison_labeled,
            "info": {
                "status": "ICC conversion successful",
                "conversion_direction": conversion_direction,
                "srgb_profile": srgb_profile_name,
                "cmyk_profile": cmyk_profile_name,
                "rendering_intent": icc_params.get("rendering_intent", "perceptual"),
                "original_shape": original_rgb.shape,
                "converted_shape": converted_display.shape,
                "output_path": output_path if output_path else "Not saved",
            },
        }

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        return {
            "converted_image": None,
            "original_image": cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB),
            "comparison": None,
            "info": {
                "status": f"ICC conversion failed: {str(e)}",
                "error_details": error_details,
            },
        }
