import traceback

import cv2

from ..config import save_config
from ..processing import apply_icc_conversion, process_images


def process_images_handler(
    template_file,
    target_file,
    detector_type,
    compare_all,
    nfeatures,
    cross_check,
    ratio_test,
    ratio_threshold,
    ransac_threshold,
    min_matches,
    enable_refinement,
    threshold,
    mask_background,
    max_dimension,
    enhance_contrast,
    output_dir,
    save_intermediate,
    block_size_h=32,
    block_size_w=32,
    use_mask=True,
    # ICC parameters
    icc_enabled=False,
    srgb_profile="sRGB IEC61966-21.icc",
    cmyk_profile="JapanColor2001Coated.icc",
    apply_conversion=False,
    conversion_direction="srgb_to_cmyk",
    save_converted=True,
    rendering_intent="perceptual",
    config=None,
):
    if template_file is None or target_file is None:
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "Please upload both template and target images",
            "Please upload images",
            None,
            None,
            None,
            None,
            None,
            # ICC results (no files case)
            None,
            None,
            None,
            {"status": "Please upload both template and target images"},
        )

    # Get file paths
    template_path = template_file.name if hasattr(template_file, "name") else None
    target_path = target_file.name if hasattr(target_file, "name") else None

    # Output debug information for troubleshooting
    print(f"Template image path: {template_path}")
    print(f"Target image path: {target_path}")

    if not template_path or not target_path:
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "Unable to get file paths",
            "File error",
            None,
            None,
            None,
            None,
            None,
            # ICC results (file path error case)
            None,
            None,
            None,
            {"status": "Unable to get file paths"},
        )

    # Update progress

    # Update configuration
    if config is None or not isinstance(config, dict):
        # Load default configuration if config is not provided or is invalid
        from ..config import load_config

        params = load_config()
    else:
        params = config.copy()
    params["detector"]["type"] = detector_type
    params["detector"]["nfeatures"] = nfeatures
    params["matcher"]["crossCheck"] = cross_check
    params["matcher"]["ratio_test"] = ratio_test
    params["matcher"]["ratio_threshold"] = ratio_threshold
    params["alignment"]["ransac_reproj_threshold"] = ransac_threshold
    params["alignment"]["min_matches"] = min_matches
    params["alignment"]["refinement"]["enabled"] = enable_refinement
    params["color_difference"]["method"] = "ciede2000"
    params["color_difference"]["threshold"] = threshold
    params["color_difference"]["mask_background"] = mask_background
    params["preprocessing"]["resize_max_dimension"] = max_dimension
    params["preprocessing"]["enhance_contrast"] = enhance_contrast
    params["visualization"]["output_dir"] = output_dir
    params["visualization"]["save_intermediate"] = save_intermediate
    params["visualization"]["compare_all_methods"] = compare_all

    # Update blockwise analysis configuration
    if "blockwise" not in params:
        params["blockwise"] = {}
    params["blockwise"]["enabled"] = True
    params["blockwise"]["block_size_h"] = block_size_h
    params["blockwise"]["block_size_w"] = block_size_w
    params["blockwise"]["use_mask"] = use_mask

    # Update ICC configuration
    if "icc" not in params:
        params["icc"] = {}
    params["icc"]["enabled"] = icc_enabled
    params["icc"]["srgb_profile"] = srgb_profile
    params["icc"]["cmyk_profile"] = cmyk_profile
    params["icc"]["apply_conversion"] = apply_conversion
    params["icc"]["conversion_direction"] = conversion_direction
    params["icc"]["save_converted"] = save_converted
    params["icc"]["rendering_intent"] = rendering_intent

    # Process images
    try:
        # Use file paths
        results, message = process_images(template_path, target_path, params)

        if results is None:
            return (
                message,  # result_text
                None,  # avg_delta_e
                "Processing failed",  # progress
                None,  # aligned_image
                None,  # diff_map
                None,  # heatmap
                None,  # heatmap_colorbar
                None,  # overlayed_heatmap
                None,  # highlighted
                None,  # block_heatmap
                None,  # overlay_blocks
                None,  # composite
                None,  # histogram
                None,  # stats_chart
                None,  # stats_display
                None,  # comparison_aligned
                None,  # comparison_heatmap
                None,  # comparison_stats
                None,  # icc_original
                None,  # icc_converted
                None,  # icc_comparison
                "Processing failed",  # icc_info
            )

        # Extract results
        aligned_img = results.get("aligned_image")
        diff_img = results.get("diff_map")
        heatmap_img = results.get("heatmap")
        heatmap_cb_img = results.get("heatmap_colorbar")
        overlay_img = results.get("overlayed_heatmap")
        highlight_img = results.get("highlighted")
        histogram_img = results.get("histogram")
        stats_img = results.get("stats_chart")
        composite_img = results.get("composite")
        stats_data = results.get("stats")
        delta_e = results.get("avg_delta_e")

        # Blockwise analysis results
        block_heatmap_img = results.get("block_heatmap")
        overlay_blocks_img = results.get("overlay_blocks")

        # Comparison results
        comp_aligned = results.get("comparison_aligned")
        comp_heatmap = results.get("comparison_heatmap")
        comp_stats = results.get("comparison_stats")

        # ICC conversion
        icc_results = None
        if icc_enabled and apply_conversion and aligned_img is not None:
            # Convert aligned image using ICC profiles
            aligned_bgr = cv2.cvtColor(aligned_img, cv2.COLOR_RGB2BGR)
            icc_results = apply_icc_conversion(aligned_bgr, params["icc"], output_dir)

        return (
            message,  # result_text
            delta_e,  # avg_delta_e
            "Processing completed",  # progress
            aligned_img,  # aligned_image
            diff_img,  # diff_map
            heatmap_img,  # heatmap
            heatmap_cb_img,  # heatmap_colorbar
            overlay_img,  # overlayed_heatmap
            highlight_img,  # highlighted
            block_heatmap_img,  # block_heatmap
            overlay_blocks_img,  # overlay_blocks
            composite_img,  # composite
            histogram_img,  # histogram
            stats_img,  # stats_chart
            stats_data,  # stats_display
            comp_aligned,  # comparison_aligned
            comp_heatmap,  # comparison_heatmap
            comp_stats,  # comparison_stats
            # ICC results
            icc_results.get("original_image") if icc_results else None,  # icc_original
            icc_results.get("converted_image")
            if icc_results
            else None,  # icc_converted
            icc_results.get("comparison") if icc_results else None,  # icc_comparison
            str(icc_results.get("info", {"status": "ICC conversion not performed"}))
            if icc_results
            else "ICC conversion not performed",  # icc_info
        )

    except Exception as e:
        print("Exception occurred during processing:")
        traceback.print_exc()
        error_message = f"Error occurred during processing: {str(e)}"
        return (
            error_message,  # result_text
            None,  # avg_delta_e
            "Error occurred",  # progress
            None,  # aligned_image
            None,  # diff_map
            None,  # heatmap
            None,  # heatmap_colorbar
            None,  # overlayed_heatmap
            None,  # highlighted
            None,  # block_heatmap
            None,  # overlay_blocks
            None,  # composite
            None,  # histogram
            None,  # stats_chart
            None,  # stats_display
            None,  # comparison_aligned
            None,  # comparison_heatmap
            None,  # comparison_stats
            # ICC results (error case)
            None,  # icc_original
            None,  # icc_converted
            None,  # icc_comparison
            "Error occurred during processing",  # icc_info
        )


# Save configuration function
def save_config_handler(
    detector_type,
    compare_all,
    nfeatures,
    cross_check,
    ratio_test,
    ratio_threshold,
    ransac_threshold,
    min_matches,
    enable_refinement,
    threshold,
    mask_background,
    max_dimension,
    enhance_contrast,
    output_dir,
    save_intermediate,
    block_size_h,
    block_size_w,
    use_mask,
    # ICC parameters
    icc_enabled=False,
    srgb_profile="sRGB IEC61966-21.icc",
    cmyk_profile="JapanColor2001Coated.icc",
    apply_conversion=False,
    conversion_direction="srgb_to_cmyk",
    save_converted=True,
    rendering_intent="perceptual",
    config_path="config.yaml",
    config=None,
):
    # Update configuration
    if config is None or not isinstance(config, dict):
        # Load default configuration if config is not provided or is invalid
        from ..config import load_config

        config = load_config()

    config["detector"]["type"] = detector_type
    config["detector"]["nfeatures"] = nfeatures
    config["matcher"]["crossCheck"] = cross_check
    config["matcher"]["ratio_test"] = ratio_test
    config["matcher"]["ratio_threshold"] = ratio_threshold
    config["alignment"]["ransac_reproj_threshold"] = ransac_threshold
    config["alignment"]["min_matches"] = min_matches
    config["alignment"]["refinement"]["enabled"] = enable_refinement
    config["color_difference"]["method"] = "ciede2000"
    config["color_difference"]["threshold"] = threshold
    config["color_difference"]["mask_background"] = mask_background
    config["preprocessing"]["resize_max_dimension"] = max_dimension
    config["preprocessing"]["enhance_contrast"] = enhance_contrast
    config["visualization"]["output_dir"] = output_dir
    config["visualization"]["save_intermediate"] = save_intermediate
    config["visualization"]["compare_all_methods"] = compare_all

    # Save blockwise analysis configuration
    if "blockwise" not in config:
        config["blockwise"] = {}
    config["blockwise"]["enabled"] = True
    config["blockwise"]["block_size_h"] = block_size_h
    config["blockwise"]["block_size_w"] = block_size_w
    config["blockwise"]["use_mask"] = use_mask

    # Save ICC configuration
    if "icc" not in config:
        config["icc"] = {}
    config["icc"]["enabled"] = icc_enabled
    config["icc"]["srgb_profile"] = srgb_profile
    config["icc"]["cmyk_profile"] = cmyk_profile
    config["icc"]["apply_conversion"] = apply_conversion
    config["icc"]["conversion_direction"] = conversion_direction
    config["icc"]["save_converted"] = save_converted
    config["icc"]["rendering_intent"] = rendering_intent

    # Save configuration
    save_config(config, config_path)
    return f"✅ Configuration saved to {config_path}"
