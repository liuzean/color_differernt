import gradio as gr

from core.color.icc_trans import get_available_icc_profiles


def create_settings_ui(config):
    with gr.Tabs():
        with gr.TabItem("🎨 Color & Analysis"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Preview & Display**")
                    # Move preview color space to settings
                    available_profiles = get_available_icc_profiles()
                    profile_names = ["sRGB (default)"] + (
                        list(available_profiles.keys()) if available_profiles else []
                    )
                    color_space_preview = gr.Dropdown(
                        choices=profile_names,
                        value="sRGB (default)",
                        label="Preview Color Space",
                        info="Select a color space to preview the images",
                    )

                    gr.Markdown("**Color Difference Analysis**")
                    threshold = gr.Number(
                        value=config["color_difference"]["threshold"],
                        minimum=1.0,
                        maximum=10.0,
                        step=0.5,
                        label="Color Difference Threshold",
                        info="Threshold for highlighting significant differences",
                    )
                    mask_background = gr.Checkbox(
                        value=config["color_difference"]["mask_background"],
                        label="Remove Background",
                        info="Use alpha channel to mask background",
                    )

                with gr.Column():
                    gr.Markdown("**Blockwise Analysis**")
                    block_size_h = gr.Dropdown(
                        choices=[8, 16, 24, 32, 48, 64],
                        value=config.get("blockwise", {}).get("block_size_h", 32),
                        label="Block Height",
                        info="Height of analysis blocks in pixels",
                    )
                    block_size_w = gr.Dropdown(
                        choices=[8, 16, 24, 32, 48, 64],
                        value=config.get("blockwise", {}).get("block_size_w", 32),
                        label="Block Width",
                        info="Width of analysis blocks in pixels",
                    )
                    use_mask = gr.Checkbox(
                        value=config.get("blockwise", {}).get("use_mask", True),
                        label="Analyze Foreground Only",
                        info="Use alpha channel mask to exclude background",
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**ICC Color Space Configuration**")
                    icc_enabled = gr.Checkbox(
                        value=config.get("icc", {}).get("enabled", False),
                        label="Enable ICC Color Space Conversion",
                        info="Apply ICC profile-based color space transformations",
                    )

                    srgb_profile = gr.Dropdown(
                        choices=list(available_profiles.keys())
                        if available_profiles
                        else [],
                        value=config.get("icc", {}).get(
                            "srgb_profile", "sRGB IEC61966-21.icc"
                        ),
                        label="sRGB ICC Profile",
                        info="Select the sRGB color space profile",
                        interactive=True,
                    )

                    cmyk_profile = gr.Dropdown(
                        choices=list(available_profiles.keys())
                        if available_profiles
                        else [],
                        value=config.get("icc", {}).get(
                            "cmyk_profile", "JapanColor2001Coated.icc"
                        ),
                        label="CMYK ICC Profile",
                        info="Select the CMYK color space profile",
                        interactive=True,
                    )

                with gr.Column():
                    gr.Markdown("**ICC Conversion Settings**")
                    apply_conversion = gr.Checkbox(
                        value=config.get("icc", {}).get("apply_conversion", False),
                        label="Apply ICC Conversion to Results",
                        info="Convert processed images using ICC profiles",
                    )

                    conversion_direction = gr.Radio(
                        choices=["srgb_to_cmyk", "cmyk_to_srgb"],
                        value=config.get("icc", {}).get(
                            "conversion_direction", "srgb_to_cmyk"
                        ),
                        label="Conversion Direction",
                        info="Choose the color space conversion direction",
                    )

                    save_converted = gr.Checkbox(
                        value=config.get("icc", {}).get("save_converted", True),
                        label="Save Converted Images",
                        info="Save ICC-converted images to output directory",
                    )

                    rendering_intent = gr.Dropdown(
                        choices=[
                            "perceptual",
                            "relative_colorimetric",
                            "saturation",
                            "absolute_colorimetric",
                        ],
                        value=config.get("icc", {}).get(
                            "rendering_intent", "perceptual"
                        ),
                        label="Rendering Intent",
                        info="ICC rendering intent for color conversion",
                    )

        with gr.TabItem("📐 Alignment & Detection"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Feature Detection**")
                    detector_type = gr.Dropdown(
                        ["surf", "sift", "orb"],
                        value=config["detector"]["type"],
                        label="Feature Detection Algorithm",
                        info="Choose the algorithm for feature detection",
                    )
                    nfeatures = gr.Dropdown(
                        choices=[500, 1000, 1500, 2000, 2500, 3000, 4000, 5000],
                        value=config["detector"]["nfeatures"],
                        label="Number of Features",
                        info="More features provide better accuracy but slower processing",
                    )

                    gr.Markdown("**Matching Parameters**")
                    ratio_threshold = gr.Dropdown(
                        choices=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
                        value=config["matcher"]["ratio_threshold"],
                        label="Ratio Threshold",
                        info="Lower values are more restrictive for matches",
                    )
                    cross_check = gr.Checkbox(
                        value=config["matcher"]["crossCheck"],
                        label="Enable Cross Check",
                        info="Ensures mutual best matches (slower but more accurate)",
                    )
                    ratio_test = gr.Checkbox(
                        value=config["matcher"]["ratio_test"],
                        label="Enable Ratio Test",
                        info="Filters matches based on distance ratio",
                    )

                with gr.Column():
                    gr.Markdown("**Alignment Parameters**")
                    ransac_threshold = gr.Dropdown(
                        choices=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0],
                        value=config["alignment"]["ransac_reproj_threshold"],
                        label="RANSAC Reprojection Threshold",
                        info="Maximum allowed reprojection error in pixels",
                    )
                    min_matches = gr.Dropdown(
                        choices=[4, 6, 8, 10, 12, 15, 20],
                        value=config["alignment"]["min_matches"],
                        label="Minimum Matches Required",
                        info="Minimum number of feature matches needed for alignment",
                    )
                    enable_refinement = gr.Checkbox(
                        value=config["alignment"]["refinement"]["enabled"],
                        label="Enable Refinement Alignment",
                        info="Perform a second alignment pass for higher accuracy",
                    )

                    gr.Markdown("**Analysis Options**")
                    compare_all = gr.Checkbox(
                        label="Compare All Methods",
                        value=config["visualization"].get("compare_all_methods", False),
                        info="Analyze with all three algorithms",
                    )

        with gr.TabItem("🖼️ Image Processing"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Image Preprocessing**")
                    max_dimension = gr.Dropdown(
                        choices=[500, 750, 1000, 1250, 1500, 2000, 2500, 3000],
                        value=config["preprocessing"]["resize_max_dimension"],
                        label="Maximum Image Dimension",
                        info="Larger images provide more detail but slower processing",
                    )
                    enhance_contrast = gr.Checkbox(
                        value=config["preprocessing"]["enhance_contrast"],
                        label="Enhance Contrast",
                        info="Apply CLAHE contrast enhancement",
                    )

                with gr.Column():
                    gr.Markdown("**Output Options**")
                    output_dir = gr.Textbox(
                        value=config["visualization"]["output_dir"],
                        label="Output Directory",
                        info="Directory to save analysis results",
                    )
                    save_intermediate = gr.Checkbox(
                        value=config["visualization"]["save_intermediate"],
                        label="Save Intermediate Results",
                        info="Save all visualization images to disk",
                    )

    processing_components = (
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
        icc_enabled,
        srgb_profile,
        cmyk_profile,
        apply_conversion,
        conversion_direction,
        save_converted,
        rendering_intent,
    )

    return color_space_preview, processing_components
