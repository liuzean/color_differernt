#!/usr/bin/env python

"""
Color Difference Analysis Gradio Interface - Simplified
"""

import functools

import gradio as gr
import matplotlib as plt


from .components.color_checker import create_color_checker_ui
from .components.colorbar_analysis import create_colorbar_analysis_ui
from .components.ground_truth_colorbar_demo import create_ground_truth_colorbar_demo_ui
from .components.preview import create_preview_ui, update_preview
from .components.results import create_results_ui
from .components.settings import create_settings_ui
from .config import load_config
from .handlers.callbacks import process_images_handler, save_config_handler

plt.use("Agg")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

config = load_config()


def create_interface():
    """Create the main Gradio interface."""
    with gr.Blocks(title="Color Difference Analysis Tool") as demo:
        gr.Markdown("# Color Difference Analysis Tool")

        with gr.Tabs():
            # Intelligent Colorbar Analysis Tab
            with gr.TabItem("🎯 Colorbar Analysis"):
                create_colorbar_analysis_ui()

            # Ground-Truth Colorbar Demo Tab
            with gr.TabItem("🎨 Ground-Truth Demo"):
                create_ground_truth_colorbar_demo_ui()

            # CMYK Color Checker Tab
            with gr.TabItem("CMYK Color Checker"):
                create_color_checker_ui()

            # Main Analysis Tab
            with gr.TabItem("Analysis"):
                with gr.Row():
                    with gr.Column(scale=2):
                        (
                            template_file,
                            target_file,
                            template_preview,
                            target_preview,
                        ) = create_preview_ui()

                    with gr.Column(scale=1):
                        process_btn = gr.Button("Start Analysis", variant="primary")
                        save_btn = gr.Button("Save Settings", variant="secondary")

                        result_text = gr.Textbox(label="Status", lines=2)
                        avg_delta_e = gr.Number(label="Average ΔE")
                        progress = gr.Textbox(label="Progress", interactive=False)

                # Results section - simplified
                (
                    aligned_image,
                    diff_map,
                    heatmap,
                    heatmap_colorbar,
                    overlayed_heatmap,
                    highlighted,
                    block_heatmap,
                    overlay_blocks,
                    composite,
                    histogram,
                    stats_chart,
                    stats_display,
                    comparison_tab,
                    comparison_aligned,
                    comparison_heatmap,
                    comparison_stats,
                    icc_original,
                    icc_converted,
                    icc_comparison,
                    icc_info,
                ) = create_results_ui()

                # Settings section - simplified
                color_space_preview, settings_components = create_settings_ui(config)
                # color_space_preview = settings_components[0]  # Extract color_space_preview separately

        # Set up callbacks for main analysis
        # color_space_preview = settings_components[0]  # First setting component

        template_file.change(
            fn=functools.partial(
                update_preview,
                srgb_profile_name=config.get("icc", {}).get(
                    "srgb_profile", "sRGB IEC61966-21.icc"
                ),
                cmyk_profile_name=config.get("icc", {}).get(
                    "cmyk_profile", "JapanColor2001Coated.icc"
                ),
            ),
            inputs=[template_file, color_space_preview],
            outputs=[template_preview],
        )

        target_file.change(
            fn=functools.partial(
                update_preview,
                srgb_profile_name=config.get("icc", {}).get(
                    "srgb_profile", "sRGB IEC61966-21.icc"
                ),
                cmyk_profile_name=config.get("icc", {}).get(
                    "cmyk_profile", "JapanColor2001Coated.icc"
                ),
            ),
            inputs=[target_file, color_space_preview],
            outputs=[target_preview],
        )

        color_space_preview.change(
            fn=functools.partial(
                update_preview,
                srgb_profile_name=config.get("icc", {}).get(
                    "srgb_profile", "sRGB IEC61966-21.icc"
                ),
                cmyk_profile_name=config.get("icc", {}).get(
                    "cmyk_profile", "JapanColor2001Coated.icc"
                ),
            ),
            inputs=[template_file, color_space_preview],
            outputs=[template_preview],
        )

        color_space_preview.change(
            fn=functools.partial(
                update_preview,
                srgb_profile_name=config.get("icc", {}).get(
                    "srgb_profile", "sRGB IEC61966-21.icc"
                ),
                cmyk_profile_name=config.get("icc", {}).get(
                    "cmyk_profile", "JapanColor2001Coated.icc"
                ),
            ),
            inputs=[target_file, color_space_preview],
            outputs=[target_preview],
        )

        # Process button callback
        process_btn.click(
            fn=process_images_handler,
            inputs=[
                template_file,
                target_file,
                *settings_components,  # Unpack the processing components tuple
            ],
            outputs=[
                result_text,
                avg_delta_e,
                progress,
                aligned_image,
                diff_map,
                heatmap,
                heatmap_colorbar,
                overlayed_heatmap,
                highlighted,
                block_heatmap,
                overlay_blocks,
                composite,
                histogram,
                stats_chart,
                stats_display,
                comparison_aligned,
                comparison_heatmap,
                comparison_stats,
                icc_original,
                icc_converted,
                icc_comparison,
                icc_info,
            ],
        )

        # Save config callback
        save_btn.click(
            fn=save_config_handler,
            inputs=settings_components,  # Use the processing components tuple
            outputs=[result_text],
        )

    return demo


def launch_interface():
    """Launch the Gradio interface."""
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True,
        show_api=False,
    )
