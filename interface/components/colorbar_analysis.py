"""
Redesigned Colorbar Analysis Interface
Features:
- Pure color-based analysis only
- Concise shared result components
- Enhanced ground truth matching
"""

import gradio as gr
from PIL import Image

from core.block_detection.pure_colorbar_analysis import (
    pure_colorbar_analysis_for_gradio,
)
from .shared_results import update_shared_results_display


def process_colorbar_analysis(
    input_image: Image.Image,
    # YOLO parameters
    confidence_threshold: float = 0.6,
    box_expansion: int = 10,
    # Block detection parameters
    block_area_threshold: int = 50,
    block_aspect_ratio: float = 0.3,
    min_square_size: int = 10,
    # Pure color analysis parameters
    purity_threshold: float = 0.8,
) -> tuple[Image.Image, str, str]:
    """
    Process pure color-based colorbar analysis and return formatted results.

    Returns:
        Tuple of (annotated_image, status_message, results_html)
    """
    if input_image is None:
        return None, "No image provided", ""

    try:
        # Use pure color analysis only
        (
            annotated_image,
            colorbar_data,
            analysis_report,
            total_blocks,
        ) = pure_colorbar_analysis_for_gradio(
            input_image,
            confidence_threshold=confidence_threshold,
            box_expansion=box_expansion,
            block_area_threshold=block_area_threshold,
            block_aspect_ratio=block_aspect_ratio,
            min_square_size=min_square_size,
            purity_threshold=purity_threshold,
        )

        if not colorbar_data:
            return annotated_image, "No colorbars detected", ""

        # Create concise HTML display using shared component
        results_html = update_shared_results_display(colorbar_data)

        status = f"✅ Pure color analysis complete: {len(colorbar_data)} colorbar(s), {total_blocks} pure color blocks"

        return annotated_image, status, results_html

    except Exception as e:
        error_msg = f"❌ Error during colorbar analysis: {str(e)}"
        return input_image, error_msg, ""


def create_colorbar_analysis_ui():
    """Create colorbar analysis UI with pure color analysis only."""

    with gr.Column():
        with gr.Row():
            # Input column (compact)
            with gr.Column(scale=1):
                input_image = gr.Image(label="📷 Image", type="pil", height=250)

                # Compact parameters in single accordion
                with gr.Accordion("⚙️ Parameters", open=False):
                    with gr.Row():
                        confidence_threshold = gr.Slider(
                            0.1, 1.0, 0.5, step=0.05, label="Confidence"
                        )
                        box_expansion = gr.Slider(0, 50, 10, step=1, label="Expansion")
                    with gr.Row():
                        block_area_threshold = gr.Slider(
                            10, 200, 50, step=5, label="Min Area"
                        )
                        min_square_size = gr.Slider(1, 50, 5, step=1, label="Min Size")
                    with gr.Row():
                        block_aspect_ratio = gr.Slider(
                            0.1, 1.0, 0.3, step=0.05, label="Aspect Ratio"
                        )
                        purity_threshold = gr.Slider(
                            0.5, 1.0, 0.8, step=0.05, label="Purity"
                        )

                # Compact buttons
                with gr.Row():
                    analyze_btn = gr.Button("🚀 Analyze", variant="primary", scale=2)
                    clear_btn = gr.Button("🧹 Clear", scale=1)

                status_text = gr.Textbox(
                    label="Status", value="Upload → Analyze", interactive=False, lines=1
                )

            # Results column
            with gr.Column(scale=2):
                result_image = gr.Image(label="🎯 Results", type="pil", height=250)

        # Full-width results
        results_display = gr.HTML(
            value="<div style='text-align: center; padding: 15px; color: #666; background: #f9f9f9; border-radius: 6px;'>📷 Upload and analyze to see results</div>"
        )

    # Event handlers
    def run_analysis(
        image,
        conf_thresh,
        box_exp,
        block_area,
        block_ratio,
        min_square,
        purity_thresh,
    ):
        return process_colorbar_analysis(
            image,
            confidence_threshold=conf_thresh,
            box_expansion=box_exp,
            block_area_threshold=block_area,
            block_aspect_ratio=block_ratio,
            min_square_size=min_square,
            purity_threshold=purity_thresh,
        )

    def clear_all():
        return (
            None,
            None,
            "Upload → Analyze",
            "<div style='text-align: center; padding: 15px; color: #666; background: #f9f9f9; border-radius: 6px;'>📷 Upload and analyze</div>",
        )

    # Wire up events
    analyze_btn.click(
        fn=run_analysis,
        inputs=[
            input_image,
            confidence_threshold,
            box_expansion,
            block_area_threshold,
            block_aspect_ratio,
            min_square_size,
            purity_threshold,
        ],
        outputs=[result_image, status_text, results_display],
    )

    clear_btn.click(
        fn=clear_all, outputs=[input_image, result_image, status_text, results_display]
    )

    input_image.change(
        fn=lambda img: "Ready → Analyze" if img else "Upload → Analyze",
        inputs=[input_image],
        outputs=[status_text],
    )

    return (
        input_image,
        result_image,
        status_text,
        results_display,
        confidence_threshold,
        box_expansion,
        block_area_threshold,
        block_aspect_ratio,
        min_square_size,
        purity_threshold,
        analyze_btn,
        clear_btn,
    )
