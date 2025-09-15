# interface/components/colorbar_analysis.py

"""
Redesigned Colorbar Analysis Interface
Features:
- Pure color-based analysis only
- Concise shared result components
- Enhanced ground truth matching
"""

import gradio as gr
from PIL import Image

# [修改] 虽然函数名未变，但我们现在导入的是支持“双YOLO”和11组色卡匹配的新版本
from core.block_detection.pure_colorbar_analysis import (
    pure_colorbar_analysis_for_gradio,
)
from .shared_results import update_shared_results_display


def process_colorbar_analysis(
    input_image: Image.Image,
    # YOLO parameters
    confidence_threshold: float = 0.6,
    box_expansion: int = 10,
    # [修改] 更新参数以匹配新的YOLO色块检测流程
    yolo_block_confidence: float = 0.5,
    block_area_threshold: int = 50,
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
        # [修改] 调用新的后台函数，并传入新的参数
        (
            annotated_image,
            colorbar_data,
            analysis_report,
            total_blocks,
        ) = pure_colorbar_analysis_for_gradio(
            input_image,
            confidence_threshold=confidence_threshold,
            box_expansion=box_expansion,
            yolo_block_confidence=yolo_block_confidence,  # 新增YOLO色块置信度
            block_area_threshold=block_area_threshold,  # 保留最小面积作为过滤条件
            purity_threshold=purity_threshold,
        )

        if not colorbar_data:
            # 确保即使没有检测到色板，也返回标注后的图像
            return annotated_image or input_image, "No colorbars detected", ""

        # Create concise HTML display using shared component
        results_html = update_shared_results_display(colorbar_data)

        status = f"✅ YOLO analysis complete: {len(colorbar_data)} colorbar(s), {total_blocks} pure color blocks found."

        return annotated_image, status, results_html

    except Exception as e:
        import traceback

        traceback.print_exc()
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
                    gr.Markdown("Colorbar Detection (YOLO)")
                    with gr.Row():
                        confidence_threshold = gr.Slider(
                            0.1, 1.0, 0.5, step=0.05, label="Confidence"
                        )
                        box_expansion = gr.Slider(0, 50, 10, step=1, label="Expansion")

                    gr.Markdown("Block Detection (YOLO)")
                    with gr.Row():
                        # [修改] 替换旧滑块为新的YOLO色块置信度滑块
                        yolo_block_confidence = gr.Slider(
                            0.1, 1.0, 0.5, step=0.05, label="Block Confidence"
                        )
                        block_area_threshold = gr.Slider(
                            10, 200, 50, step=5, label="Min Area"
                        )

                    gr.Markdown("Color Analysis")
                    with gr.Row():
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
    # [修改] 更新事件处理函数以使用新参数
    def run_analysis(
        image,
        conf_thresh,
        box_exp,
        yolo_block_conf,  # 新参数
        block_area,
        purity_thresh,
    ):
        return process_colorbar_analysis(
            image,
            confidence_threshold=conf_thresh,
            box_expansion=box_exp,
            yolo_block_confidence=yolo_block_conf,  # 传递新参数
            block_area_threshold=block_area,
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
    # [修改] 更新传递给点击事件的输入组件列表
    analyze_btn.click(
        fn=run_analysis,
        inputs=[
            input_image,
            confidence_threshold,
            box_expansion,
            yolo_block_confidence,
            block_area_threshold,
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

    # [兼容性保留] 为了确保返回的组件数量与其他UI模块一致，这里返回None占位
    # 删除了 block_aspect_ratio 和 min_square_size
    return (
        input_image,
        result_image,
        status_text,
        results_display,
        confidence_threshold,
        box_expansion,
        yolo_block_confidence,  # 新增
        block_area_threshold,
        purity_threshold,
        analyze_btn,
        clear_btn,
    )
