"""
Pure Colorbar Analysis Interface Component

This component provides the interface for the redesigned pure color-based 
colorbar analysis system with enhanced ground truth comparison and clear 
CMYK/delta E reporting.
"""

import gradio as gr
from PIL import Image

from core.block_detection.pure_colorbar_analysis import (
    pure_colorbar_analysis_for_gradio,
)
from core.color.ground_truth_checker import ground_truth_checker


def process_pure_colorbar_analysis(
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
    Process pure colorbar analysis and return formatted results.

    Returns:
        Tuple of (annotated_image, status_message, results_html)
    """
    if input_image is None:
        return None, "No image provided", ""

    try:
        # Run the pure colorbar analysis
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

        # Create enhanced HTML display with pure color focus
        results_html = create_pure_colorbar_display(colorbar_data)

        status = f"✅ Pure color analysis complete: {len(colorbar_data)} colorbar(s), {total_blocks} pure color blocks"

        return annotated_image, status, results_html

    except Exception as e:
        error_msg = f"❌ Error during pure color analysis: {str(e)}"
        return input_image, error_msg, ""


def create_pure_colorbar_display(colorbar_data: list[dict]) -> str:
    """Create enhanced HTML display focused on pure color analysis results."""
    if not colorbar_data:
        return "<div class='no-results'>No pure colorbar data available</div>"

    html = """
    <style>
    .pure-colorbar-container { margin-bottom: 20px; border: 2px solid #2196F3; border-radius: 8px; padding: 15px; background: #f8f9fa; }
    .pure-colorbar-header { display: flex; align-items: center; margin-bottom: 12px; font-weight: bold; color: #1976D2; font-size: 16px; }
    .confidence-badge { background: #4CAF50; color: white; padding: 3px 8px; border-radius: 12px; font-size: 12px; margin-left: 10px; }
    .colorbar-segments { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px; }
    .segment-box { text-align: center; border: 1px solid #ddd; border-radius: 6px; padding: 8px; background: white; }
    .segment-label { font-size: 12px; color: #666; margin-bottom: 6px; font-weight: bold; }
    .segment-image { border: 1px solid #ccc; border-radius: 4px; max-width: 100%; height: auto; }
    .pure-color-blocks-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px; margin-top: 10px; }
    .pure-color-block { border: 2px solid #e0e0e0; border-radius: 6px; padding: 10px; background: white; text-align: center; font-size: 11px; transition: border-color 0.3s; }
    .pure-color-block.excellent { border-color: #4CAF50; background: #f1f8e9; }
    .pure-color-block.acceptable { border-color: #FF9800; background: #fff3e0; }
    .pure-color-block.poor { border-color: #f44336; background: #ffebee; }
    .color-preview { width: 40px; height: 40px; border-radius: 8px; margin: 0 auto 8px auto; border: 2px solid #333; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }
    .color-info { margin-bottom: 8px; }
    .pure-rgb-info { color: #1976D2; font-weight: bold; font-size: 12px; margin-bottom: 4px; }
    .cmyk-values { color: #333; font-size: 11px; margin-bottom: 6px; font-weight: bold; background: #f5f5f5; padding: 4px; border-radius: 3px; }
    .purity-info { margin-bottom: 6px; padding: 4px; border-radius: 3px; font-size: 10px; }
    .purity-info.high { background: #e8f5e8; color: #2e7d32; }
    .purity-info.medium { background: #fff3e0; color: #f57c00; }
    .purity-info.low { background: #ffebee; color: #c62828; }
    .delta-e-info { margin-top: 6px; padding: 5px; border-radius: 4px; font-size: 10px; font-weight: bold; }
    .delta-e-info.excellent { background: #c8e6c9; color: #1b5e20; }
    .delta-e-info.good { background: #ffe0b2; color: #e65100; }
    .delta-e-info.poor { background: #ffcdd2; color: #b71c1c; }
    .delta-e-value { font-size: 12px; font-weight: bold; }
    .accuracy-level { font-size: 9px; opacity: 0.9; }
    .ground-truth-match { font-size: 9px; opacity: 0.8; margin-top: 3px; }
    .status-indicator { font-size: 16px; margin-left: 5px; }
    .summary-stats { background: #e3f2fd; border: 1px solid #2196F3; border-radius: 6px; padding: 10px; margin-bottom: 15px; font-size: 12px; }
    .summary-stats h4 { margin: 0 0 8px 0; color: #1976D2; }
    </style>
    """

    # Calculate overall statistics
    total_blocks = 0
    excellent_count = 0
    acceptable_count = 0
    high_purity_count = 0
    all_delta_e = []

    for colorbar in colorbar_data:
        for analysis in colorbar.get("pure_color_analyses", []):
            if "error" not in analysis:
                total_blocks += 1
                gt_match = analysis.get("ground_truth_match", {})
                if gt_match.get("is_excellent"):
                    excellent_count += 1
                if gt_match.get("is_acceptable"):
                    acceptable_count += 1
                if analysis.get("purity_score", 0) >= 0.8:
                    high_purity_count += 1
                if "delta_e" in gt_match:
                    all_delta_e.append(gt_match["delta_e"])

    # Add summary statistics
    html += f"""
    <div class="summary-stats">
        <h4>📊 Pure Color Analysis Summary</h4>
        <div>Total pure color blocks analyzed: <strong>{total_blocks}</strong></div>
    """

    if all_delta_e:
        avg_delta_e = sum(all_delta_e) / len(all_delta_e)
        html += f"""
        <div>Average ΔE: <strong>{avg_delta_e:.2f}</strong></div>
        <div>Excellent colors (ΔE &lt; 1.0): <strong>{excellent_count}/{total_blocks}</strong> ({(excellent_count/total_blocks*100):.1f}%)</div>
        <div>Acceptable colors (ΔE &lt; 3.0): <strong>{acceptable_count}/{total_blocks}</strong> ({(acceptable_count/total_blocks*100):.1f}%)</div>
        <div>High purity colors (&gt; 0.8): <strong>{high_purity_count}/{total_blocks}</strong> ({(high_purity_count/total_blocks*100):.1f}%)</div>
        """

    html += "</div>"

    for colorbar in colorbar_data:
        colorbar_id = colorbar.get("colorbar_id", "?")
        confidence = colorbar.get("confidence", 0)
        block_count = colorbar.get("block_count", 0)
        original_colorbar = colorbar.get("original_colorbar")
        segmented_colorbar = colorbar.get("segmented_colorbar")
        pure_color_analyses = colorbar.get("pure_color_analyses", [])

        html += f"""
        <div class="pure-colorbar-container">
            <div class="pure-colorbar-header">
                🎯 Pure Colorbar {colorbar_id}
                <span class="confidence-badge">{confidence:.2f}</span>
            </div>
        """

        # Show original and segmented colorbar images
        if original_colorbar or segmented_colorbar:
            html += """
            <div class="colorbar-segments">
            """
            if original_colorbar:
                # Convert PIL image to base64 for display
                import base64
                import io

                buffer = io.BytesIO()
                original_colorbar.save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode()
                html += f"""
                <div class="segment-box">
                    <div class="segment-label">Original Colorbar</div>
                    <img src="data:image/png;base64,{img_str}" class="segment-image" alt="Original colorbar">
                </div>
                """

            if segmented_colorbar:
                # Convert PIL image to base64 for display
                buffer = io.BytesIO()
                segmented_colorbar.save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode()
                html += f"""
                <div class="segment-box">
                    <div class="segment-label">Detected Blocks</div>
                    <img src="data:image/png;base64,{img_str}" class="segment-image" alt="Segmented colorbar">
                </div>
                """

            html += "</div>"

        # Pure color blocks analysis
        if pure_color_analyses:
            html += """
            <div class="pure-color-blocks-grid">
            """

            for analysis in pure_color_analyses:
                if "error" in analysis:
                    continue

                block_id = analysis.get("block_id", "?")
                pure_rgb = analysis.get("pure_color_rgb", (0, 0, 0))
                pure_cmyk = analysis.get("pure_color_cmyk", (0, 0, 0, 0))
                purity_score = analysis.get("purity_score", 0.0)
                color_quality = analysis.get("color_quality", "Unknown")
                gt_match = analysis.get("ground_truth_match", {})

                # Determine block styling based on performance
                block_class = "pure-color-block"
                if gt_match.get("is_excellent"):
                    block_class += " excellent"
                elif gt_match.get("is_acceptable"):
                    block_class += " acceptable"
                else:
                    block_class += " poor"

                # Determine purity styling
                purity_class = "purity-info"
                if purity_score >= 0.8:
                    purity_class += " high"
                elif purity_score >= 0.6:
                    purity_class += " medium"
                else:
                    purity_class += " low"

                # Color preview style
                color_style = f"background-color: rgb({pure_rgb[0]}, {pure_rgb[1]}, {pure_rgb[2]});"

                html += f"""
                <div class="{block_class}">
                    <div class="color-preview" style="{color_style}"></div>
                    <div class="color-info">
                        <div class="pure-rgb-info">Block {colorbar_id}.{block_id}</div>
                        <div class="pure-rgb-info">RGB({pure_rgb[0]}, {pure_rgb[1]}, {pure_rgb[2]})</div>
                        <div class="cmyk-values">
                            C={pure_cmyk[0]}% M={pure_cmyk[1]}%<br>
                            Y={pure_cmyk[2]}% K={pure_cmyk[3]}%
                        </div>
                        <div class="{purity_class}">
                            Purity: {purity_score:.2f} ({color_quality})
                        </div>
                """

                # Ground truth comparison
                if gt_match.get("closest_color"):
                    delta_e = gt_match.get("delta_e", 0)
                    accuracy_level = gt_match.get("accuracy_level", "Unknown")
                    gt_color = gt_match["closest_color"]

                    # Delta E styling
                    delta_e_class = "delta-e-info"
                    status_icon = ""
                    if gt_match.get("is_excellent"):
                        delta_e_class += " excellent"
                        status_icon = "✅"
                    elif gt_match.get("is_acceptable"):
                        delta_e_class += " good"
                        status_icon = "⚠️"
                    else:
                        delta_e_class += " poor"
                        status_icon = "❌"

                    html += f"""
                        <div class="{delta_e_class}">
                            <div class="delta-e-value">ΔE: {delta_e:.2f} <span class="status-indicator">{status_icon}</span></div>
                            <div class="accuracy-level">{accuracy_level}</div>
                            <div class="ground-truth-match">vs {gt_color['name']}</div>
                        </div>
                    """

                html += """
                    </div>
                </div>
                """

            html += "</div>"
        else:
            html += f"<div style='text-align: center; color: #666; font-style: italic;'>No pure color blocks detected in colorbar {colorbar_id}</div>"

        html += "</div>"

    return html


def create_pure_colorbar_analysis_interface():
    """Create the Gradio interface for pure colorbar analysis"""

    with gr.Row():
        with gr.Column():
            gr.Markdown("## 🎯 Pure Color-Based Colorbar Analysis")
            gr.Markdown(
                "Upload an image with colorbars for **pure color-based analysis** with precise "
                "**CMYK matching** and **delta E calculations** against ground truth colors."
            )

            input_image = gr.Image(
                label="📷 Upload Colorbar Image", type="pil", scale=2
            )

            with gr.Accordion("🔧 Pure Color Analysis Settings", open=False):
                with gr.Row():
                    confidence_threshold = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.6,
                        step=0.1,
                        label="YOLO Confidence",
                        info="Detection confidence threshold",
                    )
                    box_expansion = gr.Slider(
                        minimum=0,
                        maximum=50,
                        value=10,
                        step=5,
                        label="Box Expansion (px)",
                        info="Expand detected colorbar boxes",
                    )

                with gr.Row():
                    block_area_threshold = gr.Slider(
                        minimum=10,
                        maximum=200,
                        value=50,
                        step=10,
                        label="Min Block Area",
                        info="Minimum area for color blocks",
                    )
                    block_aspect_ratio = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.3,
                        step=0.1,
                        label="Min Aspect Ratio",
                        info="Minimum aspect ratio for blocks",
                    )

                with gr.Row():
                    min_square_size = gr.Slider(
                        minimum=5,
                        maximum=50,
                        value=10,
                        step=5,
                        label="Min Block Size (px)",
                        info="Minimum block width/height",
                    )
                    purity_threshold = gr.Slider(
                        minimum=0.5,
                        maximum=1.0,
                        value=0.8,
                        step=0.05,
                        label="Purity Threshold",
                        info="Minimum color purity score",
                    )

            with gr.Row():
                analyze_btn = gr.Button(
                    "🎯 Analyze Pure Colors", variant="primary", scale=2
                )
                clear_btn = gr.Button("🗑️ Clear", scale=1)

        with gr.Column():
            result_image = gr.Image(label="📊 Analysis Results", type="pil", scale=2)

            status_text = gr.Textbox(
                label="Status", value="Upload → Analyze", interactive=False, scale=1
            )

    # Results section
    with gr.Row():
        with gr.Column():
            results_display = gr.HTML(
                label="🎨 Pure Color Analysis Results",
                value="<div style='text-align: center; color: #666; padding: 20px;'>Upload an image and click 'Analyze Pure Colors' to see detailed results with CMYK values and delta E comparisons.</div>",
            )

    # Ground truth reference section
    with gr.Row():
        with gr.Column():
            gr.Markdown("## 📋 Ground Truth Color Reference")
            with gr.Row():
                show_reference_btn = gr.Button("📊 Show Reference Chart")
                show_yaml_btn = gr.Button("📝 Show YAML Config")

            reference_chart = gr.Image(
                label="Ground Truth Reference Chart", visible=False
            )

            yaml_config = gr.Code(
                label="Ground Truth YAML Configuration", language="yaml", visible=False
            )

    # Event handlers
    def run_analysis(
        img, conf, box_exp, area_thresh, aspect_ratio, min_size, purity_thresh
    ):
        if img is None:
            return None, "❌ Please upload an image", ""

        return process_pure_colorbar_analysis(
            img,
            confidence_threshold=conf,
            box_expansion=box_exp,
            block_area_threshold=area_thresh,
            block_aspect_ratio=aspect_ratio,
            min_square_size=min_size,
            purity_threshold=purity_thresh,
        )

    def clear_all():
        return None, None, "Upload → Analyze", ""

    def show_reference_chart():
        try:
            reference_image = ground_truth_checker.generate_reference_chart()
            return gr.Image(value=reference_image, visible=True)
        except Exception as e:
            print(f"Error generating reference chart: {e}")
            return gr.Image(visible=False)

    def show_yaml_config():
        try:
            yaml_content = ground_truth_checker.get_palette_yaml()
            return gr.Code(value=yaml_content, visible=True)
        except Exception as e:
            print(f"Error generating YAML config: {e}")
            return gr.Code(value="# Error generating YAML config", visible=True)

    def hide_yaml_config():
        return gr.Code(visible=False)

    # Connect event handlers
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

    show_reference_btn.click(
        fn=show_reference_chart,
        outputs=[reference_chart],
    )

    show_yaml_btn.click(
        fn=show_yaml_config,
        outputs=[yaml_config],
    )

    input_image.change(
        fn=lambda img: "Ready → Analyze Pure Colors" if img else "Upload → Analyze",
        inputs=[input_image],
        outputs=[status_text],
    )

    return {
        "input_image": input_image,
        "result_image": result_image,
        "status_text": status_text,
        "results_display": results_display,
    }
