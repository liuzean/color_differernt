"""
Ground-Truth Colorbar Demo Interface Component

This component provides a demonstration interface for ground-truth colorbar generation
and testing with the colorbar analysis system. It includes pre-generated examples
and allows users to test the colorbar detection on known ground-truth images.
"""

import os
import tempfile

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from core.color.ground_truth_checker import ground_truth_checker
from core.color.palette_generator import ColorPaletteGenerator
from core.color.utils import cmyk_to_rgb
from .shared_results import update_shared_results_display


class GroundTruthColorbarDemo:
    """Generate ground-truth colorbar images for demonstration and testing"""

    def __init__(self):
        self.palette_generator = ColorPaletteGenerator()
        self.demo_configs = self._create_demo_configurations()

    def _create_demo_configurations(self) -> list[dict]:
        """Create fixed reference colorbar configurations in order: C, M, Y, K, Complex 1, 2, 3"""
        # Fixed reference colors from ground truth checker
        fixed_colors = [
            [100, 0, 0, 0],  # Pure Cyan
            [0, 100, 0, 0],  # Pure Magenta
            [0, 0, 100, 0],  # Pure Yellow
            [0, 0, 0, 100],  # Pure Black
            [50, 60, 0, 50],  # Complex Color 1
            [80, 50, 60, 0],  # Complex Color 2
            [60, 60, 60, 0],  # Complex Color 3
        ]

        return [
            {
                "name": "Ground Truth Reference",
                "description": "Fixed reference colors: C, M, Y, K, Complex 1, 2, 3",
                "gradients": [
                    {
                        "name": f"Color {i + 1}",
                        "start_color": color,
                        "end_color": color,
                        "steps": 1,
                    }
                    for i, color in enumerate(fixed_colors)
                ],
                "layout": {
                    "swatch_size_mm": 8,
                    "swatch_spacing_mm": 0.5,
                    "group_spacing_mm": 2,
                    "arrangement": "single_row",
                    "columns": 7,
                },
            },
            {
                "name": "Pure Cyan",
                "description": "Pure Cyan reference (C:100, M:0, Y:0, K:0)",
                "gradients": [
                    {
                        "name": "Cyan",
                        "start_color": [100, 0, 0, 0],
                        "end_color": [100, 0, 0, 0],
                        "steps": 1,
                    },
                ],
                "layout": {
                    "swatch_size_mm": 10,
                    "swatch_spacing_mm": 0.5,
                    "group_spacing_mm": 2,
                    "arrangement": "single_column",
                    "columns": 1,
                },
            },
            {
                "name": "Pure Magenta",
                "description": "Pure Magenta reference (C:0, M:100, Y:0, K:0)",
                "gradients": [
                    {
                        "name": "Magenta",
                        "start_color": [0, 100, 0, 0],
                        "end_color": [0, 100, 0, 0],
                        "steps": 1,
                    },
                ],
                "layout": {
                    "swatch_size_mm": 10,
                    "swatch_spacing_mm": 0.5,
                    "group_spacing_mm": 2,
                    "arrangement": "single_column",
                    "columns": 1,
                },
            },
            {
                "name": "Pure Yellow",
                "description": "Pure Yellow reference (C:0, M:0, Y:100, K:0)",
                "gradients": [
                    {
                        "name": "Yellow",
                        "start_color": [0, 0, 100, 0],
                        "end_color": [0, 0, 100, 0],
                        "steps": 1,
                    },
                ],
                "layout": {
                    "swatch_size_mm": 10,
                    "swatch_spacing_mm": 0.5,
                    "group_spacing_mm": 2,
                    "arrangement": "single_column",
                    "columns": 1,
                },
            },
            {
                "name": "Pure Black",
                "description": "Pure Black reference (C:0, M:0, Y:0, K:100)",
                "gradients": [
                    {
                        "name": "Black",
                        "start_color": [0, 0, 0, 100],
                        "end_color": [0, 0, 0, 100],
                        "steps": 1,
                    },
                ],
                "layout": {
                    "swatch_size_mm": 10,
                    "swatch_spacing_mm": 0.5,
                    "group_spacing_mm": 2,
                    "arrangement": "single_column",
                    "columns": 1,
                },
            },
            {
                "name": "Complex Color 1",
                "description": "Complex Color 1 reference (C:50, M:60, Y:0, K:50)",
                "gradients": [
                    {
                        "name": "Complex1",
                        "start_color": [50, 60, 0, 50],
                        "end_color": [50, 60, 0, 50],
                        "steps": 1,
                    },
                ],
                "layout": {
                    "swatch_size_mm": 10,
                    "swatch_spacing_mm": 0.5,
                    "group_spacing_mm": 2,
                    "arrangement": "single_column",
                    "columns": 1,
                },
            },
            {
                "name": "Complex Color 2",
                "description": "Complex Color 2 reference (C:80, M:50, Y:60, K:0)",
                "gradients": [
                    {
                        "name": "Complex2",
                        "start_color": [80, 50, 60, 0],
                        "end_color": [80, 50, 60, 0],
                        "steps": 1,
                    },
                ],
                "layout": {
                    "swatch_size_mm": 10,
                    "swatch_spacing_mm": 0.5,
                    "group_spacing_mm": 2,
                    "arrangement": "single_column",
                    "columns": 1,
                },
            },
            {
                "name": "Complex Color 3",
                "description": "Complex Color 3 reference (C:60, M:60, Y:60, K:0)",
                "gradients": [
                    {
                        "name": "Complex3",
                        "start_color": [60, 60, 60, 0],
                        "end_color": [60, 60, 60, 0],
                        "steps": 1,
                    },
                ],
                "layout": {
                    "swatch_size_mm": 10,
                    "swatch_spacing_mm": 0.5,
                    "group_spacing_mm": 2,
                    "arrangement": "single_column",
                    "columns": 1,
                },
            },
        ]

    def generate_colorbar_image(self, config: dict, dpi: int = 300) -> Image.Image:
        """Generate a colorbar image from configuration"""
        try:
            # Create temporary directory for palette generation
            with tempfile.TemporaryDirectory() as temp_dir:
                # Generate palette using the existing generator
                palette_path = os.path.join(temp_dir, "colorbar")

                self.palette_generator.generate_palette(
                    palette_data=config["gradients"],
                    output_path=palette_path,
                    layout_config=config["layout"],
                    output_dpi=dpi,
                    generate_pdf=False,
                    generate_tiff=True,
                )

                # Convert TIFF to RGB image for display
                tiff_path = f"{palette_path}.tiff"
                if os.path.exists(tiff_path):
                    # Convert CMYK TIFF to RGB PNG for display
                    png_path = f"{palette_path}.png"
                    self.palette_generator.convert_cmyk_tiff_to_png(tiff_path, png_path)

                    if os.path.exists(png_path):
                        return Image.open(png_path)

                # Fallback: create simple colorbar directly
                return self._create_simple_colorbar(config, dpi)

        except Exception as e:
            print(f"Error generating colorbar: {e}")
            return self._create_simple_colorbar(config, dpi)

    def _create_simple_colorbar(self, config: dict, dpi: int = 300) -> Image.Image:
        """Create a simple colorbar as fallback"""
        try:
            # Calculate dimensions
            mm_to_pixels = dpi / 25.4
            swatch_size_px = int(config["layout"]["swatch_size_mm"] * mm_to_pixels)
            spacing_px = int(config["layout"]["swatch_spacing_mm"] * mm_to_pixels)

            # Generate colors
            gradient_config = config["gradients"][0]
            colors = []
            steps = gradient_config["steps"]

            for i in range(steps):
                if steps == 1:
                    factor = 0
                else:
                    factor = i / (steps - 1)

                start_color = gradient_config["start_color"]
                end_color = gradient_config["end_color"]

                # 使用 strict=False 来兼容不同长度的颜色分量列表
                # 在某些情况下，CMYK 颜色可能只有 3 个分量而不是 4 个
                # strict=False 允许 zip 函数处理不等长的可迭代对象
                color = [
                    start + (end - start) * factor
                    for start, end in zip(start_color, end_color, strict=False)
                ]
                colors.append(color)

            # Create image
            image_width = len(colors) * swatch_size_px + (len(colors) - 1) * spacing_px
            image_height = swatch_size_px

            image = Image.new("RGB", (image_width, image_height), "white")
            draw = ImageDraw.Draw(image)

            # Draw color swatches
            current_x = 0
            for cmyk in colors:
                # Convert CMYK to RGB
                cmyk_array = np.array(
                    [[[cmyk[0] / 100, cmyk[1] / 100, cmyk[2] / 100, cmyk[3] / 100]]],
                    dtype=np.float32,
                )
                rgb_array = cmyk_to_rgb(cmyk_array)
                rgb_color = tuple(int(c * 255) for c in rgb_array[0, 0])

                # Draw swatch
                draw.rectangle(
                    [current_x, 0, current_x + swatch_size_px, swatch_size_px],
                    fill=rgb_color,
                )

                # Add border
                draw.rectangle(
                    [current_x, 0, current_x + swatch_size_px, swatch_size_px],
                    outline="black",
                    width=1,
                )

                current_x += swatch_size_px + spacing_px

            return image

        except Exception as e:
            print(f"Error creating simple colorbar: {e}")
            # Return a basic error image
            error_image = Image.new("RGB", (300, 50), "red")
            draw = ImageDraw.Draw(error_image)
            draw.text((10, 20), "Error generating colorbar", fill="white")
            return error_image

    def generate_all_demo_images(self) -> dict[str, Image.Image]:
        """Generate all demo colorbar images"""
        demo_images = {}

        for config in self.demo_configs:
            try:
                image = self.generate_colorbar_image(config)
                demo_images[config["name"]] = image
            except Exception as e:
                print(f"Error generating demo image for {config['name']}: {e}")
                # Create placeholder image
                placeholder = Image.new("RGB", (300, 50), "lightgray")
                draw = ImageDraw.Draw(placeholder)
                draw.text((10, 20), f"Error: {config['name']}", fill="black")
                demo_images[config["name"]] = placeholder

        return demo_images


def process_ground_truth_colorbar_analysis(
    input_image: Image.Image,
    confidence_threshold: float = 0.6,
    box_expansion: int = 10,
    block_area_threshold: int = 50,
    block_aspect_ratio: float = 0.3,
    min_square_size: int = 10,
) -> tuple[Image.Image, str, str]:
    """
    Process ground-truth colorbar analysis using pure color analysis

    Returns:
        Tuple of (annotated_image, status_message, results_html)
    """

    try:
        # Import required modules
        from core.block_detection.pure_colorbar_analysis import (
            pure_colorbar_analysis_for_gradio,
        )

        # Use pure color analysis with purity threshold for better color matching
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
            purity_threshold=0.8,  # High purity for ground truth comparison
        )

        if not colorbar_data:
            return annotated_image, "No colorbars detected", ""

        # Create concise HTML display using shared component
        results_html = update_shared_results_display(colorbar_data)

        status = f"✅ Ground truth analysis complete: {len(colorbar_data)} colorbar(s), {total_blocks} pure color blocks"

        return annotated_image, status, results_html

    except Exception as e:
        error_msg = f"❌ Error during ground truth analysis: {str(e)}"
        return input_image, error_msg, ""


def create_fixed_order_analysis_display(
    gt_results: list[dict], detected_colors: list[tuple[int, int, int]]
) -> str:
    """
    Create HTML display for fixed order ground truth comparison results

    Args:
        gt_results: Ground truth comparison results
        detected_colors: List of detected RGB colors

    Returns:
        HTML string for display
    """
    html = """
    <div style="font-family: Arial, sans-serif; max-width: 100%; margin: 0 auto;">
        <h3 style="color: #333; margin-bottom: 20px; text-align: center;">
            🎯 Fixed Order Ground Truth Comparison
        </h3>
        <div style="display: grid; gap: 10px; margin-bottom: 20px;">
    """

    # Display comparison for each color
    for i, result in enumerate(gt_results):
        detected_rgb = result["detected_rgb"]
        expected_gt = result.get("expected_ground_truth")
        delta_e = result["delta_e"]
        accuracy = result["accuracy_level"]

        # Color for accuracy level
        accuracy_color = "#28a745" if accuracy == "Acceptable" else "#dc3545"
        if accuracy == "No Reference":
            accuracy_color = "#6c757d"

        html += f"""
            <div style="display: flex; align-items: center; padding: 10px; border: 1px solid #ddd; border-radius: 8px; background: #f8f9fa;">
                <div style="width: 40px; height: 40px; background-color: rgb{detected_rgb}; border: 1px solid #333; border-radius: 4px; margin-right: 15px;"></div>
                <div style="flex: 1;">
                    <div style="font-weight: bold; color: #333;">Color {i + 1}</div>
                    <div style="color: #666; font-size: 0.9em;">
                        Detected: RGB{detected_rgb}
                    </div>
                    {f'<div style="color: #666; font-size: 0.9em;">Expected: {expected_gt["name"]} - CMYK{expected_gt["cmyk"]}</div>' if expected_gt else '<div style="color: #666; font-size: 0.9em;">No expected reference</div>'}
                </div>
                <div style="text-align: right; margin-left: 15px;">
                    <div style="font-weight: bold; color: {accuracy_color};">
                        ΔE: {delta_e:.2f}
                    </div>
                    <div style="color: {accuracy_color}; font-size: 0.9em;">
                        {accuracy}
                    </div>
                </div>
            </div>
        """

    html += """
        </div>
    </div>
    """

    return html


def create_annotated_image(
    original_image: Image.Image, analysis_results: dict, gt_results: list[dict]
) -> Image.Image:
    """
    Create annotated image showing detected colors and comparison results

    Args:
        original_image: Original input image
        analysis_results: Results from colorbar analysis
        gt_results: Ground truth comparison results

    Returns:
        Annotated PIL Image
    """
    try:
        # Create a copy of the original image
        annotated = original_image.copy()
        draw = ImageDraw.Draw(annotated)

        # Get detected regions if available
        detected_regions = analysis_results.get("detected_regions", [])

        # Draw bounding boxes and labels
        # 使用 strict=False 来处理检测区域和真实值结果数量可能不匹配的情况
        # 这样可以避免在列表长度不同时引发 ValueError
        # 在实际应用中，检测到的区域数量可能与预期的真实值数量不完全一致
        for i, (region, gt_result) in enumerate(
            zip(detected_regions, gt_results, strict=False)
        ):
            # Extract bounding box coordinates
            x1, y1, x2, y2 = region.get("bbox", [0, 0, 50, 50])

            # Color for accuracy level
            accuracy = gt_result["accuracy_level"]
            box_color = "green" if accuracy == "Acceptable" else "red"
            if accuracy == "No Reference":
                box_color = "gray"

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)

            # Draw label
            delta_e = gt_result["delta_e"]
            expected_gt = gt_result.get("expected_ground_truth")

            if expected_gt:
                label = f"{i + 1}: {expected_gt['name']}\nΔE: {delta_e:.2f}"
            else:
                label = f"{i + 1}: No Ref\nΔE: {delta_e:.2f}"

            # Draw label background
            draw.rectangle([x1, y1 - 40, x1 + 120, y1], fill=box_color)
            draw.text((x1 + 5, y1 - 35), label, fill="white")

        return annotated

    except Exception as e:
        print(f"Error creating annotated image: {e}")
        return original_image


def create_ground_truth_colorbar_demo_ui():
    """Create the Ground-Truth Colorbar Demo UI"""

    # Initialize demo generator
    demo_generator = GroundTruthColorbarDemo()

    with gr.Column():
        gr.Markdown("## 🎨 Ground-Truth Colorbar Demo & Testing")
        gr.Markdown(
            """
        This demo generates **ground-truth colorbar images** with known CMYK values and tests 
        the colorbar detection system. Select from pre-generated examples or upload your own image.
        """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Example selection
                gr.Markdown("### 📋 Pre-Generated Examples")

                example_dropdown = gr.Dropdown(
                    choices=[config["name"] for config in demo_generator.demo_configs],
                    value=demo_generator.demo_configs[0]["name"],
                    label="Select Example Colorbar",
                    info="Choose a pre-generated ground-truth colorbar",
                )

                example_description = gr.Markdown(
                    demo_generator.demo_configs[0]["description"]
                )

                generate_example_btn = gr.Button(
                    "🎨 Generate Example", variant="primary"
                )

                # Custom image upload
                gr.Markdown("### 📷 Custom Image Upload")
                custom_image = gr.Image(
                    label="Upload Custom Colorbar Image", type="pil", height=150
                )

                # Analysis parameters
                with gr.Accordion("🔧 Analysis Parameters", open=False):
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

                    with gr.Row():
                        shrink_width = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=30,
                            step=5,
                            label="Analysis Width",
                            info="Analysis region width",
                        )
                        shrink_height = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=30,
                            step=5,
                            label="Analysis Height",
                            info="Analysis region height",
                        )

                # Action buttons
                with gr.Row():
                    analyze_btn = gr.Button(
                        "🔍 Analyze Colorbar", variant="primary", scale=2
                    )
                    clear_btn = gr.Button("🧹 Clear", scale=1)

                # Status
                status_text = gr.Textbox(
                    label="Status",
                    value="Select example → Generate → Analyze",
                    interactive=False,
                    lines=1,
                )

            with gr.Column(scale=2):
                # Image display
                current_image = gr.Image(
                    label="📷 Current Colorbar Image", type="pil", height=200
                )

                # Analysis results
                result_image = gr.Image(
                    label="🎯 Analysis Results", type="pil", height=250
                )

        # Results display
        results_display = gr.HTML(
            value="<div style='text-align: center; padding: 20px; color: #666; background: #f9f9f9; border-radius: 6px;'>🎨 Generate example or upload image → Analyze to see detailed results</div>"
        )

        # Ground truth reference - moved from colorbar analysis
        with gr.Accordion("📚 Ground Truth Color Reference", open=False):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(
                        """
                    **Standard CMYK Colors for Reference:**
                    - Pure Cyan (C:100, M:0, Y:0, K:0)
                    - Pure Magenta (C:0, M:100, Y:0, K:0)
                    - Pure Yellow (C:0, M:0, Y:100, K:0)
                    - Pure Black (C:0, M:0, Y:0, K:100)
                    - Complex Color 1 (C:50, M:60, Y:0, K:50)
                    - Complex Color 2 (C:80, M:50, Y:60, K:0)
                    - Complex Color 3 (C:60, M:60, Y:60, K:0)
                    
                    **Delta E Threshold:** < 3.0 for acceptable accuracy
                    """
                    )
                with gr.Column(scale=1):
                    show_gt_reference_btn = gr.Button("📊 Show Ground Truth Chart")
                    show_gt_yaml_btn = gr.Button("📄 Show Ground Truth YAML")

            gt_reference_chart = gr.Image(
                label="Ground Truth Reference Chart", visible=False
            )

            gt_yaml_config = gr.Code(
                label="Ground Truth YAML Configuration", language="yaml", visible=False
            )

    # Event handlers
    def generate_example_colorbar(example_name):
        """Generate example colorbar from selection"""
        try:
            # Find the selected configuration
            config = None
            for demo_config in demo_generator.demo_configs:
                if demo_config["name"] == example_name:
                    config = demo_config
                    break

            if config is None:
                return None, "❌ Configuration not found"

            # Generate the colorbar image
            image = demo_generator.generate_colorbar_image(config)

            return image, f"✅ Generated example: {example_name}"

        except Exception as e:
            return None, f"❌ Error generating example: {str(e)}"

    def update_example_description(example_name):
        """Update example description"""
        for config in demo_generator.demo_configs:
            if config["name"] == example_name:
                return config["description"]
        return "Description not available"

    def run_analysis(
        image, conf, box_exp, area_thresh, aspect_ratio, min_size, shrink_w, shrink_h
    ):
        """Run colorbar analysis"""
        if image is None:
            return None, "❌ No image provided", ""

        return process_ground_truth_colorbar_analysis(
            image,
            confidence_threshold=conf,
            box_expansion=box_exp,
            block_area_threshold=area_thresh,
            block_aspect_ratio=aspect_ratio,
            min_square_size=min_size,
            shrink_size_width=shrink_w,
            shrink_size_height=shrink_h,
        )

    def clear_all():
        """Clear all inputs and outputs"""
        return (
            None,  # current_image
            None,  # result_image
            "Select example → Generate → Analyze",  # status
            "<div style='text-align: center; padding: 20px; color: #666; background: #f9f9f9; border-radius: 6px;'>🎨 Generate example or upload image → Analyze</div>",  # results_display
        )

    def show_gt_reference_chart():
        """Show ground truth reference chart"""
        try:
            reference_image = ground_truth_checker.generate_reference_chart()
            return gr.Image(value=reference_image, visible=True)
        except Exception as e:
            print(f"Error generating reference chart: {e}")
            return gr.Image(visible=False)

    def show_gt_yaml_config():
        """Show ground truth YAML configuration"""
        try:
            yaml_content = ground_truth_checker.get_palette_yaml()
            return gr.Code(value=yaml_content, visible=True)
        except Exception as e:
            print(f"Error generating YAML config: {e}")
            return gr.Code(value="# Error generating YAML config", visible=True)

    # Connect event handlers
    example_dropdown.change(
        fn=update_example_description,
        inputs=[example_dropdown],
        outputs=[example_description],
    )

    generate_example_btn.click(
        fn=generate_example_colorbar,
        inputs=[example_dropdown],
        outputs=[current_image, status_text],
    )

    custom_image.change(
        fn=lambda img: (
            img,
            "Custom image loaded → Ready to analyze"
            if img
            else "Select example → Generate → Analyze",
        ),
        inputs=[custom_image],
        outputs=[current_image, status_text],
    )

    analyze_btn.click(
        fn=run_analysis,
        inputs=[
            current_image,
            confidence_threshold,
            box_expansion,
            block_area_threshold,
            block_aspect_ratio,
            min_square_size,
            shrink_width,
            shrink_height,
        ],
        outputs=[result_image, status_text, results_display],
    )

    clear_btn.click(
        fn=clear_all,
        outputs=[current_image, result_image, status_text, results_display],
    )

    show_gt_reference_btn.click(
        fn=show_gt_reference_chart,
        outputs=[gt_reference_chart],
    )

    show_gt_yaml_btn.click(
        fn=show_gt_yaml_config,
        outputs=[gt_yaml_config],
    )

    # Set up Gradio examples
    example_images = demo_generator.generate_all_demo_images()

    # Create examples component
    examples_list = []
    for name, image in example_images.items():
        examples_list.append([image])

    if examples_list:
        gr.Examples(
            examples=examples_list,
            inputs=[current_image],
            label="🎨 Click to Load Example Images",
        )

    return {
        "current_image": current_image,
        "result_image": result_image,
        "status_text": status_text,
        "results_display": results_display,
        "example_dropdown": example_dropdown,
        "analyze_btn": analyze_btn,
        "clear_btn": clear_btn,
    }
