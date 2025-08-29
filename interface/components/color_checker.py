"""
Enhanced CMYK Color Checker Generation Component
Features:
- RAR compression with organized folder structure
- Live preview showing first step and first gradient
- Configuration saving and loading
- Documentation for configurable parameters
"""

import os
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path

import gradio as gr

from core.color.palette_configs import CMYKConfigManager
from core.color.palette_generator import ColorPaletteGenerator


def create_color_checker_ui():
    """Create enhanced CMYK color checker UI with documentation and improved features"""

    config_manager = CMYKConfigManager()

    with gr.Column():
        gr.Markdown("## CMYK Color Checker Generator")

        # Documentation section
        with gr.Accordion("📚 Configuration Documentation", open=False):
            gr.Markdown(
                """
            ### Configurable Parameters

            **Layout Settings:**
            - `swatch_size_mm`: Size of each color swatch in millimeters (3-10mm recommended)
            - `swatch_spacing_mm`: Space between swatches in millimeters (0.5-2mm)
            - `group_spacing_mm`: Extra space between gradient groups (2-5mm)
            - `arrangement`: Layout style - "single_column" or "grid"
            - `columns`: Number of columns for grid layout (3-10)
            - `page_width_mm`/`page_height_mm`: Page dimensions in millimeters
            - `margin_mm`: Page margins in millimeters

            **Gradient Settings:**
            - `name`: Descriptive name for the gradient
            - `start`: Starting CMYK color as [C, M, Y, K] percentages (0-100)
            - `end`: Ending CMYK color as [C, M, Y, K] percentages (0-100)
            - `steps`: Number of color steps in the gradient (2-20)

            **Color Space Notes:**
            - CMYK values range from 0-100 (percentages)
            - [100, 0, 0, 0] = Pure Cyan
            - [0, 100, 0, 0] = Pure Magenta
            - [0, 0, 100, 0] = Pure Yellow
            - [0, 0, 0, 100] = Pure Black
            - [0, 0, 0, 0] = White (no ink)

            **Output Options:**
            - Generated files are organized by gradient and step
            - PDF files for vector output
            - TIFF files with CMYK color profiles for printing
            - RAR/ZIP compression for easy distribution
            """
            )

        # Configuration Section
        with gr.Group():
            gr.Markdown("### Configuration")
            with gr.Row():
                preset_selector = gr.Dropdown(
                    choices=config_manager.get_preset_names(),
                    value="cmyk_basic",
                    label="Preset",
                    scale=2,
                )
                load_preset_btn = gr.Button("Load", size="sm", scale=1)
                save_config_btn = gr.Button(
                    "Save", size="sm", variant="secondary", scale=1
                )

            yaml_editor = gr.Code(
                value=config_manager.get_yaml_string(),
                language="yaml",
                label="CMYK Configuration",
                lines=8,
                max_lines=12,
            )

            validation_status = gr.Textbox(
                label="Configuration Status", interactive=False, lines=1, max_lines=2
            )

        # Preview and Generation Section
        with gr.Row():
            # Live Preview (left side)
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### Live Preview")
                    gr.Markdown("*First step of first gradient*")

                    auto_preview_checkbox = gr.Checkbox(
                        label="Auto-preview on changes",
                        value=True,
                        info="Updates preview when config changes",
                    )

                    pdf_preview = gr.File(
                        label="PDF Preview", file_types=[".pdf"], height=200
                    )

                    preview_btn = gr.Button(
                        "Manual Preview", variant="secondary", size="sm"
                    )

            # Generation Controls (right side)
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### Generation")

                    compression_format = gr.Radio(
                        choices=["ZIP", "RAR"],
                        value="ZIP",
                        label="Archive Format",
                        info="RAR requires system command",
                    )

                    with gr.Row():
                        include_individual = gr.Checkbox(
                            label="Individual gradients", value=True, scale=1
                        )

                        include_steps = gr.Checkbox(
                            label="Step files", value=True, scale=1
                        )

                    generate_btn = gr.Button(
                        "Generate Archive", variant="primary", size="lg"
                    )

        # Output Section
        with gr.Group():
            gr.Markdown("### Output")
            output_files = gr.File(label="Generated Archive", file_count="single")
            status_text = gr.Textbox(
                label="Status", interactive=False, lines=2, max_lines=4
            )

        def create_output_dir():
            """Create output directory and clean old files"""
            output_dir = "temp_output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Clean files older than 1 hour to prevent accumulation
            current_time = datetime.now().timestamp()
            for filename in os.listdir(output_dir):
                filepath = os.path.join(output_dir, filename)
                if os.path.isfile(filepath):
                    file_age = current_time - os.path.getmtime(filepath)
                    if file_age > 3600:  # 1 hour
                        try:
                            os.remove(filepath)
                        except:
                            pass

            return output_dir

        def save_configuration(yaml_content):
            """Save current configuration to file"""
            try:
                success, message = config_manager.load_from_yaml(yaml_content)
                if not success:
                    return f"Cannot save invalid configuration: {message}"

                # Create configs directory if it doesn't exist
                config_dir = Path("saved_configs")
                config_dir.mkdir(exist_ok=True)

                # Save with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                config_path = config_dir / f"cmyk_config_{timestamp}.yaml"

                with open(config_path, "w") as f:
                    f.write(yaml_content)

                return f"Configuration saved to {config_path}"

            except Exception as e:
                return f"Error saving configuration: {str(e)}"

        def load_preset_handler(preset_name):
            """Load preset configuration"""
            yaml_content = config_manager.load_preset(preset_name)
            success, message = config_manager.load_from_yaml(yaml_content)
            return yaml_content, message

        def validate_yaml_handler(yaml_content):
            """Validate YAML configuration"""
            success, message = config_manager.load_from_yaml(yaml_content)
            return message

        def create_first_step_preview(yaml_content):
            """Create preview showing first step of first gradient"""
            try:
                success, message = config_manager.load_from_yaml(yaml_content)
                if not success:
                    return None, f"❌ Config error: {message}"

                palette_data = config_manager.get_palette_data()
                layout_config = config_manager.get_layout_config()

                if not palette_data:
                    return None, "❌ No gradients defined"

                # Get first gradient and create single-step preview
                first_gradient = palette_data[0]

                # Create preview with just the first color (step 1) of first gradient
                first_step_data = [
                    {
                        "name": f"{first_gradient['name']} - Preview",
                        "start_color": first_gradient["start_color"],
                        "end_color": first_gradient[
                            "start_color"
                        ],  # Same color for single swatch
                        "steps": 1,
                    }
                ]

                # Use compact layout for preview
                preview_layout = layout_config.copy()
                preview_layout["swatch_size_mm"] = min(
                    preview_layout.get("swatch_size_mm", 5), 8
                )
                preview_layout["arrangement"] = "single_column"

                generator = ColorPaletteGenerator()
                output_dir = create_output_dir()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                output_path = os.path.join(output_dir, f"preview_{timestamp}.pdf")

                generator.generate_palette(
                    palette_data=first_step_data,
                    output_path=output_path,
                    layout_config=preview_layout,
                    output_dpi=150,  # Lower DPI for faster preview
                    generate_pdf=True,
                    generate_tiff=False,
                )

                if os.path.exists(output_path):
                    return output_path, f"✓ Preview: {first_gradient['name']} (Step 1)"
                else:
                    return None, "❌ Preview generation failed"

            except Exception as e:
                return None, f"❌ Preview error: {str(e)}"

        def check_rar_availability():
            """Check if RAR command is available"""
            try:
                subprocess.run(["rar"], capture_output=True, check=False)
                return True
            except FileNotFoundError:
                return False

        def create_rar_archive(source_dir, archive_path):
            """Create RAR archive using system command"""
            try:
                cmd = ["rar", "a", "-r", archive_path, f"{source_dir}/*"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                return result.returncode == 0, result.stderr
            except Exception as e:
                return False, str(e)

        def create_zip_archive(source_dir, archive_path):
            """Create ZIP archive"""
            try:
                with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for root, _dirs, files in os.walk(source_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arc_path = os.path.relpath(file_path, source_dir)
                            zipf.write(file_path, arc_path)
                return True, "ZIP archive created successfully"
            except Exception as e:
                return False, str(e)

        def generate_handler(
            yaml_content, compression_format, include_individual, include_steps
        ):
            """Generate organized archive with all files"""
            try:
                success, message = config_manager.load_from_yaml(yaml_content)
                if not success:
                    return None, f"Validation failed: {message}"

                generator = ColorPaletteGenerator()
                palette_data = config_manager.get_palette_data()
                layout_config = config_manager.get_layout_config()

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = create_output_dir()

                # Create organized directory structure
                work_dir = os.path.join(output_dir, f"cmyk_checker_{timestamp}")
                gradients_dir = os.path.join(work_dir, "gradients")
                steps_dir = os.path.join(work_dir, "steps")
                config_dir = os.path.join(work_dir, "configuration")

                os.makedirs(gradients_dir, exist_ok=True)
                os.makedirs(steps_dir, exist_ok=True)
                os.makedirs(config_dir, exist_ok=True)

                # Save configuration
                config_path = os.path.join(config_dir, "configuration.yaml")
                with open(config_path, "w") as f:
                    f.write(yaml_content)

                files_created = 0

                # Generate individual gradient files
                if include_individual:
                    for gradient in palette_data:
                        name = (
                            gradient["name"]
                            .lower()
                            .replace(" ", "-")
                            .replace("(", "")
                            .replace(")", "")
                        )

                        # PDF
                        pdf_path = os.path.join(gradients_dir, f"{name}.pdf")
                        generator.generate_palette(
                            palette_data=[gradient],
                            output_path=pdf_path,
                            layout_config=layout_config,
                            output_dpi=300,
                            generate_pdf=True,
                            generate_tiff=False,
                        )
                        if os.path.exists(pdf_path):
                            files_created += 1

                        # TIFF
                        tiff_path = os.path.join(gradients_dir, f"{name}.tiff")
                        generator.generate_palette(
                            palette_data=[gradient],
                            output_path=tiff_path,
                            layout_config=layout_config,
                            output_dpi=300,
                            generate_pdf=False,
                            generate_tiff=True,
                        )
                        if os.path.exists(tiff_path):
                            files_created += 1

                # Generate step-by-step files
                if include_steps:
                    max_steps = max(g["steps"] for g in palette_data)
                    for step in range(1, max_steps + 1):
                        step_colors = []
                        for gradient in palette_data:
                            if step <= gradient["steps"]:
                                # Calculate color for this step
                                start = gradient["start_color"]
                                end = gradient["end_color"]
                                ratio = (
                                    (step - 1) / (gradient["steps"] - 1)
                                    if gradient["steps"] > 1
                                    else 0
                                )
                                step_color = tuple(
                                    start[i] + (end[i] - start[i]) * ratio
                                    for i in range(4)
                                )
                                step_colors.append(
                                    {
                                        "name": f"{gradient['name']} Step {step}",
                                        "start_color": step_color,
                                        "end_color": step_color,
                                        "steps": 1,
                                    }
                                )

                        if step_colors:
                            # PDF
                            step_pdf = os.path.join(steps_dir, f"step-{step:02d}.pdf")
                            generator.generate_palette(
                                palette_data=step_colors,
                                output_path=step_pdf,
                                layout_config=layout_config,
                                output_dpi=300,
                                generate_pdf=True,
                                generate_tiff=False,
                            )
                            if os.path.exists(step_pdf):
                                files_created += 1

                            # TIFF
                            step_tiff = os.path.join(steps_dir, f"step-{step:02d}.tiff")
                            generator.generate_palette(
                                palette_data=step_colors,
                                output_path=step_tiff,
                                layout_config=layout_config,
                                output_dpi=300,
                                generate_pdf=False,
                                generate_tiff=True,
                            )
                            if os.path.exists(step_tiff):
                                files_created += 1

                # Create archive
                if compression_format == "RAR" and check_rar_availability():
                    archive_path = os.path.join(
                        output_dir, f"cmyk_checker_{timestamp}.rar"
                    )
                    success, error_msg = create_rar_archive(work_dir, archive_path)
                    if not success:
                        # Fallback to ZIP
                        archive_path = os.path.join(
                            output_dir, f"cmyk_checker_{timestamp}.zip"
                        )
                        success, error_msg = create_zip_archive(work_dir, archive_path)
                        format_used = "ZIP (RAR failed)"
                    else:
                        format_used = "RAR"
                else:
                    archive_path = os.path.join(
                        output_dir, f"cmyk_checker_{timestamp}.zip"
                    )
                    success, error_msg = create_zip_archive(work_dir, archive_path)
                    format_used = "ZIP"

                # Clean up temporary directory
                shutil.rmtree(work_dir)

                if success and os.path.exists(archive_path):
                    return (
                        archive_path,
                        f"✓ Generated {files_created} files in {format_used} archive\n"
                        f"📁 Structure: gradients/, steps/, configuration/\n"
                        f"📄 Includes configuration file for reference",
                    )
                else:
                    return None, f"Failed to create archive: {error_msg}"

            except Exception as e:
                return None, f"Generation error: {str(e)}"

        # Event handlers
        load_preset_btn.click(
            load_preset_handler,
            inputs=[preset_selector],
            outputs=[yaml_editor, validation_status],
        )

        save_config_btn.click(
            save_configuration,
            inputs=[yaml_editor],
            outputs=[status_text],
        )

        yaml_editor.change(
            validate_yaml_handler, inputs=[yaml_editor], outputs=[validation_status]
        )

        # Auto-preview when configuration changes
        def auto_preview_handler(yaml_content, auto_preview):
            if auto_preview:
                return create_first_step_preview(yaml_content)
            return gr.update(), "Auto-preview disabled"

        yaml_editor.change(
            auto_preview_handler,
            inputs=[yaml_editor, auto_preview_checkbox],
            outputs=[pdf_preview, status_text],
        )

        preview_btn.click(
            create_first_step_preview,
            inputs=[yaml_editor],
            outputs=[pdf_preview, status_text],
        )

        generate_btn.click(
            generate_handler,
            inputs=[yaml_editor, compression_format, include_individual, include_steps],
            outputs=[output_files, status_text],
        )

    return (
        preset_selector,
        load_preset_btn,
        save_config_btn,
        yaml_editor,
        validation_status,
        auto_preview_checkbox,
        pdf_preview,
        preview_btn,
        compression_format,
        include_individual,
        include_steps,
        generate_btn,
        output_files,
        status_text,
    )
