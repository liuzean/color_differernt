"""
General Color Palette Generator
Generates color palettes with configurable gradients between start and end colors
"""

import io
import os
import zipfile

from PIL import Image, ImageCms, ImageDraw
from reportlab.lib.colors import CMYKColor
from reportlab.pdfgen import canvas


class ColorPaletteGenerator:
    """General color palette generator for various color spaces"""

    def __init__(self, config: dict = None):
        """Initialize with configuration"""
        self.config = config or {}

    def generate_palette(
        self,
        palette_data: list[dict],
        output_path: str,
        layout_config: dict = None,
        output_dpi: int = 300,
        generate_pdf: bool = True,
        generate_tiff: bool = False,
    ) -> None:
        """
        Generate color palette files with new simplified configuration

        Args:
            palette_data: List of gradient configs with name, start_color, end_color, steps
            output_path: Base output path (extension will be added)
            layout_config: Layout configuration from YAML
            output_dpi: Output resolution
            generate_pdf: Generate PDF file
            generate_tiff: Generate TIFF file
        """
        # Set default layout if not provided
        if layout_config is None:
            layout_config = {
                "swatch_size_mm": 5,
                "swatch_spacing_mm": 1,
                "group_spacing_mm": 3,
                "arrangement": "single_column",
                "columns": 7,
                "swatches_per_group": 11,
            }

        # Generate color data from palette configs
        color_data = []
        index = 1

        for gradient_config in palette_data:
            gradient = self.generate_linear_gradient(
                gradient_config["start_color"],
                gradient_config["end_color"],
                gradient_config["steps"],
            )

            for color in gradient:
                color_data.append((index, color))
                index += 1

        # Remove extension from output_path to add proper extension
        base_path = os.path.splitext(output_path)[0]

        # Generate PDF if requested
        if generate_pdf:
            pdf_path = f"{base_path}.pdf"
            self.create_color_swatches_pdf(color_data, pdf_path, layout_config)

        # Generate TIFF if requested
        if generate_tiff:
            tiff_path = f"{base_path}.tiff"
            self.create_color_swatches_cmyk_tiff(
                color_data, tiff_path, layout_config, output_dpi
            )

    def generate_linear_gradient(
        self,
        start_color: tuple,
        end_color: tuple,
        steps: int,
        color_space: str = "CMYK",
    ) -> list[tuple]:
        """Generate linear gradient between two colors"""
        if len(start_color) != len(end_color):
            raise ValueError("Start and end colors must have same number of channels")

        gradient = []
        for i in range(steps):
            if steps == 1:
                factor = 0
            else:
                factor = i / (steps - 1)

            color = tuple(
                int(start + (end - start) * factor)
                for start, end in zip(start_color, end_color, strict=False)
            )
            gradient.append(color)

        return gradient

    def generate_multi_channel_palette(
        self, palette_configs: list[dict]
    ) -> list[tuple[int, tuple]]:
        """Generate palette with multiple channels varying"""
        color_data = []
        index = 1

        for config in palette_configs:
            gradient = self.generate_linear_gradient(
                config["start_color"],
                config["end_color"],
                config["steps"],
                config.get("color_space", "CMYK"),
            )

            for color in gradient:
                color_data.append((index, color))
                index += 1

        return color_data

    def cmyk_to_rgb_conversion(
        self, c: float, m: float, y: float, k: float
    ) -> tuple[int, int, int]:
        """Convert CMYK values to RGB using standard conversion formula"""
        c_norm = c / 100.0
        m_norm = m / 100.0
        y_norm = y / 100.0
        k_norm = k / 100.0

        r = 255 * (1 - c_norm) * (1 - k_norm)
        g = 255 * (1 - m_norm) * (1 - k_norm)
        b = 255 * (1 - y_norm) * (1 - k_norm)

        return (
            max(0, min(255, int(round(r)))),
            max(0, min(255, int(round(g)))),
            max(0, min(255, int(round(b)))),
        )

    def create_color_swatches_image(
        self,
        color_data: list[tuple[int, tuple]],
        filename: str,
        layout_config: dict,
        format: str = "PNG",
        dpi: int = 300,
    ) -> None:
        """Create image file with color swatches"""
        mm_to_pixels = dpi / 25.4

        swatch_size_px = int(layout_config["swatch_size_mm"] * mm_to_pixels)
        swatch_spacing_px = int(layout_config["swatch_spacing_mm"] * mm_to_pixels)
        group_spacing_px = int(layout_config["group_spacing_mm"] * mm_to_pixels)

        total_swatches = len(color_data)
        groups = layout_config.get("swatches_per_group", 11)

        if layout_config["arrangement"] == "single_column":
            image_width = swatch_size_px
            num_groups = len(color_data) // groups
            basic_height = (
                total_swatches * swatch_size_px
                + (total_swatches - 1) * swatch_spacing_px
            )
            extra_height = (num_groups - 1) * group_spacing_px
            image_height = basic_height + extra_height
        else:
            cols = layout_config.get("columns", 7)
            rows = (total_swatches + cols - 1) // cols
            image_width = cols * swatch_size_px + (cols - 1) * swatch_spacing_px
            image_height = rows * swatch_size_px + (rows - 1) * swatch_spacing_px

        image = Image.new("RGB", (image_width, image_height), "white")
        draw = ImageDraw.Draw(image)

        current_y = 0
        for i, (_index, color) in enumerate(color_data):
            if len(color) == 4:  # CMYK
                rgb_color = self.cmyk_to_rgb_conversion(*color)
            else:  # Assume RGB
                rgb_color = color

            if layout_config["arrangement"] == "single_column":
                draw.rectangle(
                    [0, current_y, swatch_size_px, current_y + swatch_size_px],
                    fill=rgb_color,
                )
                current_y += swatch_size_px + swatch_spacing_px

                if (i + 1) % groups == 0 and i < len(color_data) - 1:
                    current_y += group_spacing_px
            else:
                row = i // layout_config.get("columns", 7)
                col = i % layout_config.get("columns", 7)
                x = col * (swatch_size_px + swatch_spacing_px)
                y = row * (swatch_size_px + swatch_spacing_px)

                draw.rectangle(
                    [x, y, x + swatch_size_px, y + swatch_size_px], fill=rgb_color
                )

        image.save(filename, format=format, dpi=(dpi, dpi))

    def create_color_swatches_cmyk_tiff(
        self,
        color_data: list[tuple[int, tuple]],
        filename: str,
        layout_config: dict,
        dpi: int = 300,
    ) -> None:
        """Create CMYK TIFF with ICC profile"""
        mm_to_pixels = dpi / 25.4

        swatch_size_px = int(layout_config["swatch_size_mm"] * mm_to_pixels)
        swatch_spacing_px = int(layout_config["swatch_spacing_mm"] * mm_to_pixels)
        group_spacing_px = int(layout_config["group_spacing_mm"] * mm_to_pixels)

        total_swatches = len(color_data)
        groups = layout_config.get("swatches_per_group", 11)

        if layout_config["arrangement"] == "single_column":
            image_width = swatch_size_px
            num_groups = len(color_data) // groups
            basic_height = (
                total_swatches * swatch_size_px
                + (total_swatches - 1) * swatch_spacing_px
            )
            extra_height = (num_groups - 1) * group_spacing_px
            image_height = basic_height + extra_height
        else:
            cols = layout_config.get("columns", 7)
            rows = (total_swatches + cols - 1) // cols
            image_width = cols * swatch_size_px + (cols - 1) * swatch_spacing_px
            image_height = rows * swatch_size_px + (rows - 1) * swatch_spacing_px

        image = Image.new("CMYK", (image_width, image_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        current_y = 0
        for i, (_index, color) in enumerate(color_data):
            if len(color) == 4:  # CMYK
                c_255 = int(color[0] * 255 / 100)
                m_255 = int(color[1] * 255 / 100)
                y_255 = int(color[2] * 255 / 100)
                k_255 = int(color[3] * 255 / 100)
                cmyk_color = (c_255, m_255, y_255, k_255)
            else:
                cmyk_color = color

            if layout_config["arrangement"] == "single_column":
                draw.rectangle(
                    [0, current_y, swatch_size_px, current_y + swatch_size_px],
                    fill=cmyk_color,
                )
                current_y += swatch_size_px + swatch_spacing_px

                if (i + 1) % groups == 0 and i < len(color_data) - 1:
                    current_y += group_spacing_px
            else:
                row = i // layout_config.get("columns", 7)
                col = i % layout_config.get("columns", 7)
                x = col * (swatch_size_px + swatch_spacing_px)
                y = row * (swatch_size_px + swatch_spacing_px)

                draw.rectangle(
                    [x, y, x + swatch_size_px, y + swatch_size_px], fill=cmyk_color
                )

        try:
            icc_profile_path = os.path.join(
                os.path.dirname(__file__), "icc", "JapanColor2001Coated.icc"
            )
            if os.path.exists(icc_profile_path):
                with open(icc_profile_path, "rb") as f:
                    icc_profile = f.read()
                image.save(
                    filename, format="TIFF", dpi=(dpi, dpi), icc_profile=icc_profile
                )
            else:
                image.save(filename, format="TIFF", dpi=(dpi, dpi))
        except Exception as e:
            print(f"Error saving CMYK TIFF: {e}")
            image.save(filename, format="TIFF", dpi=(dpi, dpi))

    def create_color_swatches_pdf(
        self, color_data: list[tuple[int, tuple]], filename: str, layout_config: dict
    ) -> None:
        """Create PDF with color swatches matching TIFF layout exactly"""
        # Use the same logic as TIFF generation for consistent layout
        mm_to_points = 72 / 25.4  # Convert mm to points (PDF uses points)

        swatch_size_pts = layout_config["swatch_size_mm"] * mm_to_points
        swatch_spacing_pts = layout_config["swatch_spacing_mm"] * mm_to_points
        group_spacing_pts = layout_config["group_spacing_mm"] * mm_to_points

        total_swatches = len(color_data)
        groups = layout_config.get("swatches_per_group", 11)

        # Calculate page dimensions exactly like TIFF
        if layout_config["arrangement"] == "single_column":
            page_width = swatch_size_pts + 20  # Small margin
            num_groups = len(color_data) // groups
            basic_height = (
                total_swatches * swatch_size_pts
                + (total_swatches - 1) * swatch_spacing_pts
            )
            extra_height = (num_groups - 1) * group_spacing_pts
            page_height = basic_height + extra_height + 20  # Small margin
        else:  # grid layout
            cols = layout_config.get("columns", 7)
            rows = (total_swatches + cols - 1) // cols
            page_width = cols * swatch_size_pts + (cols - 1) * swatch_spacing_pts + 20
            page_height = rows * swatch_size_pts + (rows - 1) * swatch_spacing_pts + 20

        c = canvas.Canvas(filename, pagesize=(page_width, page_height))

        # Start from bottom-left corner (PDF coordinate system)
        margin = 10
        current_y = page_height - margin - swatch_size_pts

        for i, (_index, color) in enumerate(color_data):
            if len(color) == 4:  # CMYK
                cmyk_color = CMYKColor(
                    color[0] / 100.0,
                    color[1] / 100.0,
                    color[2] / 100.0,
                    color[3] / 100.0,
                )
            else:
                cmyk_color = CMYKColor(0, 0, 0, 0)

            c.setFillColor(cmyk_color)

            if layout_config["arrangement"] == "single_column":
                # Single column layout matching TIFF
                c.rect(
                    margin,
                    current_y,
                    swatch_size_pts,
                    swatch_size_pts,
                    fill=1,
                    stroke=0,
                )
                current_y -= swatch_size_pts + swatch_spacing_pts

                # Add group spacing
                if (i + 1) % groups == 0 and i < len(color_data) - 1:
                    current_y -= group_spacing_pts
            else:  # grid layout
                # Grid layout matching TIFF
                row = i // layout_config.get("columns", 7)
                col = i % layout_config.get("columns", 7)
                x = margin + col * (swatch_size_pts + swatch_spacing_pts)
                y = (
                    page_height
                    - margin
                    - swatch_size_pts
                    - row * (swatch_size_pts + swatch_spacing_pts)
                )

                c.rect(x, y, swatch_size_pts, swatch_size_pts, fill=1, stroke=0)

        c.save()

    def convert_cmyk_tiff_to_png(
        self, cmyk_tiff_path: str, png_output_path: str
    ) -> None:
        """Convert CMYK TIFF with ICC profile to RGB PNG using proper color profile conversion"""
        try:
            cmyk_image = Image.open(cmyk_tiff_path)

            if "icc_profile" in cmyk_image.info:
                input_profile = ImageCms.ImageCmsProfile(
                    io.BytesIO(cmyk_image.info["icc_profile"])
                )

                srgb_profile_path = os.path.join(
                    os.path.dirname(__file__), "icc", "sRGB IEC61966-21.icc"
                )

                if os.path.exists(srgb_profile_path):
                    output_profile = ImageCms.ImageCmsProfile(srgb_profile_path)
                else:
                    output_profile = ImageCms.createProfile("sRGB")

                # Try different rendering intent constants for compatibility
                try:
                    rendering_intent = ImageCms.INTENT_PERCEPTUAL
                except AttributeError:
                    try:
                        rendering_intent = ImageCms.Intent.PERCEPTUAL
                    except (AttributeError, NameError):
                        rendering_intent = 0  # Fallback to perceptual (0)

                transform = ImageCms.buildTransformFromOpenProfiles(
                    input_profile,
                    output_profile,
                    "CMYK",
                    "RGB",
                    renderingIntent=rendering_intent,
                )

                rgb_image = ImageCms.applyTransform(cmyk_image, transform)

                if "dpi" in cmyk_image.info:
                    dpi = cmyk_image.info["dpi"]
                    rgb_image.save(png_output_path, format="PNG", dpi=dpi)
                else:
                    rgb_image.save(png_output_path, format="PNG", dpi=(300, 300))

            else:
                rgb_image = cmyk_image.convert("RGB")
                if "dpi" in cmyk_image.info:
                    dpi = cmyk_image.info["dpi"]
                    rgb_image.save(png_output_path, format="PNG", dpi=dpi)
                else:
                    rgb_image.save(png_output_path, format="PNG", dpi=(300, 300))

        except Exception as e:
            print(f"Error converting CMYK TIFF to PNG: {e}")
            raise

    def create_individual_color_tiffs(
        self,
        color_data: list[tuple[int, tuple]],
        output_dir: str,
        base_filename: str,
        dpi: int = 300,
    ) -> list[str]:
        """Create individual TIFF files for each color and return list of file paths"""
        generated_files = []
        swatch_size_px = int(5 * dpi / 25.4)  # 5mm standard size

        os.makedirs(output_dir, exist_ok=True)

        for index, color in color_data:
            # Create individual color image
            image = Image.new("CMYK", (swatch_size_px, swatch_size_px), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)

            if len(color) == 4:  # CMYK
                c_255 = int(color[0] * 255 / 100)
                m_255 = int(color[1] * 255 / 100)
                y_255 = int(color[2] * 255 / 100)
                k_255 = int(color[3] * 255 / 100)
                cmyk_color = (c_255, m_255, y_255, k_255)
            else:
                cmyk_color = color

            # Fill the entire image with the color
            draw.rectangle([0, 0, swatch_size_px, swatch_size_px], fill=cmyk_color)

            # Save individual TIFF file
            filename = f"{base_filename}_color_{index:03d}.tiff"
            filepath = os.path.join(output_dir, filename)

            try:
                # Try to save with ICC profile
                icc_profile_path = os.path.join(
                    os.path.dirname(__file__), "icc", "JapanColor2001Coated.icc"
                )
                if os.path.exists(icc_profile_path):
                    with open(icc_profile_path, "rb") as f:
                        icc_profile = f.read()
                    image.save(
                        filepath, format="TIFF", dpi=(dpi, dpi), icc_profile=icc_profile
                    )
                else:
                    image.save(filepath, format="TIFF", dpi=(dpi, dpi))

                generated_files.append(filepath)

            except Exception as e:
                print(f"Error saving individual TIFF {filename}: {e}")

        return generated_files

    def create_color_palette_zip(
        self,
        palette_configs: list[dict],
        output_dir: str,
        zip_filename: str,
        dpi: int = 300,
    ) -> str:
        """Create a zip file containing gradient-based and step-based TIFF files"""
        zip_path = os.path.join(output_dir, zip_filename)
        temp_dir = os.path.join(output_dir, "temp_zip_files")
        os.makedirs(temp_dir, exist_ok=True)

        # Default layout configuration for individual gradient TIFFs
        layout_config = {
            "swatch_size_mm": 5,
            "swatch_spacing_mm": 1,
            "group_spacing_mm": 3,
            "arrangement": "single_column",
            "swatches_per_group": 11,
            "columns": 7,
            "page_width_mm": 60,
            "page_height_mm": 550,
            "margin_mm": 20,
        }

        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                # 1. Create gradient-based images (one per gradient group)
                for palette_config in palette_configs:
                    # Generate gradient data for this palette
                    gradient = self.generate_linear_gradient(
                        palette_config["start_color"],
                        palette_config["end_color"],
                        palette_config["steps"],
                        palette_config.get("color_space", "CMYK"),
                    )

                    # Create color data with proper indexing
                    gradient_data = [(i + 1, color) for i, color in enumerate(gradient)]

                    # Create filename based on gradient name
                    gradient_name = (
                        palette_config["name"]
                        .lower()
                        .replace(" ", "-")
                        .replace("(", "")
                        .replace(")", "")
                        .replace("+", "-")
                    )
                    filename = f"{gradient_name}.tiff"
                    filepath = os.path.join(temp_dir, filename)

                    # Create TIFF file for this gradient group
                    self.create_color_swatches_cmyk_tiff(
                        gradient_data, filepath, layout_config, dpi
                    )

                    if os.path.exists(filepath):
                        zipf.write(filepath, f"gradients/{filename}")
                        print(f"Added gradient TIFF file: {filename}")

                    # Also create PDF file for this gradient group
                    pdf_filename = f"{gradient_name}.pdf"
                    pdf_filepath = os.path.join(temp_dir, pdf_filename)
                    self.create_color_swatches_pdf(
                        gradient_data, pdf_filepath, layout_config
                    )

                    if os.path.exists(pdf_filepath):
                        zipf.write(pdf_filepath, f"gradients/{pdf_filename}")
                        print(f"Added gradient PDF file: {pdf_filename}")

                # 2. Create step-based images (one per step, containing that step from all gradients)
                if palette_configs:
                    max_steps = max(config["steps"] for config in palette_configs)

                    for step in range(1, max_steps + 1):
                        step_colors = []
                        valid_gradients = 0

                        for palette_config in palette_configs:
                            if step <= palette_config["steps"]:
                                # Calculate the color for this step
                                if palette_config["steps"] == 1:
                                    factor = 0
                                else:
                                    factor = (step - 1) / (palette_config["steps"] - 1)

                                start_color = palette_config["start_color"]
                                end_color = palette_config["end_color"]

                                color = tuple(
                                    int(start + (end - start) * factor)
                                    for start, end in zip(
                                        start_color, end_color, strict=False
                                    )
                                )

                                step_colors.append((valid_gradients + 1, color))
                                valid_gradients += 1

                        if step_colors:
                            # Create step-based layout (horizontal arrangement)
                            step_layout_config = layout_config.copy()
                            step_layout_config["arrangement"] = "grid"
                            step_layout_config["columns"] = len(step_colors)

                            step_filename = f"step-{step:02d}.tiff"
                            step_filepath = os.path.join(temp_dir, step_filename)

                            self.create_color_swatches_cmyk_tiff(
                                step_colors, step_filepath, step_layout_config, dpi
                            )

                            if os.path.exists(step_filepath):
                                zipf.write(step_filepath, f"steps/{step_filename}")
                                print(f"Added step TIFF file: {step_filename}")

                            # Also create PDF file for this step
                            step_pdf_filename = f"step-{step:02d}.pdf"
                            step_pdf_filepath = os.path.join(
                                temp_dir, step_pdf_filename
                            )
                            self.create_color_swatches_pdf(
                                step_colors, step_pdf_filepath, step_layout_config
                            )

                            if os.path.exists(step_pdf_filepath):
                                zipf.write(
                                    step_pdf_filepath, f"steps/{step_pdf_filename}"
                                )
                                print(f"Added step PDF file: {step_pdf_filename}")

                # Clean up temporary files
                for file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)

        except Exception as e:
            print(f"Error creating ZIP file: {e}")
            # Clean up on error
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    try:
                        os.remove(os.path.join(temp_dir, file))
                    except:
                        pass
                try:
                    os.rmdir(temp_dir)
                except:
                    pass
            raise

        return zip_path

    def create_palette_preview_tiff(
        self, color_data: list[tuple[int, tuple]], output_path: str, dpi: int = 300
    ) -> str:
        """Create a TIFF preview of the color palette"""
        mm_to_pixels = dpi / 25.4
        swatch_size_px = int(5 * mm_to_pixels)  # 5mm swatches
        swatch_spacing_px = int(1 * mm_to_pixels)  # 1mm spacing

        # Calculate layout for preview - horizontal strip
        image_width = (
            len(color_data) * swatch_size_px + (len(color_data) - 1) * swatch_spacing_px
        )
        image_height = swatch_size_px

        image = Image.new("CMYK", (image_width, image_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        current_x = 0
        for _index, color in color_data:
            if len(color) == 4:  # CMYK
                c_255 = int(color[0] * 255 / 100)
                m_255 = int(color[1] * 255 / 100)
                y_255 = int(color[2] * 255 / 100)
                k_255 = int(color[3] * 255 / 100)
                cmyk_color = (c_255, m_255, y_255, k_255)
            else:
                cmyk_color = color

            draw.rectangle(
                [current_x, 0, current_x + swatch_size_px, swatch_size_px],
                fill=cmyk_color,
            )
            current_x += swatch_size_px + swatch_spacing_px

        # Save as TIFF
        try:
            icc_profile_path = os.path.join(
                os.path.dirname(__file__), "icc", "JapanColor2001Coated.icc"
            )
            if os.path.exists(icc_profile_path):
                with open(icc_profile_path, "rb") as f:
                    icc_profile = f.read()
                image.save(
                    output_path, format="TIFF", dpi=(dpi, dpi), icc_profile=icc_profile
                )
            else:
                image.save(output_path, format="TIFF", dpi=(dpi, dpi))
        except Exception as e:
            print(f"Error saving preview TIFF: {e}")

        return output_path

    def create_individual_gradient_tiffs(
        self,
        palette_configs: list[dict],
        output_dir: str,
        layout_config: dict,
        dpi: int = 300,
    ) -> list[str]:
        """Create individual TIFF files for each color gradient group (e.g., cyan-gradient.tiff)"""
        generated_files = []
        os.makedirs(output_dir, exist_ok=True)

        for palette_config in palette_configs:
            # Generate gradient data for this palette
            gradient = self.generate_linear_gradient(
                palette_config["start_color"],
                palette_config["end_color"],
                palette_config["steps"],
                palette_config.get("color_space", "CMYK"),
            )

            # Create color data with proper indexing
            gradient_data = [(i + 1, color) for i, color in enumerate(gradient)]

            # Create filename based on gradient name
            gradient_name = (
                palette_config["name"]
                .lower()
                .replace(" ", "-")
                .replace("(", "")
                .replace(")", "")
                .replace("+", "-")
            )
            filename = f"{gradient_name}.tiff"
            filepath = os.path.join(output_dir, filename)

            # Create TIFF file for this gradient group
            try:
                self.create_color_swatches_cmyk_tiff(
                    gradient_data, filepath, layout_config, dpi
                )

                if os.path.exists(filepath):
                    generated_files.append(filepath)
                    print(f"Generated gradient file: {filename}")

            except Exception as e:
                print(f"Error creating gradient TIFF {filename}: {e}")

        return generated_files
