"""
Ground-truth CMYK Color Checker Module

Provides preset ground-truth CMYK color checker data for colorbar analysis
and delta E calculations.
"""

import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .icc_trans import cmyk_to_srgb_array
from .utils import calculate_color_difference, cmyk_to_rgb, rgb_to_lab


@dataclass
class GroundTruthColor:
    """Represents a ground-truth color with CMYK and computed RGB/LAB values"""

    id: int
    name: str
    cmyk: tuple[float, float, float, float]  # C, M, Y, K values (0-100)
    rgb: tuple[int, int, int] | None = None  # RGB values (0-255)
    lab: tuple[float, float, float] | None = None  # LAB values


class GroundTruthColorChecker:
    """
    Ground-truth CMYK color checker with preset standard colors
    """

    def __init__(self):
        self.colors = self._create_standard_colors()
        self._compute_rgb_lab_values()

    def _create_standard_colors(self) -> list[GroundTruthColor]:
        """Create standard CMYK color checker colors using the same format as the existing generator"""
        standard_colors = [
            GroundTruthColor(id=1, name="Pure Cyan", cmyk=(100, 0, 0, 0)),
            GroundTruthColor(id=2, name="Pure Magenta", cmyk=(0, 100, 0, 0)),
            GroundTruthColor(id=3, name="Pure Yellow", cmyk=(0, 0, 100, 0)),
            GroundTruthColor(id=4, name="Pure Black", cmyk=(0, 0, 0, 100)),
            GroundTruthColor(id=5, name="Complex Color 1", cmyk=(50, 60, 0, 50)),
            GroundTruthColor(id=6, name="Complex Color 2", cmyk=(80, 50, 60, 0)),
            GroundTruthColor(id=7, name="Complex Color 3", cmyk=(60, 60, 60, 0)),
        ]

        return standard_colors

    def _compute_rgb_lab_values(self):
        """Compute RGB and LAB values for all ground truth colors using ICC profiles"""
        for color in self.colors:
            # Create a 1x1 CMYK image for conversion
            cmyk_normalized = np.array([c / 100.0 for c in color.cmyk])
            cmyk_image = (cmyk_normalized * 255).astype(np.uint8).reshape(1, 1, 4)

            try:
                # Convert CMYK to RGB using ICC profiles
                rgb_array, _ = cmyk_to_srgb_array(cmyk_image)

                # Extract RGB values
                color.rgb = tuple(rgb_array[0, 0].astype(int))

                # Convert RGB to LAB using existing utility
                rgb_for_lab = rgb_array.reshape(1, 1, 3)
                lab_image = rgb_to_lab(rgb_for_lab)
                color.lab = tuple(lab_image[0, 0])

            except Exception as e:
                print(f"Error computing RGB/LAB for {color.name}: {e}")
                # Fallback to basic conversion if ICC conversion fails
                cmyk_image_fallback = cmyk_normalized.reshape(1, 1, 4)
                rgb_image = cmyk_to_rgb(cmyk_image_fallback)
                color.rgb = tuple(rgb_image[0, 0].astype(int))

                lab_image = rgb_to_lab(rgb_image)
                color.lab = tuple(lab_image[0, 0])

    def get_color_by_id(self, color_id: int) -> GroundTruthColor | None:
        """Get ground truth color by ID"""
        for color in self.colors:
            if color.id == color_id:
                return color
        return None

    def get_all_colors(self) -> list[GroundTruthColor]:
        """Get all ground truth colors"""
        return self.colors

    def find_closest_color(
        self, rgb_color: tuple[int, int, int]
    ) -> tuple[GroundTruthColor, float]:
        """
        Find the closest ground truth color to a given RGB color

        Args:
            rgb_color: RGB color tuple (0-255)

        Returns:
            Tuple of (closest_ground_truth_color, delta_e_value)
        """
        min_delta_e = float("inf")
        closest_color = None

        # Convert input RGB to image format for delta E calculation
        rgb_array = np.array([[rgb_color]], dtype=np.uint8)

        for gt_color in self.colors:
            if gt_color.rgb is None:
                continue

            # Calculate delta E using proper ICC-based color comparison
            delta_e = self._calculate_single_color_delta_e(rgb_color, gt_color)

            if delta_e < min_delta_e:
                min_delta_e = delta_e
                closest_color = gt_color

        if closest_color is None:
            # This should not happen if we have valid colors, but handle it gracefully
            if self.colors:
                return self.colors[0], float("inf")
            else:
                raise ValueError("No ground truth colors available")

        return closest_color, min_delta_e

    def calculate_delta_e_for_colors(
        self, detected_colors: list[tuple[int, int, int]]
    ) -> list[dict]:
        """
        Calculate delta E values for a list of detected colors against ground truth

        Args:
            detected_colors: List of RGB color tuples

        Returns:
            List of dictionaries containing color matching results
        """
        results = []

        for i, rgb_color in enumerate(detected_colors):
            closest_gt, delta_e = self.find_closest_color(rgb_color)

            result = {
                "detected_color_id": i,
                "detected_rgb": rgb_color,
                "closest_ground_truth": {
                    "id": closest_gt.id,
                    "name": closest_gt.name,
                    "cmyk": closest_gt.cmyk,
                    "rgb": closest_gt.rgb,
                    "lab": closest_gt.lab,
                }
                if closest_gt
                else None,
                "delta_e": delta_e,
                "accuracy_level": self._get_accuracy_level(delta_e),
            }

            results.append(result)

        return results

    def calculate_delta_e_fixed_order(
        self, detected_colors: list[tuple[int, int, int]]
    ) -> list[dict]:
        """
        Calculate delta E values for detected colors in fixed order against ground truth

        Compares 1st detected color to 1st ground truth color, 2nd to 2nd, etc.
        This is useful for colorbar analysis where the order of colors is known.

        Args:
            detected_colors: List of RGB color tuples

        Returns:
            List of dictionaries containing color matching results
        """
        results = []

        for i, rgb_color in enumerate(detected_colors):
            # Get ground truth color by position (fixed order)
            if i < len(self.colors):
                gt_color = self.colors[i]

                # Calculate delta E using proper ICC-based comparison
                delta_e = self._calculate_single_color_delta_e(rgb_color, gt_color)

                result = {
                    "detected_color_id": i,
                    "detected_rgb": rgb_color,
                    "expected_ground_truth": {
                        "id": gt_color.id,
                        "name": gt_color.name,
                        "cmyk": gt_color.cmyk,
                        "rgb": gt_color.rgb,
                        "lab": gt_color.lab,
                    },
                    "delta_e": delta_e,
                    "accuracy_level": self._get_accuracy_level(delta_e),
                }
            else:
                # More detected colors than ground truth colors
                result = {
                    "detected_color_id": i,
                    "detected_rgb": rgb_color,
                    "expected_ground_truth": None,
                    "delta_e": float("inf"),
                    "accuracy_level": "No Reference",
                }

            results.append(result)

        return results

    def _calculate_single_color_delta_e(
        self, rgb_color: tuple[int, int, int], gt_color: GroundTruthColor
    ) -> float:
        """
        Calculate delta E between two single colors using ICC-based color comparison

        Args:
            rgb_color: Detected RGB color tuple
            gt_color: Ground truth color object

        Returns:
            Delta E value
        """
        try:
            if gt_color.rgb is None:
                return float("inf")

            # Create 1x1 images for both colors for proper delta E calculation
            detected_image = np.array([[rgb_color]], dtype=np.uint8)
            gt_image = np.array([[gt_color.rgb]], dtype=np.uint8)

            # Use the existing calculate_color_difference function which uses ICC-based conversion
            delta_e, _ = calculate_color_difference(detected_image, gt_image)

            return float(delta_e)

        except Exception as e:
            print(
                f"Error calculating delta E between {rgb_color} and {gt_color.name}: {e}"
            )
            return float("inf")

    def _get_accuracy_level(self, delta_e: float) -> str:
        """
        Get accuracy level description based on delta E value
        Using delta E < 3.0 as the threshold for acceptable colors

        Args:
            delta_e: Delta E value

        Returns:
            Accuracy level string
        """
        return "Acceptable" if delta_e < 3.0 else "Poor"

    def generate_reference_chart(
        self, output_path: str | None = None
    ) -> Image.Image | None:
        """
        Generate a reference chart showing all ground truth colors using TIFF generation

        Args:
            output_path: Optional path to save the chart

        Returns:
            PIL Image of the reference chart
        """
        try:
            import tempfile

            from .palette_generator import ColorPaletteGenerator

            # Get palette configuration
            config = self.get_palette_config()

            # Create temporary files
            with tempfile.TemporaryDirectory() as temp_dir:
                tiff_path = os.path.join(temp_dir, "ground_truth_chart.tiff")
                png_path = os.path.join(temp_dir, "ground_truth_chart.png")

                # Create palette generator and generate TIFF chart
                generator = ColorPaletteGenerator()
                generator.generate_palette(
                    palette_data=config["gradients"],
                    output_path=tiff_path,
                    layout_config=config["layout"],
                    output_dpi=150,
                    generate_pdf=False,
                    generate_tiff=True,
                )

                # Convert TIFF to PNG for display
                if os.path.exists(tiff_path):
                    try:
                        # Convert CMYK TIFF to RGB PNG
                        generator.convert_cmyk_tiff_to_png(tiff_path, png_path)

                        if os.path.exists(png_path):
                            chart = Image.open(png_path)
                            if output_path:
                                chart.save(output_path)
                            return chart

                    except Exception as e:
                        print(f"Error converting TIFF to PNG: {e}")
                        # Fallback to simple grid layout
                        return self._generate_simple_reference_chart(output_path)

                # Fallback if TIFF generation failed
                return self._generate_simple_reference_chart(output_path)

        except Exception as e:
            print(f"Error generating reference chart: {e}")
            return self._generate_simple_reference_chart(output_path)

    def _generate_simple_reference_chart(
        self, output_path: str | None = None
    ) -> Image.Image | None:
        """
        Generate a simple reference chart as fallback - horizontal layout

        Args:
            output_path: Optional path to save the chart

        Returns:
            PIL Image of the reference chart
        """
        # Create a horizontal layout
        swatch_size = 80
        spacing = 10
        text_height = 40

        chart_width = len(self.colors) * swatch_size + (len(self.colors) - 1) * spacing
        chart_height = swatch_size + text_height + spacing

        # Create image
        chart = Image.new("RGB", (chart_width, chart_height), "white")

        try:
            from PIL import ImageDraw

            draw = ImageDraw.Draw(chart)
            # Draw color swatches horizontally
            for i, color in enumerate(self.colors):
                x = i * (swatch_size + spacing)
                y = 0

                # Create color swatch
                swatch_coords = [x, y, x + swatch_size, y + swatch_size]
                draw.rectangle(swatch_coords, fill=color.rgb)

                # Add text label below
                text = f"{color.name}\nC{color.cmyk[0]} M{color.cmyk[1]}\nY{color.cmyk[2]} K{color.cmyk[3]}"
                draw.text((x, y + swatch_size + 5), text, fill="black")

            if output_path:
                chart.save(output_path)

            return chart

        except Exception as e:
            print(f"Error in simple chart generation: {e}")
            return None

    def get_summary_stats(self) -> dict:
        """Get summary statistics about the ground truth colors"""
        return {
            "total_colors": len(self.colors),
            "color_names": [color.name for color in self.colors],
            "cmyk_range": {
                "c_min": min(color.cmyk[0] for color in self.colors),
                "c_max": max(color.cmyk[0] for color in self.colors),
                "m_min": min(color.cmyk[1] for color in self.colors),
                "m_max": max(color.cmyk[1] for color in self.colors),
                "y_min": min(color.cmyk[2] for color in self.colors),
                "y_max": max(color.cmyk[2] for color in self.colors),
                "k_min": min(color.cmyk[3] for color in self.colors),
                "k_max": max(color.cmyk[3] for color in self.colors),
            },
        }

    def get_palette_config(self) -> dict:
        """
        Get the ground truth colors as a palette configuration compatible with the existing generator

        Returns:
            Dictionary containing layout and gradients configuration
        """
        config = {
            "layout": {
                "swatch_size_mm": 8,
                "swatch_spacing_mm": 1,
                "group_spacing_mm": 3,
                "arrangement": "single_row",
                "columns": 7,
                "page_width_mm": 200,
                "page_height_mm": 60,
                "margin_mm": 10,
            },
            "gradients": [],
        }

        # Create single-step gradients for each ground truth color
        for color in self.colors:
            gradient = {
                "name": color.name,
                "start_color": list(color.cmyk),  # [C, M, Y, K]
                "end_color": list(color.cmyk),  # Same color for single step
                "steps": 1,
            }
            config["gradients"].append(gradient)

        return config

    def get_palette_yaml(self) -> str:
        """
        Get the ground truth colors as a YAML string compatible with the existing generator

        Returns:
            YAML string containing the ground truth color configuration
        """
        import yaml

        config = self.get_palette_config()

        # Create a nicely formatted YAML string
        yaml_str = "# Ground Truth CMYK Color Checker Configuration\n"
        yaml_str += "# Standard colors for colorbar analysis validation\n\n"
        yaml_str += yaml.dump(config, default_flow_style=False, indent=2)

        return yaml_str


# Global instance for easy access
ground_truth_checker = GroundTruthColorChecker()
