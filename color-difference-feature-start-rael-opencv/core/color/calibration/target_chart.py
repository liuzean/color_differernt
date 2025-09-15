#!/usr/bin/env python

"""
CMYK Target Chart Generation

This module creates digital CMYK target charts with comprehensive color patches
for printer calibration following ICC workflow standards.
"""

import json

import cv2
import numpy as np

from ..utils import cmyk_to_rgb


class CMYKTargetChart:
    """
    Generates CMYK target charts for printer calibration

    Creates standardized color charts with patches for:
    - 100% Cyan, Magenta, Yellow, and Black primaries
    - Gradations of each ink (10%, 20%, ..., 90%)
    - Overprints (combinations of 100% inks)
    - Gray scales using K-only and composite CMY
    - Various CMYK combinations to sample printer gamut
    """

    def __init__(self, patch_size: int = 50, grid_spacing: int = 10):
        """
        Initialize target chart generator

        Args:
            patch_size: Size of each color patch in pixels
            grid_spacing: Spacing between patches in pixels
        """
        self.patch_size = patch_size
        self.grid_spacing = grid_spacing
        self.patches: list[tuple[float, float, float, float]] = []
        self.patch_positions: list[tuple[int, int, int, int]] = []
        self.patch_labels: list[str] = []

    def generate_primary_patches(self) -> list[tuple[float, float, float, float]]:
        """
        Generate 100% primary color patches

        Returns:
            List of CMYK tuples for primary colors (C, M, Y, K at 100%)
        """
        primaries = [
            (1.0, 0.0, 0.0, 0.0),  # 100% Cyan
            (0.0, 1.0, 0.0, 0.0),  # 100% Magenta
            (0.0, 0.0, 1.0, 0.0),  # 100% Yellow
            (0.0, 0.0, 0.0, 1.0),  # 100% Black
        ]
        return primaries

    def generate_gradation_patches(
        self, steps: int = 10
    ) -> list[tuple[float, float, float, float]]:
        """
        Generate gradation patches for each ink (10%, 20%, ..., 90%)

        Args:
            steps: Number of gradation steps (default creates 10%, 20%, ..., 90%)

        Returns:
            List of CMYK tuples for gradations
        """
        gradations = []

        for channel in range(4):  # C, M, Y, K
            for step in range(1, steps):
                value = step / steps
                cmyk = [0.0, 0.0, 0.0, 0.0]
                cmyk[channel] = value
                gradations.append(tuple(cmyk))

        return gradations

    def generate_overprint_patches(self) -> list[tuple[float, float, float, float]]:
        """
        Generate overprint patches (combinations of 100% inks)

        These patches test how inks interact when printed on top of each other,
        which is crucial for accurate color reproduction.

        Returns:
            List of CMYK tuples for overprints
        """
        overprints = [
            # Two-color overprints
            (1.0, 1.0, 0.0, 0.0),  # C + M (Blue)
            (1.0, 0.0, 1.0, 0.0),  # C + Y (Green)
            (1.0, 0.0, 0.0, 1.0),  # C + K
            (0.0, 1.0, 1.0, 0.0),  # M + Y (Red)
            (0.0, 1.0, 0.0, 1.0),  # M + K
            (0.0, 0.0, 1.0, 1.0),  # Y + K
            # Three-color overprints
            (1.0, 1.0, 1.0, 0.0),  # C + M + Y (Composite Black)
            (1.0, 1.0, 0.0, 1.0),  # C + M + K
            (1.0, 0.0, 1.0, 1.0),  # C + Y + K
            (0.0, 1.0, 1.0, 1.0),  # M + Y + K
            # Four-color overprint
            (1.0, 1.0, 1.0, 1.0),  # C + M + Y + K (Maximum density)
        ]
        return overprints

    def generate_gray_patches(
        self, steps: int = 10
    ) -> list[tuple[float, float, float, float]]:
        """
        Generate gray patches using K-only and composite CMY grays

        This tests neutral reproduction, which is critical for print quality.

        Args:
            steps: Number of gray levels

        Returns:
            List of CMYK tuples for grays
        """
        grays = []

        # K-only grays (pure black ink gradations)
        for step in range(1, steps):
            k_value = step / steps
            grays.append((0.0, 0.0, 0.0, k_value))

        # Composite CMY grays (no black ink, equal CMY)
        for step in range(1, steps):
            cmy_value = step / steps
            grays.append((cmy_value, cmy_value, cmy_value, 0.0))

        # Mixed grays (CMY + K combinations for testing gray balance)
        for step in range(1, steps // 2):
            base_cmy = step / steps
            k_value = step / (steps * 2)
            grays.append((base_cmy, base_cmy, base_cmy, k_value))

        return grays

    def generate_gamut_sampling_patches(
        self, density: int = 5
    ) -> list[tuple[float, float, float, float]]:
        """
        Generate various CMYK combinations to sample the printer's gamut

        This creates a systematic sampling of the CMYK color space to ensure
        good interpolation coverage across the printer's entire gamut.

        Args:
            density: Sampling density (higher = more patches, exponentially)

        Returns:
            List of CMYK tuples for gamut sampling
        """
        gamut_patches = []

        # Generate systematic combinations
        for c in np.linspace(0, 1, density):
            for m in np.linspace(0, 1, density):
                for y in np.linspace(0, 1, density):
                    for k in np.linspace(0, 1, max(2, density // 2)):
                        # Total ink coverage control (avoid excessive ink)
                        total_ink = c + m + y + k

                        # Skip pure white and limit total ink coverage
                        if 0.1 < total_ink < 3.5:  # Practical ink limits
                            gamut_patches.append((c, m, y, k))

        # Add some specific critical patches
        critical_patches = [
            # Skin tone patches
            (0.15, 0.35, 0.45, 0.05),  # Light skin tone
            (0.25, 0.50, 0.65, 0.15),  # Medium skin tone
            (0.35, 0.65, 0.85, 0.25),  # Dark skin tone
            # Memory colors
            (0.85, 0.10, 1.0, 0.0),  # Blue sky
            (0.30, 0.0, 0.90, 0.0),  # Grass green
            (0.0, 0.75, 1.0, 0.0),  # Apple red
        ]

        gamut_patches.extend(critical_patches)

        return gamut_patches

    def generate_complete_target_chart(self) -> list[tuple[float, float, float, float]]:
        """
        Generate complete CMYK target chart with all patch types

        This creates the comprehensive set of patches needed for accurate
        printer characterization following ICC workflow standards.

        Returns:
            List of all CMYK patches with labels
        """
        all_patches = []
        self.patch_labels = []

        # Add pure white patch first
        all_patches.append((0.0, 0.0, 0.0, 0.0))
        self.patch_labels.append("White")

        # Add primary patches
        primaries = self.generate_primary_patches()
        all_patches.extend(primaries)
        self.patch_labels.extend(["C100", "M100", "Y100", "K100"])

        # Add gradation patches
        gradations = self.generate_gradation_patches()
        all_patches.extend(gradations)

        # Generate labels for gradations
        channels = ["C", "M", "Y", "K"]
        for channel in range(4):
            for step in range(1, 10):  # 10%, 20%, ..., 90%
                self.patch_labels.append(f"{channels[channel]}{step * 10}")

        # Add overprint patches
        overprints = self.generate_overprint_patches()
        all_patches.extend(overprints)

        # Generate labels for overprints
        overprint_labels = [
            "CM100",
            "CY100",
            "CK100",
            "MY100",
            "MK100",
            "YK100",
            "CMY100",
            "CMK100",
            "CYK100",
            "MYK100",
            "CMYK100",
        ]
        self.patch_labels.extend(overprint_labels)

        # Add gray patches
        grays = self.generate_gray_patches()
        all_patches.extend(grays)

        # Generate labels for grays
        for step in range(1, 10):
            self.patch_labels.append(f"K{step * 10}")
        for step in range(1, 10):
            self.patch_labels.append(f"CMY{step * 10}")
        for step in range(1, 5):
            self.patch_labels.append(f"Mixed{step * 20}")

        # Add gamut sampling patches
        gamut_patches = self.generate_gamut_sampling_patches()
        all_patches.extend(gamut_patches)

        # Generate labels for gamut patches
        for i, patch in enumerate(gamut_patches):
            if i < 3:  # Skin tones
                self.patch_labels.append(f"Skin{i + 1}")
            elif i < 6:  # Memory colors
                memory_colors = ["Sky", "Grass", "Apple"]
                self.patch_labels.append(memory_colors[i - 3])
            else:
                self.patch_labels.append(f"Gamut{i - 5}")

        # Remove duplicates while preserving order and labels
        unique_patches = []
        unique_labels = []
        seen = set()

        for patch, label in zip(all_patches, self.patch_labels, strict=False):
            # Round to avoid floating point precision issues
            rounded_patch = tuple(round(x, 3) for x in patch)
            if rounded_patch not in seen:
                seen.add(rounded_patch)
                unique_patches.append(patch)
                unique_labels.append(label)

        self.patches = unique_patches
        self.patch_labels = unique_labels

        return unique_patches

    def create_chart_image(
        self,
        patches: list[tuple[float, float, float, float]] | None = None,
        include_labels: bool = True,
    ) -> np.ndarray:
        """
        Create visual representation of the target chart

        Args:
            patches: List of CMYK patches (uses generated patches if None)
            include_labels: Whether to include patch labels

        Returns:
            RGB image of the target chart
        """
        if patches is None:
            patches = self.patches

        if not patches:
            patches = self.generate_complete_target_chart()

        # Calculate grid dimensions for optimal layout
        num_patches = len(patches)
        grid_cols = int(
            np.ceil(np.sqrt(num_patches * 1.2))
        )  # Slightly wider than square
        grid_rows = int(np.ceil(num_patches / grid_cols))

        # Calculate image dimensions
        patch_total_size = self.patch_size + self.grid_spacing
        img_width = grid_cols * patch_total_size - self.grid_spacing
        img_height = grid_rows * patch_total_size - self.grid_spacing

        # Add extra space for labels if needed
        if include_labels:
            img_height += 30  # Extra space for text

        # Create white background
        chart_image = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

        # Draw patches
        self.patch_positions = []
        for i, cmyk in enumerate(patches):
            row = i // grid_cols
            col = i % grid_cols

            # Calculate patch position
            x = col * patch_total_size
            y = row * patch_total_size

            self.patch_positions.append(
                (x, y, x + self.patch_size, y + self.patch_size)
            )

            # Convert CMYK to RGB for visualization
            cmyk_array = np.array([[cmyk]], dtype=np.float32)
            rgb_array = cmyk_to_rgb(cmyk_array)

            # Ensure valid color values
            rgb_normalized = np.clip(rgb_array[0, 0], 0, 1)
            rgb_color = tuple(int(c * 255) for c in rgb_normalized)

            # Draw patch
            cv2.rectangle(
                chart_image,
                (x, y),
                (x + self.patch_size, y + self.patch_size),
                rgb_color,
                -1,
            )

            # Add border
            cv2.rectangle(
                chart_image,
                (x, y),
                (x + self.patch_size, y + self.patch_size),
                (0, 0, 0),
                1,
            )

            # Add label if requested and available
            if include_labels and i < len(self.patch_labels):
                label = self.patch_labels[i]

                # Add text below patch
                text_x = x + 2
                text_y = y + self.patch_size + 15

                if text_y < img_height - 5:  # Make sure text fits
                    cv2.putText(
                        chart_image,
                        label,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (0, 0, 0),
                        1,
                    )

        return chart_image

    def save_chart_definition(self, filename: str) -> None:
        """
        Save chart definition to JSON file

        Args:
            filename: Path to save the chart definition
        """
        chart_data = {
            "patches": self.patches,
            "patch_labels": self.patch_labels,
            "patch_positions": self.patch_positions,
            "patch_size": self.patch_size,
            "grid_spacing": self.grid_spacing,
            "num_patches": len(self.patches),
            "chart_metadata": {
                "created_with": "CMYK Calibration System",
                "chart_type": "ICC Target Chart",
                "color_space": "CMYK",
                "patch_types": [
                    "primaries",
                    "gradations",
                    "overprints",
                    "grays",
                    "gamut_sampling",
                ],
            },
        }

        with open(filename, "w") as f:
            json.dump(chart_data, f, indent=2)

    def load_chart_definition(self, filename: str) -> None:
        """
        Load chart definition from JSON file

        Args:
            filename: Path to load the chart definition from
        """
        with open(filename) as f:
            chart_data = json.load(f)

        self.patches = [tuple(patch) for patch in chart_data["patches"]]
        self.patch_labels = chart_data.get("patch_labels", [])
        self.patch_positions = chart_data.get("patch_positions", [])
        self.patch_size = chart_data["patch_size"]
        self.grid_spacing = chart_data["grid_spacing"]

    def export_patch_list(self, filename: str) -> None:
        """
        Export patch list in CSV format for reference

        Args:
            filename: Path to save CSV file
        """
        import csv

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow(
                ["Patch_ID", "Label", "C", "M", "Y", "K", "C%", "M%", "Y%", "K%"]
            )

            # Write patch data
            for i, (patch, label) in enumerate(
                zip(self.patches, self.patch_labels, strict=False)
            ):
                c, m, y, k = patch
                writer.writerow(
                    [
                        i + 1,
                        label,
                        f"{c:.3f}",
                        f"{m:.3f}",
                        f"{y:.3f}",
                        f"{k:.3f}",
                        f"{c * 100:.1f}",
                        f"{m * 100:.1f}",
                        f"{y * 100:.1f}",
                        f"{k * 100:.1f}",
                    ]
                )

    def get_chart_statistics(self) -> dict:
        """
        Get statistics about the generated chart

        Returns:
            Dictionary with chart statistics
        """
        if not self.patches:
            return {}

        patches_array = np.array(self.patches)

        return {
            "total_patches": len(self.patches),
            "coverage_statistics": {
                "c_range": [
                    float(np.min(patches_array[:, 0])),
                    float(np.max(patches_array[:, 0])),
                ],
                "m_range": [
                    float(np.min(patches_array[:, 1])),
                    float(np.max(patches_array[:, 1])),
                ],
                "y_range": [
                    float(np.min(patches_array[:, 2])),
                    float(np.max(patches_array[:, 2])),
                ],
                "k_range": [
                    float(np.min(patches_array[:, 3])),
                    float(np.max(patches_array[:, 3])),
                ],
            },
            "ink_usage": {
                "avg_c": float(np.mean(patches_array[:, 0])),
                "avg_m": float(np.mean(patches_array[:, 1])),
                "avg_y": float(np.mean(patches_array[:, 2])),
                "avg_k": float(np.mean(patches_array[:, 3])),
                "avg_total_ink": float(np.mean(np.sum(patches_array, axis=1))),
            },
            "patch_distribution": {
                "low_coverage": int(np.sum(np.sum(patches_array, axis=1) < 0.5)),
                "medium_coverage": int(
                    np.sum(
                        (np.sum(patches_array, axis=1) >= 0.5)
                        & (np.sum(patches_array, axis=1) < 2.0)
                    )
                ),
                "high_coverage": int(np.sum(np.sum(patches_array, axis=1) >= 2.0)),
            },
        }
