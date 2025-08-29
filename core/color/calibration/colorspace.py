#!/usr/bin/env python

"""
CMYK Color Space Module

This module defines the CMYKColorSpace class that manages calibration data
and provides calibrated color transformations using the built lookup tables.
"""

import pickle

import numpy as np

from ..utils import calculate_color_difference, convert_color, rgb_to_lab
from .lut_builder import LookupTableBuilder


class CMYKColorSpace:
    """
    Calibrated CMYK color space

    This class represents a calibrated CMYK color space that uses measured
    data and lookup tables to provide accurate color transformations.
    It's the core of an ICC profile for CMYK printing.
    """

    def __init__(self, name: str = "Calibrated CMYK"):
        """
        Initialize CMYK color space

        Args:
            name: Name of the color space
        """
        self.name = name
        self.is_calibrated = False
        self.lut_builder = LookupTableBuilder()
        self.calibration_data = None

        # Color space properties
        self.white_point = (95.047, 100.0, 108.883)  # D65 white point in XYZ
        self.black_point = (0.0, 0.0, 0.0)  # Black point in XYZ
        self.gamma = 2.2  # Approximate gamma for viewing conditions

        # Printing characteristics
        self.ink_limit = 350  # Total ink limit (%)
        self.rendering_intent = "perceptual"  # ICC rendering intent

    def calibrate(self, measurement_dataset: dict) -> None:
        """
        Calibrate the CMYK color space with measurement data

        Args:
            measurement_dataset: Dataset with CMYK inputs and Lab measurements
        """
        print(f"Calibrating CMYK color space '{self.name}'...")

        # Build the lookup table
        self.lut_builder.build_lut(measurement_dataset)

        # Store calibration data
        self.calibration_data = {
            "dataset": measurement_dataset,
            "calibration_date": np.datetime64("now"),
            "num_patches": len(measurement_dataset["cmyk_inputs"]),
        }

        self.is_calibrated = True
        print(f"Calibration completed for '{self.name}'")

    def rgb_to_cmyk_calibrated(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Convert RGB to CMYK using calibrated color space

        Args:
            rgb_image: RGB image array

        Returns:
            Calibrated CMYK image array
        """
        if not self.is_calibrated:
            raise ValueError("Color space not calibrated. Call calibrate() first.")

        # Convert RGB to Lab
        lab_image = rgb_to_lab(rgb_image)

        # Use LUT to convert Lab to CMYK
        cmyk_image = self.lab_to_cmyk_calibrated(lab_image)

        return cmyk_image

    def cmyk_to_rgb_calibrated(self, cmyk_image: np.ndarray) -> np.ndarray:
        """
        Convert CMYK to RGB using calibrated color space

        This uses the forward model (CMYK -> Lab -> RGB) based on
        the measurement data to predict what the printed result will look like.

        Args:
            cmyk_image: CMYK image array

        Returns:
            RGB image array representing the predicted print result
        """
        if not self.is_calibrated:
            raise ValueError("Color space not calibrated. Call calibrate() first.")

        # Get original shape for reshaping
        original_shape = cmyk_image.shape

        # Flatten for processing
        cmyk_flat = cmyk_image.reshape(-1, 4)

        # Find closest measured points for each CMYK value
        lab_predicted = self._predict_lab_from_cmyk(cmyk_flat)

        # Reshape back to image shape
        lab_image = lab_predicted.reshape(original_shape[:-1] + (3,))

        # Convert Lab to RGB
        rgb_image = lab_to_rgb(lab_image)

        return rgb_image

    def lab_to_cmyk_calibrated(self, lab_image: np.ndarray) -> np.ndarray:
        """
        Convert Lab to CMYK using calibrated lookup table

        Args:
            lab_image: Lab image array

        Returns:
            CMYK image array
        """
        if not self.is_calibrated:
            raise ValueError("Color space not calibrated. Call calibrate() first.")

        # Get original shape for reshaping
        original_shape = lab_image.shape

        # Flatten for processing
        lab_flat = lab_image.reshape(-1, 3)

        # Use LUT to interpolate CMYK values
        cmyk_flat = self.lut_builder.interpolate_cmyk(lab_flat)

        # Reshape back to image shape
        cmyk_image = cmyk_flat.reshape(original_shape[:-1] + (4,))

        return cmyk_image

    def _predict_lab_from_cmyk(self, cmyk_values: np.ndarray) -> np.ndarray:
        """
        Predict Lab values from CMYK using measurement data

        This implements the forward model using nearest neighbor interpolation
        of the measurement data.

        Args:
            cmyk_values: CMYK values, shape (N, 4)

        Returns:
            Predicted Lab values, shape (N, 3)
        """
        if self.calibration_data is None:
            raise ValueError("No calibration data available")

        measured_cmyk = np.array(self.calibration_data["dataset"]["cmyk_inputs"])
        measured_lab = np.array(self.calibration_data["dataset"]["lab_measurements"])

        # Find closest measured CMYK for each input CMYK
        from scipy.spatial.distance import cdist

        distances = cdist(cmyk_values, measured_cmyk)
        closest_indices = np.argmin(distances, axis=1)

        # Return corresponding Lab values
        return measured_lab[closest_indices]

    def validate_calibration(
        self, test_colors: list[tuple[int, int, int]] | None = None
    ) -> dict:
        """
        Validate calibration accuracy with test colors

        Args:
            test_colors: List of RGB test colors

        Returns:
            Validation results
        """
        if not self.is_calibrated:
            raise ValueError("Color space not calibrated")

        if test_colors is None:
            # Default test colors
            test_colors = [
                (255, 0, 0),  # Red
                (0, 255, 0),  # Green
                (0, 0, 255),  # Blue
                (255, 255, 0),  # Yellow
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
                (128, 128, 128),  # Gray
                (255, 255, 255),  # White
                (0, 0, 0),  # Black
            ]

        validation_results = {"test_colors": [], "accuracy_metrics": {}}

        total_delta_e = 0

        for rgb_color in test_colors:
            # Convert to numpy array
            rgb_array = np.array([[rgb_color]], dtype=np.uint8)

            # Round trip: RGB -> CMYK -> RGB
            cmyk_calibrated = self.rgb_to_cmyk_calibrated(rgb_array)
            rgb_back_calibrated = self.cmyk_to_rgb_calibrated(cmyk_calibrated)

            # Calculate color difference
            delta_e, _ = calculate_color_difference(rgb_array, rgb_back_calibrated)
            total_delta_e += delta_e

            validation_results["test_colors"].append(
                {
                    "input_rgb": rgb_color,
                    "calibrated_cmyk": cmyk_calibrated[0, 0].tolist(),
                    "calibrated_rgb_back": rgb_back_calibrated[0, 0].tolist(),
                    "delta_e": float(delta_e),
                }
            )

        # Calculate accuracy metrics
        validation_results["accuracy_metrics"] = {
            "average_delta_e": total_delta_e / len(test_colors),
            "num_test_colors": len(test_colors),
            "calibration_patches": len(self.calibration_data["dataset"]["cmyk_inputs"]),
        }

        return validation_results

    def get_gamut_volume(self) -> float:
        """
        Estimate the gamut volume of this CMYK color space

        Returns:
            Estimated gamut volume in Lab space
        """
        if not self.is_calibrated:
            return 0.0

        lab_points = np.array(self.calibration_data["dataset"]["lab_measurements"])

        # Calculate approximate gamut volume using convex hull
        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(lab_points)
            return float(hull.volume)
        except Exception:
            # Fallback: use bounding box volume
            lab_ranges = np.ptp(lab_points, axis=0)  # Peak-to-peak (max - min)
            return float(np.prod(lab_ranges))

    def get_color_space_info(self) -> dict:
        """
        Get comprehensive information about this color space

        Returns:
            Dictionary with color space information
        """
        info = {
            "name": self.name,
            "is_calibrated": self.is_calibrated,
            "rendering_intent": self.rendering_intent,
            "ink_limit": self.ink_limit,
            "white_point": self.white_point,
            "black_point": self.black_point,
        }

        if self.is_calibrated:
            info.update(
                {
                    "calibration_date": str(self.calibration_data["calibration_date"]),
                    "measurement_patches": self.calibration_data["num_patches"],
                    "gamut_volume": self.get_gamut_volume(),
                    "lut_info": self.lut_builder.analyze_lut_coverage(),
                }
            )

        return info

    def apply_rendering_intent(
        self, source_lab: np.ndarray, intent: str = None
    ) -> np.ndarray:
        """
        Apply rendering intent for color conversion

        Args:
            source_lab: Source Lab values
            intent: Rendering intent ('perceptual', 'relative', 'saturation', 'absolute')

        Returns:
            Lab values adjusted for rendering intent
        """
        if intent is None:
            intent = self.rendering_intent

        # This is a simplified implementation
        # A full ICC implementation would have complex gamut mapping algorithms

        if intent == "perceptual":
            # Perceptual: maintain overall appearance
            return self._apply_perceptual_mapping(source_lab)
        elif intent == "relative":
            # Relative colorimetric: clip out-of-gamut colors
            return self._apply_relative_mapping(source_lab)
        elif intent == "saturation":
            # Saturation: maintain vibrant colors
            return self._apply_saturation_mapping(source_lab)
        elif intent == "absolute":
            # Absolute colorimetric: exact color match where possible
            return source_lab  # No adjustment
        else:
            return source_lab

    def _apply_perceptual_mapping(self, lab_values: np.ndarray) -> np.ndarray:
        """Apply perceptual rendering intent"""
        # Check if points are in gamut
        in_gamut = self.lut_builder.is_in_gamut(lab_values)

        # For out-of-gamut points, compress toward gamut center
        if not np.all(in_gamut):
            out_of_gamut = ~in_gamut

            # Calculate gamut center
            if self.calibration_data:
                lab_measurements = np.array(
                    self.calibration_data["dataset"]["lab_measurements"]
                )
                gamut_center = np.mean(lab_measurements, axis=0)
            else:
                gamut_center = np.array([50.0, 0.0, 0.0])  # Default Lab center

            # Compress out-of-gamut colors toward center
            adjusted_lab = lab_values.copy()
            for i in np.where(out_of_gamut)[0]:
                direction = adjusted_lab[i] - gamut_center
                # Reduce the distance by 20% iteratively until in gamut
                scale = 0.8
                while (
                    not self.lut_builder.is_in_gamut(adjusted_lab[i : i + 1])[0]
                    and scale > 0.1
                ):
                    adjusted_lab[i] = gamut_center + direction * scale
                    scale *= 0.9

            return adjusted_lab

        return lab_values

    def _apply_relative_mapping(self, lab_values: np.ndarray) -> np.ndarray:
        """Apply relative colorimetric rendering intent"""
        # Simply clip to gamut boundary
        in_gamut = self.lut_builder.is_in_gamut(lab_values)

        if not np.all(in_gamut):
            adjusted_lab = lab_values.copy()
            out_of_gamut_indices = np.where(~in_gamut)[0]

            for i in out_of_gamut_indices:
                # Find closest gamut boundary point
                boundary_point = self.lut_builder.find_gamut_boundary_point(
                    lab_values[i]
                )
                adjusted_lab[i] = boundary_point

            return adjusted_lab

        return lab_values

    def _apply_saturation_mapping(self, lab_values: np.ndarray) -> np.ndarray:
        """Apply saturation rendering intent"""
        # Enhance chroma for in-gamut colors, compress for out-of-gamut
        in_gamut = self.lut_builder.is_in_gamut(lab_values)
        adjusted_lab = lab_values.copy()

        # Calculate chroma (distance from neutral axis)
        np.sqrt(lab_values[:, 1] ** 2 + lab_values[:, 2] ** 2)

        # Enhance chroma for in-gamut colors
        if np.any(in_gamut):
            enhancement_factor = 1.1  # 10% chroma boost
            adjusted_lab[in_gamut, 1:] *= enhancement_factor

        # Check if enhanced colors are still in gamut
        still_in_gamut = self.lut_builder.is_in_gamut(adjusted_lab)

        # For colors that went out of gamut, use perceptual mapping
        out_of_gamut_after_enhancement = in_gamut & ~still_in_gamut
        if np.any(out_of_gamut_after_enhancement):
            adjusted_lab[
                out_of_gamut_after_enhancement
            ] = self._apply_perceptual_mapping(
                adjusted_lab[out_of_gamut_after_enhancement]
            )

        return adjusted_lab

    def save_colorspace(self, filename: str) -> None:
        """
        Save color space to file

        Args:
            filename: Path to save the color space
        """
        if not self.is_calibrated:
            raise ValueError("Cannot save uncalibrated color space")

        colorspace_data = {
            "name": self.name,
            "calibration_data": self.calibration_data,
            "lut_data": self.lut_builder.lut_data,
            "white_point": self.white_point,
            "black_point": self.black_point,
            "ink_limit": self.ink_limit,
            "rendering_intent": self.rendering_intent,
        }

        with open(filename, "wb") as f:
            pickle.dump(colorspace_data, f)

        print(f"Color space saved to {filename}")

    def load_colorspace(self, filename: str) -> None:
        """
        Load color space from file

        Args:
            filename: Path to load the color space from
        """
        with open(filename, "rb") as f:
            colorspace_data = pickle.load(f)

        self.name = colorspace_data["name"]
        self.calibration_data = colorspace_data["calibration_data"]
        self.white_point = colorspace_data.get("white_point", self.white_point)
        self.black_point = colorspace_data.get("black_point", self.black_point)
        self.ink_limit = colorspace_data.get("ink_limit", self.ink_limit)
        self.rendering_intent = colorspace_data.get(
            "rendering_intent", self.rendering_intent
        )

        # Recreate LUT builder
        self.lut_builder = LookupTableBuilder()
        self.lut_builder.lut_data = colorspace_data["lut_data"]
        self.lut_builder.cmyk_data = self.lut_builder.lut_data["cmyk_points"]
        self.lut_builder.lab_data = self.lut_builder.lut_data["lab_points"]
        self.lut_builder.interpolation_method = self.lut_builder.lut_data[
            "interpolation_method"
        ]

        # Rebuild interpolators
        self.lut_builder._build_interpolators()

        self.is_calibrated = True
        print(f"Color space loaded from {filename}")

    def export_as_icc_profile(self, filename: str) -> None:
        """
        Export color space as ICC profile data

        Args:
            filename: Path to save ICC profile data
        """
        if not self.is_calibrated:
            raise ValueError("Cannot export uncalibrated color space")

        # Export the LUT data in ICC-compatible format
        icc_filename = filename.replace(".icc", "_lut.pkl")
        self.lut_builder.export_lut_as_icc_data(icc_filename)

        # Create ICC profile metadata
        icc_metadata = {
            "profile_description": f"{self.name} CMYK Profile",
            "profile_class": "output",  # Output (printer) profile
            "color_space": "CMYK",
            "pcs": "Lab",  # Profile Connection Space
            "rendering_intent": self.rendering_intent,
            "white_point": self.white_point,
            "black_point": self.black_point,
            "creation_date": str(self.calibration_data["calibration_date"]),
            "measurement_data": {
                "patches": len(self.calibration_data["dataset"]["cmyk_inputs"]),
                "gamut_volume": self.get_gamut_volume(),
            },
        }

        # Save metadata
        metadata_filename = filename.replace(".icc", "_metadata.json")
        import json

        with open(metadata_filename, "w") as f:
            json.dump(icc_metadata, f, indent=2)

        print("ICC profile data exported:")
        print(f"  LUT data: {icc_filename}")
        print(f"  Metadata: {metadata_filename}")

    def convert_image(
        self,
        image: np.ndarray,
        from_space: str,
        to_space: str,
        rendering_intent: str = None,
    ) -> np.ndarray:
        """
        Convert image between color spaces using calibrated conversions

        Args:
            image: Input image array
            from_space: Source color space ('rgb', 'cmyk', 'lab')
            to_space: Target color space ('rgb', 'cmyk', 'lab')
            rendering_intent: Optional rendering intent override

        Returns:
            Converted image array
        """
        if not self.is_calibrated:
            raise ValueError("Color space not calibrated")

        # Handle direct conversions
        if from_space.lower() == "rgb" and to_space.lower() == "cmyk":
            return self.rgb_to_cmyk_calibrated(image)
        elif from_space.lower() == "cmyk" and to_space.lower() == "rgb":
            return self.cmyk_to_rgb_calibrated(image)
        elif from_space.lower() == "lab" and to_space.lower() == "cmyk":
            if rendering_intent:
                image = self.apply_rendering_intent(image, rendering_intent)
            return self.lab_to_cmyk_calibrated(image)
        elif from_space.lower() == "cmyk" and to_space.lower() == "lab":
            # Convert CMYK to Lab using forward model
            original_shape = image.shape
            cmyk_flat = image.reshape(-1, 4)
            lab_flat = self._predict_lab_from_cmyk(cmyk_flat)
            return lab_flat.reshape(original_shape[:-1] + (3,))
        else:
            # Use general color conversion for other cases
            return convert_color(image, from_space, to_space)
