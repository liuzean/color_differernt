#!/usr/bin/env python

"""
Measurement Simulation Module

This module simulates spectrophotometer measurements for CMYK patches
since we don't have access to real measurement hardware. It models
printer characteristics like dot gain, ink density, and measurement noise.
"""

import json
from dataclasses import dataclass

import numpy as np

from ..utils import cmyk_to_rgb, rgb_to_lab


@dataclass
class SpectrophotometerData:
    """
    Data structure for storing spectrophotometer measurement data

    This represents what would come from a real spectrophotometer device
    measuring printed color patches.
    """

    cmyk_input: tuple[float, float, float, float]
    lab_measured: tuple[float, float, float]
    measurement_conditions: dict[str, float]
    patch_id: str
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            import datetime

            self.timestamp = datetime.datetime.now().isoformat()


class MeasurementSimulator:
    """
    Simulates spectrophotometer measurements for CMYK patches

    This class models the behavior of a real spectrophotometer measuring
    printed color patches. It includes realistic printer characteristics:
    - Dot gain (ink spreading in midtones)
    - Ink density variations
    - Measurement noise
    - Color coupling between inks
    """

    def __init__(
        self,
        dot_gain: float = 0.15,
        noise_level: float = 0.5,
        ink_density_variation: float = 0.05,
    ):
        """
        Initialize measurement simulator

        Args:
            dot_gain: Dot gain coefficient (0.1-0.3 typical)
            noise_level: Measurement noise level (0.1-1.0)
            ink_density_variation: Ink density variation (0.02-0.1)
        """
        self.dot_gain = dot_gain
        self.noise_level = noise_level
        self.ink_density_variation = ink_density_variation

        # Simulate printer characteristics
        self.ink_characteristics = {
            "C": {
                "max_density": 1.4,
                "hue_shift": 0.02,
                "transparency": 0.85,
                "dot_gain_curve": "standard",
            },
            "M": {
                "max_density": 1.5,
                "hue_shift": -0.01,
                "transparency": 0.80,
                "dot_gain_curve": "standard",
            },
            "Y": {
                "max_density": 1.3,
                "hue_shift": 0.01,
                "transparency": 0.90,
                "dot_gain_curve": "high",
            },
            "K": {
                "max_density": 1.8,
                "hue_shift": 0.0,
                "transparency": 0.95,
                "dot_gain_curve": "low",
            },
        }

        # Substrate properties (paper characteristics)
        self.substrate = {
            "whiteness": 0.95,
            "brightness": 0.92,
            "opacity": 0.98,
            "texture": "coated",
            "lab_white_point": (95.0, 0.0, 0.0),
        }

        # Measurement conditions
        self.measurement_conditions = {
            "illuminant": "D65",
            "observer": "2deg",
            "geometry": "45/0",
            "aperture_size": "4mm",
            "uv_filter": "included",
        }

    def apply_dot_gain(self, cmyk_values: np.ndarray) -> np.ndarray:
        """
        Apply dot gain model to CMYK values

        Dot gain is the phenomenon where ink dots spread during printing,
        especially in midtones, making colors appear darker than intended.

        Args:
            cmyk_values: Original CMYK values [0-1]

        Returns:
            CMYK values with dot gain applied
        """
        adjusted_cmyk = np.zeros_like(cmyk_values)

        for channel in range(4):
            ink_name = ["C", "M", "Y", "K"][channel]
            curve_type = self.ink_characteristics[ink_name]["dot_gain_curve"]

            # Different dot gain curves for different inks
            if curve_type == "standard":
                # Standard S-curve for CMK
                def dot_gain_curve(x):
                    return x + self.dot_gain * x * (1 - x) * 4
            elif curve_type == "high":
                # Higher dot gain for Yellow
                def dot_gain_curve(x):
                    return x + self.dot_gain * 1.3 * x * (1 - x) * 4
            elif curve_type == "low":
                # Lower dot gain for Black
                def dot_gain_curve(x):
                    return x + self.dot_gain * 0.7 * x * (1 - x) * 4
            else:

                def dot_gain_curve(x):
                    return x

            # Apply dot gain curve
            adjusted_cmyk[:, channel] = np.clip(
                dot_gain_curve(cmyk_values[:, channel]), 0, 1
            )

        return adjusted_cmyk

    def apply_ink_density_variation(self, cmyk_values: np.ndarray) -> np.ndarray:
        """
        Apply ink density variations

        Real printers have slight variations in ink density across the page
        and between print runs.

        Args:
            cmyk_values: CMYK values

        Returns:
            CMYK values with density variations applied
        """
        variation = np.random.normal(1.0, self.ink_density_variation, cmyk_values.shape)

        # Apply ink-specific density characteristics
        for channel in range(4):
            ink_name = ["C", "M", "Y", "K"][channel]
            max_density = self.ink_characteristics[ink_name]["max_density"]

            # Scale variation based on ink density
            density_factor = max_density / 1.5  # Normalize to typical range
            variation[:, channel] *= density_factor

        return np.clip(cmyk_values * variation, 0, 1)

    def apply_ink_interaction(self, cmyk_values: np.ndarray) -> np.ndarray:
        """
        Apply ink interaction effects

        When multiple inks are printed together, they interact in complex ways.
        This simulates trapping, transparency, and color coupling effects.

        Args:
            cmyk_values: CMYK values

        Returns:
            CMYK values with interaction effects applied
        """
        adjusted_cmyk = cmyk_values.copy()

        # Calculate total ink coverage
        total_ink = np.sum(cmyk_values, axis=1, keepdims=True)

        # Apply trapping effects (less ink deposited when total coverage is high)
        trapping_factor = 1.0 - 0.1 * np.clip(total_ink - 1.0, 0, 2.0)
        adjusted_cmyk *= trapping_factor

        # Apply transparency effects
        for channel in range(4):
            ink_name = ["C", "M", "Y", "K"][channel]
            transparency = self.ink_characteristics[ink_name]["transparency"]

            # Other inks affect this ink's apparent density
            other_inks = (
                np.sum(cmyk_values, axis=1, keepdims=True)
                - cmyk_values[:, channel : channel + 1]
            )
            transparency_effect = 1.0 - (1.0 - transparency) * other_inks / 3.0
            adjusted_cmyk[:, channel] *= transparency_effect.flatten()

        return np.clip(adjusted_cmyk, 0, 1)

    def cmyk_to_lab_simulation(self, cmyk_values: np.ndarray) -> np.ndarray:
        """
        Convert CMYK to Lab with printer simulation

        This simulates the complete printing process including all the
        non-linearities and interactions that occur in real printing.

        Args:
            cmyk_values: CMYK values [0-1]

        Returns:
            Lab values representing measured colors
        """
        # Apply printer effects in order
        adjusted_cmyk = self.apply_dot_gain(cmyk_values)
        adjusted_cmyk = self.apply_ink_density_variation(adjusted_cmyk)
        adjusted_cmyk = self.apply_ink_interaction(adjusted_cmyk)

        # Convert to RGB first (simulating the visual result)
        rgb_values = cmyk_to_rgb(adjusted_cmyk)

        # Ensure RGB has proper shape for rgb_to_lab (needs 3D array)
        if len(rgb_values.shape) == 2:
            rgb_values = rgb_values.reshape(rgb_values.shape[0], 1, rgb_values.shape[1])

        # Apply substrate effects
        np.array(self.substrate["lab_white_point"])
        substrate_factor = self.substrate["whiteness"]

        # Convert RGB to Lab
        lab_values = rgb_to_lab(rgb_values)

        # Flatten Lab values back to 2D if needed
        if len(lab_values.shape) == 3 and lab_values.shape[1] == 1:
            lab_values = lab_values.reshape(lab_values.shape[0], lab_values.shape[2])

        # Apply substrate influence
        lab_values = lab_values.astype(np.float32)
        lab_values[:, 0] *= substrate_factor  # Adjust lightness

        # Add some realistic Lab space adjustments
        # Adjust a* and b* based on substrate characteristics
        if self.substrate["texture"] == "coated":
            # Coated papers typically have better color reproduction
            lab_values[:, 1:] *= 1.05  # Slightly enhance chroma
        else:
            # Uncoated papers typically have reduced chroma
            lab_values[:, 1:] *= 0.95

        return lab_values

    def add_measurement_noise(self, lab_values: np.ndarray) -> np.ndarray:
        """
        Add realistic measurement noise to Lab values

        Real spectrophotometers have measurement uncertainty and noise.

        Args:
            lab_values: Lab values

        Returns:
            Lab values with measurement noise added
        """
        # Different noise levels for L*, a*, b*
        noise_l = np.random.normal(0, self.noise_level * 0.5, lab_values.shape[0])
        noise_a = np.random.normal(0, self.noise_level * 0.3, lab_values.shape[0])
        noise_b = np.random.normal(0, self.noise_level * 0.3, lab_values.shape[0])

        # Apply noise
        lab_noisy = lab_values.copy()
        lab_noisy[:, 0] += noise_l
        lab_noisy[:, 1] += noise_a
        lab_noisy[:, 2] += noise_b

        # Ensure valid Lab ranges
        lab_noisy[:, 0] = np.clip(lab_noisy[:, 0], 0, 100)
        lab_noisy[:, 1] = np.clip(lab_noisy[:, 1], -128, 127)
        lab_noisy[:, 2] = np.clip(lab_noisy[:, 2], -128, 127)

        return lab_noisy

    def simulate_single_measurement(
        self, cmyk_patch: tuple[float, float, float, float], patch_id: str = ""
    ) -> SpectrophotometerData:
        """
        Simulate measurement of a single CMYK patch

        Args:
            cmyk_patch: CMYK values for the patch
            patch_id: Identifier for the patch

        Returns:
            Simulated spectrophotometer data
        """
        # Convert to numpy array for processing
        cmyk_array = np.array([cmyk_patch]).reshape(1, 4)

        # Simulate printing and measurement
        lab_values = self.cmyk_to_lab_simulation(cmyk_array)
        lab_noisy = self.add_measurement_noise(lab_values)

        # Convert back to tuple
        lab_measured = tuple(lab_noisy[0])

        return SpectrophotometerData(
            cmyk_input=cmyk_patch,
            lab_measured=lab_measured,
            measurement_conditions=self.measurement_conditions.copy(),
            patch_id=patch_id,
        )

    def simulate_measurement_set(
        self,
        cmyk_patches: list[tuple[float, float, float, float]],
        patch_labels: list[str] | None = None,
    ) -> list[SpectrophotometerData]:
        """
        Simulate measurement of multiple CMYK patches

        Args:
            cmyk_patches: List of CMYK patches to measure
            patch_labels: Optional labels for patches

        Returns:
            List of simulated spectrophotometer data
        """
        if patch_labels is None:
            patch_labels = [f"Patch_{i + 1}" for i in range(len(cmyk_patches))]

        measurements = []

        for i, cmyk_patch in enumerate(cmyk_patches):
            label = patch_labels[i] if i < len(patch_labels) else f"Patch_{i + 1}"
            measurement = self.simulate_single_measurement(cmyk_patch, label)
            measurements.append(measurement)

        return measurements

    def create_measurement_dataset(self, target_chart) -> dict:
        """
        Create a complete measurement dataset from target chart

        Args:
            target_chart: CMYKTargetChart instance

        Returns:
            Dictionary with measurement data
        """
        # Generate patches if not already done
        if not target_chart.patches:
            target_chart.generate_complete_target_chart()

        # Simulate measurements
        measurements = self.simulate_measurement_set(
            target_chart.patches, target_chart.patch_labels
        )

        # Convert to arrays for analysis
        cmyk_inputs = [m.cmyk_input for m in measurements]
        lab_measurements = [m.lab_measured for m in measurements]

        dataset = {
            "cmyk_inputs": cmyk_inputs,
            "lab_measurements": lab_measurements,
            "measurement_metadata": {
                "simulator_settings": {
                    "dot_gain": self.dot_gain,
                    "noise_level": self.noise_level,
                    "ink_density_variation": self.ink_density_variation,
                },
                "measurement_conditions": self.measurement_conditions,
                "substrate_properties": self.substrate,
                "ink_characteristics": self.ink_characteristics,
                "num_patches": len(measurements),
            },
            "measurements": measurements,  # Full measurement objects
        }

        return dataset

    def save_measurements(
        self, measurements: list[SpectrophotometerData], filename: str
    ) -> None:
        """
        Save measurement data to file

        Args:
            measurements: List of measurement data
            filename: File path to save to
        """
        # Convert measurements to serializable format
        serializable_data = {
            "measurements": [
                {
                    "cmyk_input": m.cmyk_input,
                    "lab_measured": m.lab_measured,
                    "measurement_conditions": m.measurement_conditions,
                    "patch_id": m.patch_id,
                    "timestamp": m.timestamp,
                }
                for m in measurements
            ],
            "simulator_settings": {
                "dot_gain": self.dot_gain,
                "noise_level": self.noise_level,
                "ink_density_variation": self.ink_density_variation,
            },
            "measurement_metadata": {
                "measurement_conditions": self.measurement_conditions,
                "substrate_properties": self.substrate,
                "ink_characteristics": self.ink_characteristics,
            },
        }

        with open(filename, "w") as f:
            json.dump(serializable_data, f, indent=2)

    def load_measurements(self, filename: str) -> list[SpectrophotometerData]:
        """
        Load measurement data from file

        Args:
            filename: File path to load from

        Returns:
            List of measurement data
        """
        with open(filename) as f:
            data = json.load(f)

        measurements = []
        for m_data in data["measurements"]:
            measurement = SpectrophotometerData(
                cmyk_input=tuple(m_data["cmyk_input"]),
                lab_measured=tuple(m_data["lab_measured"]),
                measurement_conditions=m_data["measurement_conditions"],
                patch_id=m_data["patch_id"],
                timestamp=m_data["timestamp"],
            )
            measurements.append(measurement)

        return measurements

    def analyze_measurement_quality(
        self, measurements: list[SpectrophotometerData]
    ) -> dict:
        """
        Analyze the quality of measurements

        Args:
            measurements: List of measurement data

        Returns:
            Quality analysis results
        """
        if not measurements:
            return {}

        # Extract Lab values
        lab_values = np.array([m.lab_measured for m in measurements])
        np.array([m.cmyk_input for m in measurements])

        # Calculate statistics
        lab_stats = {
            "L_range": [
                float(np.min(lab_values[:, 0])),
                float(np.max(lab_values[:, 0])),
            ],
            "a_range": [
                float(np.min(lab_values[:, 1])),
                float(np.max(lab_values[:, 1])),
            ],
            "b_range": [
                float(np.min(lab_values[:, 2])),
                float(np.max(lab_values[:, 2])),
            ],
            "L_mean": float(np.mean(lab_values[:, 0])),
            "a_mean": float(np.mean(lab_values[:, 1])),
            "b_mean": float(np.mean(lab_values[:, 2])),
        }

        # Calculate gamut volume estimation
        lab_centered = lab_values - np.mean(lab_values, axis=0)
        gamut_volume = float(np.sqrt(np.sum(np.var(lab_centered, axis=0))))

        # Check for potential issues
        quality_issues = []

        # Check for extremely high or low values
        if lab_stats["L_range"][1] > 98:
            quality_issues.append("Very high L* values detected")
        if lab_stats["L_range"][0] < 2:
            quality_issues.append("Very low L* values detected")

        # Check for extreme chroma values
        max_chroma = np.max(np.sqrt(lab_values[:, 1] ** 2 + lab_values[:, 2] ** 2))
        if max_chroma > 100:
            quality_issues.append("Extremely high chroma values detected")

        return {
            "num_measurements": len(measurements),
            "lab_statistics": lab_stats,
            "gamut_volume_estimate": gamut_volume,
            "quality_issues": quality_issues,
            "simulator_settings": {
                "dot_gain": self.dot_gain,
                "noise_level": self.noise_level,
                "ink_density_variation": self.ink_density_variation,
            },
        }

    def create_reference_measurements(self) -> list[SpectrophotometerData]:
        """
        Create reference measurements for standard patches

        Returns:
            List of reference measurements for validation
        """
        reference_patches = [
            # Standard reference patches
            (0.0, 0.0, 0.0, 0.0),  # Paper white
            (1.0, 0.0, 0.0, 0.0),  # 100% Cyan
            (0.0, 1.0, 0.0, 0.0),  # 100% Magenta
            (0.0, 0.0, 1.0, 0.0),  # 100% Yellow
            (0.0, 0.0, 0.0, 1.0),  # 100% Black
            (0.5, 0.5, 0.5, 0.0),  # 50% CMY
            (0.0, 0.0, 0.0, 0.5),  # 50% K
            (1.0, 1.0, 1.0, 0.0),  # 100% CMY
            (1.0, 1.0, 1.0, 1.0),  # 100% CMYK
        ]

        reference_labels = [
            "White",
            "Cyan",
            "Magenta",
            "Yellow",
            "Black",
            "CMY_50",
            "K_50",
            "CMY_100",
            "CMYK_100",
        ]

        return self.simulate_measurement_set(reference_patches, reference_labels)
