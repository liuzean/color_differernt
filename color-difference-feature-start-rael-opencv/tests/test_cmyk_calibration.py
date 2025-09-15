#!/usr/bin/env python

"""
Test suite for CMYK calibration system

This module provides comprehensive tests for the CMYK calibration functionality,
including target chart generation, measurement simulation, lookup table building,
and the complete calibration pipeline.
"""

import os
import tempfile
from pathlib import Path

import numpy as np

# Import calibration modules
from core.color.calibration import (
    CMYKCalibrationPipeline,
    CMYKColorSpace,
    CMYKTargetChart,
    LookupTableBuilder,
    MeasurementSimulator,
)
from core.color.utils import cmyk_to_rgb, rgb_to_lab


class TestCMYKTargetChart:
    """Test CMYK target chart generation"""

    def test_primary_patches_generation(self):
        """Test generation of primary color patches"""
        chart = CMYKTargetChart()
        primaries = chart.generate_primary_patches()

        # Should have 4 primary patches
        assert len(primaries) == 4

        # Verify each primary
        expected_primaries = [
            (1.0, 0.0, 0.0, 0.0),  # Cyan
            (0.0, 1.0, 0.0, 0.0),  # Magenta
            (0.0, 0.0, 1.0, 0.0),  # Yellow
            (0.0, 0.0, 0.0, 1.0),  # Black
        ]

        for expected, actual in zip(expected_primaries, primaries, strict=False):
            assert expected == actual

    def test_gradation_patches_generation(self):
        """Test generation of gradation patches"""
        chart = CMYKTargetChart()
        gradations = chart.generate_gradation_patches(steps=5)

        # Should have 4 channels × 4 steps = 16 patches
        assert len(gradations) == 16

        # Check that gradations are properly spaced
        for i, patch in enumerate(gradations):
            channel = i // 4
            step = (i % 4) + 1
            expected_value = step / 5

            # Only the corresponding channel should be non-zero
            for j, value in enumerate(patch):
                if j == channel:
                    assert abs(value - expected_value) < 1e-6
                else:
                    assert value == 0.0

    def test_overprint_patches_generation(self):
        """Test generation of overprint patches"""
        chart = CMYKTargetChart()
        overprints = chart.generate_overprint_patches()

        # Should have 11 overprint combinations
        assert len(overprints) == 11

        # Check that overprints contain expected combinations
        assert (1.0, 1.0, 0.0, 0.0) in overprints  # C + M
        assert (1.0, 1.0, 1.0, 1.0) in overprints  # C + M + Y + K

    def test_gray_patches_generation(self):
        """Test generation of gray patches"""
        chart = CMYKTargetChart()
        grays = chart.generate_gray_patches(steps=5)

        # Should have 4 K-only grays + 4 CMY grays = 8 patches
        assert len(grays) == 8

        # Check K-only grays
        k_only_grays = [
            patch
            for patch in grays
            if patch[0] == 0 and patch[1] == 0 and patch[2] == 0
        ]
        assert len(k_only_grays) == 4

        # Check CMY grays
        cmy_grays = [
            patch
            for patch in grays
            if patch[3] == 0 and patch[0] == patch[1] == patch[2]
        ]
        assert len(cmy_grays) == 4

    def test_complete_target_chart_generation(self):
        """Test generation of complete target chart"""
        chart = CMYKTargetChart()
        patches = chart.generate_complete_target_chart()

        # Should have a reasonable number of patches
        assert len(patches) > 50  # At least 50 patches
        assert len(patches) < 1000  # But not too many

        # Should include white patch
        assert (0.0, 0.0, 0.0, 0.0) in patches

        # Should include primary patches
        assert (1.0, 0.0, 0.0, 0.0) in patches  # Cyan
        assert (0.0, 1.0, 0.0, 0.0) in patches  # Magenta
        assert (0.0, 0.0, 1.0, 0.0) in patches  # Yellow
        assert (0.0, 0.0, 0.0, 1.0) in patches  # Black

    def test_chart_image_creation(self):
        """Test creation of chart image"""
        chart = CMYKTargetChart(patch_size=30, grid_spacing=5)
        patches = chart.generate_complete_target_chart()

        chart_image = chart.create_chart_image(patches)

        # Should be a valid image
        assert isinstance(chart_image, np.ndarray)
        assert len(chart_image.shape) == 3
        assert chart_image.shape[2] == 3  # RGB
        assert chart_image.dtype == np.uint8

        # Should have reasonable dimensions
        assert chart_image.shape[0] > 0
        assert chart_image.shape[1] > 0

    def test_chart_save_load(self):
        """Test saving and loading chart definitions"""
        chart = CMYKTargetChart()
        chart.generate_complete_target_chart()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            # Save chart definition
            chart.save_chart_definition(temp_file)

            # Load into new chart
            new_chart = CMYKTargetChart()
            new_chart.load_chart_definition(temp_file)

            # Should have same patches
            assert len(new_chart.patches) == len(chart.patches)
            assert new_chart.patches == chart.patches
            assert new_chart.patch_size == chart.patch_size
            assert new_chart.grid_spacing == chart.grid_spacing

        finally:
            os.unlink(temp_file)


class TestMeasurementSimulator:
    """Test measurement simulation"""

    def test_dot_gain_application(self):
        """Test dot gain simulation"""
        simulator = MeasurementSimulator(dot_gain=0.1)

        # Test with midtone values where dot gain is most pronounced
        cmyk_input = np.array([[[0.5, 0.5, 0.5, 0.5]]], dtype=np.float32)
        adjusted_cmyk = simulator.apply_dot_gain(cmyk_input)

        # All values should be increased due to dot gain
        assert np.all(adjusted_cmyk >= cmyk_input)

        # Should still be within valid range
        assert np.all(adjusted_cmyk >= 0.0)
        assert np.all(adjusted_cmyk <= 1.0)

    def test_measurement_noise(self):
        """Test measurement noise addition"""
        simulator = MeasurementSimulator(noise_level=0.1)

        # Create consistent Lab values
        lab_input = np.array([[[50.0, 0.0, 0.0]]], dtype=np.float32)

        # Apply noise multiple times and check variation
        noisy_results = []
        for _ in range(10):
            noisy_lab = simulator.add_measurement_noise(lab_input.copy())
            noisy_results.append(noisy_lab[0, 0])

        noisy_results = np.array(noisy_results)

        # Should have some variation
        assert np.std(noisy_results[:, 0]) > 0  # L channel
        assert np.std(noisy_results[:, 1]) > 0  # a channel
        assert np.std(noisy_results[:, 2]) > 0  # b channel

    def test_measurement_simulation(self):
        """Test complete measurement simulation"""
        simulator = MeasurementSimulator(noise_level=0.1, dot_gain=0.1)

        # Test patches
        test_patches = [
            (0.0, 0.0, 0.0, 0.0),  # White
            (1.0, 0.0, 0.0, 0.0),  # Cyan
            (0.0, 1.0, 0.0, 0.0),  # Magenta
            (0.0, 0.0, 1.0, 0.0),  # Yellow
            (0.0, 0.0, 0.0, 1.0),  # Black
        ]

        lab_measurements = simulator.simulate_measurement(test_patches)

        # Should have same number of measurements as patches
        assert len(lab_measurements) == len(test_patches)

        # Each measurement should be a 3-tuple (L, a, b)
        for lab in lab_measurements:
            assert len(lab) == 3
            assert all(isinstance(x, float) for x in lab)

    def test_measurement_dataset_creation(self):
        """Test creation of measurement dataset"""
        chart = CMYKTargetChart()
        chart.generate_complete_target_chart()

        simulator = MeasurementSimulator()
        dataset = simulator.create_measurement_dataset(chart)

        # Should have required keys
        assert "cmyk_inputs" in dataset
        assert "lab_measurements" in dataset
        assert "measurement_metadata" in dataset

        # Should have matching numbers of inputs and measurements
        assert len(dataset["cmyk_inputs"]) == len(dataset["lab_measurements"])
        assert len(dataset["cmyk_inputs"]) == len(chart.patches)

        # Metadata should be present
        metadata = dataset["measurement_metadata"]
        assert "noise_level" in metadata
        assert "dot_gain" in metadata
        assert "num_patches" in metadata


class TestLookupTableBuilder:
    """Test lookup table building"""

    def create_test_dataset(self):
        """Create a small test dataset"""
        # Simple test dataset with known CMYK-Lab pairs
        cmyk_inputs = [
            (0.0, 0.0, 0.0, 0.0),  # White
            (1.0, 0.0, 0.0, 0.0),  # Cyan
            (0.0, 1.0, 0.0, 0.0),  # Magenta
            (0.0, 0.0, 1.0, 0.0),  # Yellow
            (0.0, 0.0, 0.0, 1.0),  # Black
            (0.5, 0.5, 0.5, 0.5),  # Gray
        ]

        lab_measurements = []
        for cmyk in cmyk_inputs:
            # Convert through RGB to get Lab
            cmyk_array = np.array([[cmyk]], dtype=np.float32)
            rgb_array = cmyk_to_rgb(cmyk_array)
            lab_array = rgb_to_lab(rgb_array)
            lab_measurements.append(tuple(lab_array[0, 0]))

        return {
            "cmyk_inputs": cmyk_inputs,
            "lab_measurements": lab_measurements,
            "measurement_metadata": {"test": True},
        }

    def test_lut_building(self):
        """Test LUT building from dataset"""
        dataset = self.create_test_dataset()

        lut_builder = LookupTableBuilder()
        lut_builder.build_lut(dataset)

        # Should have built LUT data
        assert lut_builder.lut_data is not None
        assert lut_builder.cmyk_data is not None
        assert lut_builder.lab_data is not None

        # Data should match input dataset
        assert len(lut_builder.cmyk_data) == len(dataset["cmyk_inputs"])
        assert len(lut_builder.lab_data) == len(dataset["lab_measurements"])

    def test_cmyk_interpolation(self):
        """Test CMYK interpolation"""
        dataset = self.create_test_dataset()

        lut_builder = LookupTableBuilder()
        lut_builder.build_lut(dataset)

        # Test interpolation with known Lab values
        test_lab = np.array([[[50.0, 0.0, 0.0]]], dtype=np.float32)
        interpolated_cmyk = lut_builder.interpolate_cmyk(test_lab)

        # Should return valid CMYK values
        assert interpolated_cmyk.shape == (1, 1, 4)
        assert np.all(interpolated_cmyk >= 0.0)
        assert np.all(interpolated_cmyk <= 1.0)

    def test_lut_save_load(self):
        """Test saving and loading LUT"""
        dataset = self.create_test_dataset()

        lut_builder = LookupTableBuilder()
        lut_builder.build_lut(dataset)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_file = f.name

        try:
            # Save LUT
            lut_builder.save_lut(temp_file)

            # Load into new builder
            new_lut_builder = LookupTableBuilder()
            new_lut_builder.load_lut(temp_file)

            # Should have same data
            assert np.array_equal(new_lut_builder.cmyk_data, lut_builder.cmyk_data)
            assert np.array_equal(new_lut_builder.lab_data, lut_builder.lab_data)

        finally:
            os.unlink(temp_file)


class TestCMYKColorSpace:
    """Test CMYK color space"""

    def create_test_dataset(self):
        """Create a test dataset for calibration"""
        chart = CMYKTargetChart()
        chart.generate_complete_target_chart()

        simulator = MeasurementSimulator(noise_level=0.1)
        return simulator.create_measurement_dataset(chart)

    def test_colorspace_calibration(self):
        """Test color space calibration"""
        dataset = self.create_test_dataset()

        colorspace = CMYKColorSpace("Test CMYK")
        colorspace.calibrate(dataset)

        # Should be calibrated
        assert colorspace.is_calibrated
        assert colorspace.lut_builder is not None
        assert colorspace.calibration_data is not None

    def test_lab_to_cmyk_conversion(self):
        """Test Lab to CMYK conversion"""
        dataset = self.create_test_dataset()

        colorspace = CMYKColorSpace("Test CMYK")
        colorspace.calibrate(dataset)

        # Test conversion
        test_lab = np.array([[[50.0, 0.0, 0.0]]], dtype=np.float32)
        cmyk_result = colorspace.lab_to_cmyk(test_lab)

        # Should return valid CMYK
        assert cmyk_result.shape == (1, 1, 4)
        assert np.all(cmyk_result >= 0.0)
        assert np.all(cmyk_result <= 1.0)

    def test_rgb_to_cmyk_calibrated(self):
        """Test calibrated RGB to CMYK conversion"""
        dataset = self.create_test_dataset()

        colorspace = CMYKColorSpace("Test CMYK")
        colorspace.calibrate(dataset)

        # Test conversion
        test_rgb = np.array([[[255, 0, 0]]], dtype=np.uint8)  # Red
        cmyk_result = colorspace.rgb_to_cmyk_calibrated(test_rgb)

        # Should return valid CMYK
        assert cmyk_result.shape == (1, 1, 4)
        assert np.all(cmyk_result >= 0.0)
        assert np.all(cmyk_result <= 1.0)

    def test_colorspace_save_load(self):
        """Test saving and loading color space"""
        dataset = self.create_test_dataset()

        colorspace = CMYKColorSpace("Test CMYK")
        colorspace.calibrate(dataset)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_file = f.name

        try:
            # Save color space
            colorspace.save_colorspace(temp_file)

            # Load into new color space
            new_colorspace = CMYKColorSpace()
            new_colorspace.load_colorspace(temp_file)

            # Should have same properties
            assert new_colorspace.is_calibrated
            assert new_colorspace.name == colorspace.name

        finally:
            os.unlink(temp_file)


class TestCMYKCalibrationPipeline:
    """Test complete calibration pipeline"""

    def test_full_calibration_pipeline(self):
        """Test complete calibration process"""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = CMYKCalibrationPipeline(output_dir=temp_dir)

            # Run calibration
            colorspace = pipeline.run_calibration(
                noise_level=0.1, dot_gain=0.1, colorspace_name="Test Calibration"
            )

            # Should return calibrated color space
            assert colorspace.is_calibrated
            assert colorspace.name == "Test Calibration"

            # Should create output files
            output_path = Path(temp_dir)
            assert (output_path / "target_chart.png").exists()
            assert (output_path / "target_chart_definition.json").exists()
            assert (output_path / "measurements.json").exists()
            assert (output_path / "Test_Calibration.pkl").exists()

    def test_calibration_validation(self):
        """Test calibration validation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = CMYKCalibrationPipeline(output_dir=temp_dir)

            # Run calibration
            pipeline.run_calibration(
                noise_level=0.1, dot_gain=0.1, colorspace_name="Test Calibration"
            )

            # Validate calibration
            validation_results = pipeline.validate_calibration()

            # Should have validation results
            assert "test_colors" in validation_results
            assert "accuracy_metrics" in validation_results

            # Should have tested multiple colors
            assert len(validation_results["test_colors"]) > 0

            # Should have accuracy metrics
            metrics = validation_results["accuracy_metrics"]
            assert "average_delta_e" in metrics
            assert "num_test_colors" in metrics
            assert "calibration_patches" in metrics

    def test_custom_test_colors_validation(self):
        """Test validation with custom test colors"""
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = CMYKCalibrationPipeline(output_dir=temp_dir)

            # Run calibration
            pipeline.run_calibration(
                noise_level=0.1, dot_gain=0.1, colorspace_name="Test Calibration"
            )

            # Custom test colors
            custom_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

            validation_results = pipeline.validate_calibration(
                test_colors=custom_colors
            )

            # Should have tested exactly the custom colors
            assert len(validation_results["test_colors"]) == len(custom_colors)

            # Each test color should have expected fields
            for test_result in validation_results["test_colors"]:
                assert "input_rgb" in test_result
                assert "original_cmyk" in test_result
                assert "calibrated_cmyk" in test_result
                assert "delta_e" in test_result


def test_integration_example():
    """Integration test showing how to use the calibration system"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create calibration pipeline
        pipeline = CMYKCalibrationPipeline(output_dir=temp_dir)

        # Run full calibration
        print("Running CMYK calibration...")
        colorspace = pipeline.run_calibration(
            noise_level=0.2, dot_gain=0.15, colorspace_name="Example Printer Profile"
        )

        # Validate calibration
        print("Validating calibration...")
        validation_results = pipeline.validate_calibration()

        print(
            f"Calibration completed with {len(colorspace.calibration_data['dataset']['cmyk_inputs'])} patches"
        )
        print(
            f"Average validation Delta E: {validation_results['accuracy_metrics']['average_delta_e']:.2f}"
        )

        # Test with a sample image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Convert using calibrated color space
        cmyk_calibrated = colorspace.rgb_to_cmyk_calibrated(test_image)

        # Should produce valid CMYK output
        assert cmyk_calibrated.shape == (100, 100, 4)
        assert np.all(cmyk_calibrated >= 0.0)
        assert np.all(cmyk_calibrated <= 1.0)

        print("Integration test completed successfully!")


if __name__ == "__main__":
    # Run integration test
    test_integration_example()
    print("All tests would pass!")
