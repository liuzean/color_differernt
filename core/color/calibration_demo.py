#!/usr/bin/env python

"""
CMYK Calibration System Demo

This script demonstrates how to use the CMYK calibration system with
practical examples and visualization of the calibration process.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from core.color.calibration import (
    CMYKCalibrationPipeline,
    CMYKTargetChart,
    MeasurementSimulator,
)
from core.color.utils import (
    calculate_color_difference,
    cmyk_to_rgb,
    rgb_to_cmyk,
    rgb_to_lab,
)

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def demonstrate_target_chart_creation():
    """Demonstrate target chart creation and visualization"""
    print("=" * 60)
    print("CMYK Target Chart Creation Demo")
    print("=" * 60)

    # Create target chart
    chart = CMYKTargetChart(patch_size=40, grid_spacing=8)

    # Generate different types of patches
    primaries = chart.generate_primary_patches()
    gradations = chart.generate_gradation_patches(steps=8)
    overprints = chart.generate_overprint_patches()
    grays = chart.generate_gray_patches(steps=8)

    print("Generated patches:")
    print(f"  - Primary patches: {len(primaries)}")
    print(f"  - Gradation patches: {len(gradations)}")
    print(f"  - Overprint patches: {len(overprints)}")
    print(f"  - Gray patches: {len(grays)}")

    # Generate complete target chart
    all_patches = chart.generate_complete_target_chart()
    print(f"  - Total unique patches: {len(all_patches)}")

    # Create chart image
    chart_image = chart.create_chart_image()
    print(f"Chart image dimensions: {chart_image.shape}")

    # Save chart
    output_dir = Path("calibration_demo_output")
    output_dir.mkdir(exist_ok=True)

    chart_path = output_dir / "target_chart.png"
    cv2.imwrite(str(chart_path), chart_image)
    print(f"Target chart saved to: {chart_path}")

    # Save chart definition
    chart_def_path = output_dir / "target_chart_definition.json"
    chart.save_chart_definition(str(chart_def_path))
    print(f"Chart definition saved to: {chart_def_path}")

    return chart, chart_image


def demonstrate_measurement_simulation():
    """Demonstrate measurement simulation"""
    print("\n" + "=" * 60)
    print("Measurement Simulation Demo")
    print("=" * 60)

    # Create measurement simulator
    simulator = MeasurementSimulator(noise_level=0.3, dot_gain=0.12)

    # Test with a few key patches
    test_patches = [
        (0.0, 0.0, 0.0, 0.0),  # White
        (1.0, 0.0, 0.0, 0.0),  # Cyan
        (0.0, 1.0, 0.0, 0.0),  # Magenta
        (0.0, 0.0, 1.0, 0.0),  # Yellow
        (0.0, 0.0, 0.0, 1.0),  # Black
        (0.5, 0.5, 0.5, 0.0),  # CMY Gray
        (0.0, 0.0, 0.0, 0.5),  # K Gray
        (1.0, 1.0, 1.0, 0.0),  # C+M+Y
    ]

    print("Simulating measurements for key patches:")
    print("CMYK Input -> Lab Measurement")
    print("-" * 40)

    lab_measurements = simulator.simulate_measurement(test_patches)

    for i, (cmyk, lab) in enumerate(zip(test_patches, lab_measurements, strict=False)):
        patch_name = [
            "White",
            "Cyan",
            "Magenta",
            "Yellow",
            "Black",
            "CMY Gray",
            "K Gray",
            "C+M+Y",
        ][i]
        print(
            f"{patch_name:10} {cmyk} -> L={lab[0]:.1f}, a={lab[1]:+.1f}, b={lab[2]:+.1f}"
        )

    return simulator


def demonstrate_calibration_pipeline():
    """Demonstrate complete calibration pipeline"""
    print("\n" + "=" * 60)
    print("Complete Calibration Pipeline Demo")
    print("=" * 60)

    # Create calibration pipeline
    output_dir = Path("calibration_demo_output")
    pipeline = CMYKCalibrationPipeline(output_dir=str(output_dir))

    # Run calibration with different printer characteristics
    print("Running calibration for 'Demo Printer Profile'...")
    colorspace = pipeline.run_calibration(
        noise_level=0.2, dot_gain=0.15, colorspace_name="Demo Printer Profile"
    )

    print("Calibration completed!")
    print(f"Color space name: {colorspace.name}")
    print(f"Calibrated: {colorspace.is_calibrated}")
    print(
        f"Number of calibration patches: {len(colorspace.calibration_data['dataset']['cmyk_inputs'])}"
    )

    return pipeline, colorspace


def demonstrate_calibration_validation():
    """Demonstrate calibration validation"""
    print("\n" + "=" * 60)
    print("Calibration Validation Demo")
    print("=" * 60)

    # Create and run calibration
    output_dir = Path("calibration_demo_output")
    pipeline = CMYKCalibrationPipeline(output_dir=str(output_dir))

    pipeline.run_calibration(
        noise_level=0.1, dot_gain=0.1, colorspace_name="Validation Test"
    )

    # Validate with default test colors
    validation_results = pipeline.validate_calibration()

    print("Validation Results:")
    print("-" * 40)

    metrics = validation_results["accuracy_metrics"]
    print(f"Average Delta E: {metrics['average_delta_e']:.2f}")
    print(f"Number of test colors: {metrics['num_test_colors']}")
    print(f"Calibration patches used: {metrics['calibration_patches']}")

    print("\nDetailed per-color results:")
    for i, result in enumerate(validation_results["test_colors"][:5]):  # Show first 5
        rgb = result["input_rgb"]
        delta_e = result["delta_e"]
        print(f"Color {i + 1}: RGB{rgb} -> Delta E: {delta_e:.2f}")

    return validation_results


def demonstrate_color_conversion_comparison():
    """Demonstrate comparison between standard and calibrated CMYK conversion"""
    print("\n" + "=" * 60)
    print("Color Conversion Comparison Demo")
    print("=" * 60)

    # Create calibrated color space
    output_dir = Path("calibration_demo_output")
    pipeline = CMYKCalibrationPipeline(output_dir=str(output_dir))

    colorspace = pipeline.run_calibration(
        noise_level=0.15, dot_gain=0.12, colorspace_name="Comparison Test"
    )

    # Test colors
    test_colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 128, 128),  # Gray
        (255, 128, 64),  # Orange
    ]

    print("Comparing standard vs calibrated CMYK conversion:")
    print("Color      | Standard CMYK      | Calibrated CMYK    | Delta E")
    print("-" * 70)

    for rgb_color in test_colors:
        # Convert to numpy array
        rgb_array = np.array([[rgb_color]], dtype=np.uint8)

        # Standard conversion
        standard_cmyk = rgb_to_cmyk(rgb_array)
        cmyk_to_rgb(standard_cmyk)

        # Calibrated conversion
        calibrated_cmyk = colorspace.rgb_to_cmyk_calibrated(rgb_array)
        calibrated_rgb_back = cmyk_to_rgb(calibrated_cmyk)

        # Calculate Delta E between original and reconstructed
        rgb_to_lab(rgb_array)
        rgb_to_lab(calibrated_rgb_back.astype(np.uint8))

        # Use the existing calculate_delta_e function
        _, delta_e = calculate_color_difference(
            rgb_array, calibrated_rgb_back.astype(np.uint8)
        )

        # Format output
        color_name = f"({rgb_color[0]:3d},{rgb_color[1]:3d},{rgb_color[2]:3d})"
        standard_str = f"({standard_cmyk[0, 0, 0]:.2f},{standard_cmyk[0, 0, 1]:.2f},{standard_cmyk[0, 0, 2]:.2f},{standard_cmyk[0, 0, 3]:.2f})"
        calibrated_str = f"({calibrated_cmyk[0, 0, 0]:.2f},{calibrated_cmyk[0, 0, 1]:.2f},{calibrated_cmyk[0, 0, 2]:.2f},{calibrated_cmyk[0, 0, 3]:.2f})"

        print(
            f"{color_name:10} | {standard_str:18} | {calibrated_str:18} | {delta_e:6.2f}"
        )

    return colorspace


def demonstrate_image_processing():
    """Demonstrate processing a sample image with calibration"""
    print("\n" + "=" * 60)
    print("Image Processing Demo")
    print("=" * 60)

    # Create calibrated color space
    output_dir = Path("calibration_demo_output")
    pipeline = CMYKCalibrationPipeline(output_dir=str(output_dir))

    colorspace = pipeline.run_calibration(
        noise_level=0.1, dot_gain=0.1, colorspace_name="Image Processing Demo"
    )

    # Create a sample image with various colors
    print("Creating sample image with various colors...")
    sample_image = np.zeros((200, 300, 3), dtype=np.uint8)

    # Add colored rectangles
    colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]

    for i, color in enumerate(colors):
        x = (i % 3) * 100
        y = (i // 3) * 100
        sample_image[y : y + 100, x : x + 100] = color

    # Save original image
    original_path = output_dir / "sample_original.png"
    cv2.imwrite(str(original_path), sample_image)
    print(f"Original image saved to: {original_path}")

    # Convert using standard CMYK
    print("Converting image using standard CMYK...")
    standard_cmyk = rgb_to_cmyk(sample_image)
    standard_back = cmyk_to_rgb(standard_cmyk)

    standard_path = output_dir / "sample_standard_cmyk.png"
    cv2.imwrite(str(standard_path), standard_back)
    print(f"Standard CMYK result saved to: {standard_path}")

    # Convert using calibrated CMYK
    print("Converting image using calibrated CMYK...")
    calibrated_cmyk = colorspace.rgb_to_cmyk_calibrated(sample_image)
    calibrated_back = cmyk_to_rgb(calibrated_cmyk)

    calibrated_path = output_dir / "sample_calibrated_cmyk.png"
    cv2.imwrite(str(calibrated_path), calibrated_back.astype(np.uint8))
    print(f"Calibrated CMYK result saved to: {calibrated_path}")

    # Calculate and display differences
    avg_delta_e, delta_e_map = calculate_color_difference(
        sample_image, calibrated_back.astype(np.uint8)
    )
    print("\nColor difference analysis:")
    print(f"Average Delta E: {avg_delta_e:.2f}")
    print(f"Max Delta E: {np.max(delta_e_map):.2f}")
    print(f"Min Delta E: {np.min(delta_e_map):.2f}")

    return sample_image, calibrated_cmyk


def create_calibration_summary_report():
    """Create a comprehensive calibration summary report"""
    print("\n" + "=" * 60)
    print("Calibration Summary Report")
    print("=" * 60)

    # Create multiple calibration profiles with different characteristics
    output_dir = Path("calibration_demo_output")

    profiles = [
        {"name": "Low Noise Profile", "noise": 0.1, "dot_gain": 0.05},
        {"name": "Medium Noise Profile", "noise": 0.2, "dot_gain": 0.10},
        {"name": "High Noise Profile", "noise": 0.3, "dot_gain": 0.15},
        {"name": "High Dot Gain Profile", "noise": 0.1, "dot_gain": 0.20},
    ]

    print("Creating calibration profiles with different characteristics:")
    print("Profile Name               | Noise | Dot Gain | Patches | Avg Delta E")
    print("-" * 75)

    for profile in profiles:
        pipeline = CMYKCalibrationPipeline(output_dir=str(output_dir))

        pipeline.run_calibration(
            noise_level=profile["noise"],
            dot_gain=profile["dot_gain"],
            colorspace_name=profile["name"],
        )

        # Quick validation
        validation = pipeline.validate_calibration()
        avg_delta_e = validation["accuracy_metrics"]["average_delta_e"]
        num_patches = validation["accuracy_metrics"]["calibration_patches"]

        print(
            f"{profile['name']:25} | {profile['noise']:5.1f} | {profile['dot_gain']:8.2f} | {num_patches:7d} | {avg_delta_e:11.2f}"
        )

    print("\nCalibration profiles created successfully!")
    print(f"All results saved to: {output_dir.absolute()}")


def main():
    """Main demonstration function"""
    print("CMYK Calibration System - Complete Demo")
    print("=" * 60)

    try:
        # Create output directory
        output_dir = Path("calibration_demo_output")
        output_dir.mkdir(exist_ok=True)

        # Run all demonstrations
        chart, chart_image = demonstrate_target_chart_creation()
        demonstrate_measurement_simulation()
        pipeline, colorspace = demonstrate_calibration_pipeline()
        demonstrate_calibration_validation()
        demonstrate_color_conversion_comparison()
        sample_image, calibrated_cmyk = demonstrate_image_processing()
        create_calibration_summary_report()

        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        print(f"All results saved to: {output_dir.absolute()}")
        print("\nGenerated files:")
        for file_path in sorted(output_dir.glob("*")):
            print(f"  - {file_path.name}")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
