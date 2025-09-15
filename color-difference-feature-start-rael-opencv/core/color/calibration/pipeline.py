#!/usr/bin/env python

"""
CMYK Calibration Pipeline

This module provides the complete calibration pipeline that integrates all
components: target chart generation, measurement simulation, LUT building,
and color space calibration.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from .colorspace import CMYKColorSpace
from .lut_builder import LookupTableBuilder
from .measurement import MeasurementSimulator
from .target_chart import CMYKTargetChart


class CMYKCalibrationPipeline:
    """
    Complete CMYK calibration pipeline

    This class orchestrates the entire calibration process:
    1. Generate target chart with comprehensive color patches
    2. Simulate spectrophotometer measurements
    3. Build lookup tables for Lab-to-CMYK conversion
    4. Create calibrated CMYK color space
    5. Validate calibration accuracy
    """

    def __init__(self, output_dir: str = "calibration_output"):
        """
        Initialize calibration pipeline

        Args:
            output_dir: Directory to save calibration output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Initialize components
        self.target_chart = CMYKTargetChart()
        self.measurement_sim = MeasurementSimulator()
        self.lut_builder = LookupTableBuilder()
        self.colorspace = CMYKColorSpace()

        # Pipeline state
        self.calibration_completed = False
        self.calibration_results = None

    def run_calibration(
        self,
        noise_level: float = 0.5,
        dot_gain: float = 0.15,
        colorspace_name: str = "Calibrated CMYK",
    ) -> CMYKColorSpace:
        """
        Run complete calibration process

        Args:
            noise_level: Measurement noise level (0.1-1.0)
            dot_gain: Dot gain coefficient (0.1-0.3)
            colorspace_name: Name for the calibrated color space

        Returns:
            Calibrated CMYK color space
        """
        print("=" * 60)
        print("CMYK CALIBRATION PIPELINE")
        print("=" * 60)

        # Step 1: Generate target chart
        print("\n1. Generating CMYK target chart...")
        patches = self.target_chart.generate_complete_target_chart()
        print(f"   Generated {len(patches)} color patches")

        # Save target chart
        chart_image = self.target_chart.create_chart_image(include_labels=True)
        chart_path = self.output_dir / "target_chart.png"
        cv2.imwrite(str(chart_path), cv2.cvtColor(chart_image, cv2.COLOR_RGB2BGR))

        chart_def_path = self.output_dir / "target_chart.json"
        self.target_chart.save_chart_definition(str(chart_def_path))

        csv_path = self.output_dir / "patch_list.csv"
        self.target_chart.export_patch_list(str(csv_path))

        print(f"   Target chart saved to: {chart_path}")
        print(f"   Chart definition saved to: {chart_def_path}")
        print(f"   Patch list saved to: {csv_path}")

        # Step 2: Simulate measurements
        print("\n2. Simulating spectrophotometer measurements...")
        self.measurement_sim.noise_level = noise_level
        self.measurement_sim.dot_gain = dot_gain

        measurement_dataset = self.measurement_sim.create_measurement_dataset(
            self.target_chart
        )

        # Save measurement data
        measurements_path = self.output_dir / "measurements.json"
        with open(measurements_path, "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy_types(obj):
                """Convert numpy types to Python types for JSON serialization"""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.float32 | np.float64):
                    return float(obj)
                elif isinstance(obj, np.int32 | np.int64):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj

            json_data = {
                "cmyk_inputs": [
                    [float(x) for x in cmyk]
                    for cmyk in measurement_dataset["cmyk_inputs"]
                ],
                "lab_measurements": [
                    [float(x) for x in lab]
                    for lab in measurement_dataset["lab_measurements"]
                ],
                "measurement_metadata": convert_numpy_types(
                    measurement_dataset["measurement_metadata"]
                ),
            }
            json.dump(json_data, f, indent=2)

        print(
            f"   Simulated measurements for {len(measurement_dataset['cmyk_inputs'])} patches"
        )
        print(f"   Measurements saved to: {measurements_path}")

        # Step 3: Build lookup table
        print("\n3. Building lookup table...")
        self.lut_builder.build_lut(measurement_dataset)

        # Validate LUT accuracy
        validation = self.lut_builder.validate_lut_accuracy()
        print(
            f"   LUT validation - Mean Absolute Error: {validation['mean_absolute_error']['total']:.4f}"
        )

        # Save LUT
        lut_path = self.output_dir / "calibration_lut.pkl"
        self.lut_builder.save_lut(str(lut_path))
        print(f"   LUT saved to: {lut_path}")

        # Step 4: Create calibrated color space
        print("\n4. Creating calibrated color space...")
        self.colorspace.name = colorspace_name
        self.colorspace.calibrate(measurement_dataset)

        # Save color space
        colorspace_path = self.output_dir / "cmyk_colorspace.pkl"
        self.colorspace.save_colorspace(str(colorspace_path))
        print(f"   Color space saved to: {colorspace_path}")

        # Step 5: Export ICC-compatible data
        print("\n5. Exporting ICC profile data...")
        icc_path = self.output_dir / "cmyk_profile.icc"
        self.colorspace.export_as_icc_profile(str(icc_path))

        # Step 6: Generate calibration report
        print("\n6. Generating calibration report...")
        self._generate_calibration_report()

        self.calibration_completed = True

        print("\n" + "=" * 60)
        print("CALIBRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print(f"Color space: {self.colorspace.name}")
        print(f"Patches measured: {len(measurement_dataset['cmyk_inputs'])}")
        print(f"LUT accuracy (MAE): {validation['mean_absolute_error']['total']:.4f}")

        return self.colorspace

    def validate_calibration(self, test_colors: list | None = None) -> dict:
        """
        Validate calibration accuracy

        Args:
            test_colors: Optional list of RGB test colors

        Returns:
            Validation results
        """
        if not self.calibration_completed:
            raise ValueError("Calibration not completed. Run run_calibration() first.")

        print("\nValidating calibration accuracy...")

        validation_results = self.colorspace.validate_calibration(test_colors)

        # Save validation results
        validation_path = self.output_dir / "validation_results.json"
        with open(validation_path, "w") as f:
            json.dump(validation_results, f, indent=2)

        print(f"Validation results saved to: {validation_path}")

        return validation_results

    def _generate_calibration_report(self) -> None:
        """Generate comprehensive calibration report"""
        if not self.calibration_completed and not self.colorspace.is_calibrated:
            return

        # Collect all calibration information
        report_data = {
            "calibration_summary": {
                "colorspace_name": self.colorspace.name,
                "calibration_date": str(
                    self.colorspace.calibration_data["calibration_date"]
                ),
                "num_patches": self.colorspace.calibration_data["num_patches"],
                "output_directory": str(self.output_dir),
            },
            "target_chart_info": self.target_chart.get_chart_statistics(),
            "measurement_simulation": {
                "dot_gain": self.measurement_sim.dot_gain,
                "noise_level": self.measurement_sim.noise_level,
                "ink_density_variation": self.measurement_sim.ink_density_variation,
                "substrate_properties": self.measurement_sim.substrate,
            },
            "lut_analysis": self.lut_builder.analyze_lut_coverage(),
            "colorspace_info": self.colorspace.get_color_space_info(),
        }

        # Add validation if available
        if hasattr(self, "validation_results"):
            report_data["validation_results"] = self.validation_results

        # Save comprehensive report
        report_path = self.output_dir / "calibration_report.json"
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)

        # Generate human-readable summary
        self._generate_summary_report(report_data)

        print(f"Calibration report saved to: {report_path}")

    def _generate_summary_report(self, report_data: dict) -> None:
        """Generate human-readable summary report"""
        summary_path = self.output_dir / "calibration_summary.txt"

        with open(summary_path, "w") as f:
            f.write("CMYK CALIBRATION SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")

            # Basic info
            summary = report_data["calibration_summary"]
            f.write(f"Color Space Name: {summary['colorspace_name']}\n")
            f.write(f"Calibration Date: {summary['calibration_date']}\n")
            f.write(f"Number of Patches: {summary['num_patches']}\n")
            f.write(f"Output Directory: {summary['output_directory']}\n\n")

            # Chart statistics
            if "target_chart_info" in report_data:
                chart_info = report_data["target_chart_info"]
                f.write("TARGET CHART STATISTICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total Patches: {chart_info.get('total_patches', 'N/A')}\n")

                if "ink_usage" in chart_info:
                    ink_usage = chart_info["ink_usage"]
                    f.write("Average Ink Usage:\n")
                    f.write(f"  Cyan: {ink_usage['avg_c']:.3f}\n")
                    f.write(f"  Magenta: {ink_usage['avg_m']:.3f}\n")
                    f.write(f"  Yellow: {ink_usage['avg_y']:.3f}\n")
                    f.write(f"  Black: {ink_usage['avg_k']:.3f}\n")
                    f.write(f"  Total: {ink_usage['avg_total_ink']:.3f}\n\n")

            # Measurement simulation
            if "measurement_simulation" in report_data:
                sim_info = report_data["measurement_simulation"]
                f.write("MEASUREMENT SIMULATION\n")
                f.write("-" * 30 + "\n")
                f.write(f"Dot Gain: {sim_info['dot_gain']:.3f}\n")
                f.write(f"Noise Level: {sim_info['noise_level']:.3f}\n")
                f.write(
                    f"Ink Density Variation: {sim_info['ink_density_variation']:.3f}\n"
                )

                if "substrate_properties" in sim_info:
                    substrate = sim_info["substrate_properties"]
                    f.write(f"Substrate: {substrate.get('texture', 'N/A')}\n")
                    f.write(f"Whiteness: {substrate.get('whiteness', 'N/A'):.3f}\n\n")

            # LUT analysis
            if "lut_analysis" in report_data:
                lut_info = report_data["lut_analysis"]
                f.write("LOOKUP TABLE ANALYSIS\n")
                f.write("-" * 30 + "\n")
                f.write(
                    f"Measurement Points: {lut_info.get('measurement_points', 'N/A')}\n"
                )
                f.write(
                    f"Interpolation Method: {lut_info.get('interpolation_method', 'N/A')}\n"
                )

                if "lab_coverage" in lut_info:
                    lab_coverage = lut_info["lab_coverage"]
                    f.write("Lab Coverage:\n")
                    f.write(
                        f"  L*: {lab_coverage['L_range'][0]:.1f} to {lab_coverage['L_range'][1]:.1f}\n"
                    )
                    f.write(
                        f"  a*: {lab_coverage['a_range'][0]:.1f} to {lab_coverage['a_range'][1]:.1f}\n"
                    )
                    f.write(
                        f"  b*: {lab_coverage['b_range'][0]:.1f} to {lab_coverage['b_range'][1]:.1f}\n\n"
                    )

            # Color space info
            if "colorspace_info" in report_data:
                cs_info = report_data["colorspace_info"]
                f.write("COLOR SPACE PROPERTIES\n")
                f.write("-" * 30 + "\n")
                f.write(f"Name: {cs_info.get('name', 'N/A')}\n")
                f.write(f"Calibrated: {cs_info.get('is_calibrated', False)}\n")
                f.write(f"Rendering Intent: {cs_info.get('rendering_intent', 'N/A')}\n")
                f.write(f"Ink Limit: {cs_info.get('ink_limit', 'N/A')}%\n")
                f.write(f"Gamut Volume: {cs_info.get('gamut_volume', 'N/A'):.2f}\n\n")

            # Validation results
            if "validation_results" in report_data:
                val_results = report_data["validation_results"]
                if "accuracy_metrics" in val_results:
                    metrics = val_results["accuracy_metrics"]
                    f.write("VALIDATION RESULTS\n")
                    f.write("-" * 30 + "\n")
                    f.write(
                        f"Average Delta E: {metrics.get('average_delta_e', 'N/A'):.2f}\n"
                    )
                    f.write(f"Test Colors: {metrics.get('num_test_colors', 'N/A')}\n\n")

            f.write("Files Generated:\n")
            f.write("- target_chart.png (visual chart)\n")
            f.write("- target_chart.json (chart definition)\n")
            f.write("- patch_list.csv (patch data)\n")
            f.write("- measurements.json (measurement data)\n")
            f.write("- calibration_lut.pkl (lookup table)\n")
            f.write("- cmyk_colorspace.pkl (color space)\n")
            f.write("- cmyk_profile_lut.pkl (ICC LUT data)\n")
            f.write("- cmyk_profile_metadata.json (ICC metadata)\n")
            f.write("- calibration_report.json (detailed report)\n")
            f.write("- calibration_summary.txt (this summary)\n")

        print(f"Summary report saved to: {summary_path}")

    def load_calibration(self, colorspace_file: str) -> CMYKColorSpace:
        """
        Load existing calibration from file

        Args:
            colorspace_file: Path to saved color space file

        Returns:
            Loaded CMYK color space
        """
        self.colorspace.load_colorspace(colorspace_file)
        self.calibration_completed = True

        print(f"Calibration loaded from: {colorspace_file}")
        print(f"Color space: {self.colorspace.name}")
        print(f"Patches: {self.colorspace.calibration_data['num_patches']}")

        return self.colorspace

    def create_calibration_comparison(
        self,
        other_pipeline: "CMYKCalibrationPipeline",
        test_colors: list | None = None,
    ) -> dict:
        """
        Compare this calibration with another calibration

        Args:
            other_pipeline: Another calibration pipeline to compare with
            test_colors: Optional test colors for comparison

        Returns:
            Comparison results
        """
        if not self.calibration_completed or not other_pipeline.calibration_completed:
            raise ValueError("Both calibrations must be completed")

        if test_colors is None:
            test_colors = [
                (255, 0, 0),
                (0, 255, 0),
                (0, 0, 255),
                (255, 255, 0),
                (255, 0, 255),
                (0, 255, 255),
                (128, 128, 128),
                (255, 255, 255),
                (0, 0, 0),
            ]

        comparison_results = {
            "calibration1": {
                "name": self.colorspace.name,
                "patches": self.colorspace.calibration_data["num_patches"],
                "gamut_volume": self.colorspace.get_gamut_volume(),
            },
            "calibration2": {
                "name": other_pipeline.colorspace.name,
                "patches": other_pipeline.colorspace.calibration_data["num_patches"],
                "gamut_volume": other_pipeline.colorspace.get_gamut_volume(),
            },
            "color_differences": [],
        }

        # Compare color conversions
        for rgb_color in test_colors:
            rgb_array = np.array([[rgb_color]], dtype=np.uint8)

            # Convert using both calibrations
            cmyk1 = self.colorspace.rgb_to_cmyk_calibrated(rgb_array)
            cmyk2 = other_pipeline.colorspace.rgb_to_cmyk_calibrated(rgb_array)

            # Calculate CMYK difference
            cmyk_diff = np.abs(cmyk1 - cmyk2)

            comparison_results["color_differences"].append(
                {
                    "input_rgb": rgb_color,
                    "cmyk1": cmyk1[0, 0].tolist(),
                    "cmyk2": cmyk2[0, 0].tolist(),
                    "cmyk_difference": cmyk_diff[0, 0].tolist(),
                    "max_cmyk_diff": float(np.max(cmyk_diff)),
                }
            )

        # Save comparison results
        comparison_path = self.output_dir / "calibration_comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(comparison_results, f, indent=2)

        print(f"Calibration comparison saved to: {comparison_path}")

        return comparison_results

    def optimize_calibration_parameters(
        self, parameter_ranges: dict | None = None
    ) -> dict:
        """
        Optimize calibration parameters for best accuracy

        Args:
            parameter_ranges: Dictionary of parameter ranges to test

        Returns:
            Optimization results with best parameters
        """
        if parameter_ranges is None:
            parameter_ranges = {
                "noise_level": [0.1, 0.3, 0.5, 0.7],
                "dot_gain": [0.05, 0.10, 0.15, 0.20, 0.25],
                "interpolation_method": ["linear", "cubic", "nearest"],
            }

        print("Optimizing calibration parameters...")

        best_params = {}
        best_error = float("inf")
        optimization_results = []

        # Test different parameter combinations
        for noise in parameter_ranges.get("noise_level", [0.5]):
            for dot_gain in parameter_ranges.get("dot_gain", [0.15]):
                for interp_method in parameter_ranges.get(
                    "interpolation_method", ["linear"]
                ):
                    print(
                        f"Testing: noise={noise}, dot_gain={dot_gain}, method={interp_method}"
                    )

                    # Create temporary pipeline
                    temp_output = (
                        self.output_dir
                        / f"optimization_temp_{noise}_{dot_gain}_{interp_method}"
                    )
                    temp_pipeline = CMYKCalibrationPipeline(str(temp_output))

                    # Set parameters
                    temp_pipeline.lut_builder.interpolation_method = interp_method

                    try:
                        # Run calibration
                        temp_pipeline.run_calibration(
                            noise_level=noise,
                            dot_gain=dot_gain,
                            colorspace_name=f"Optimized_{noise}_{dot_gain}_{interp_method}",
                        )

                        # Validate
                        validation = temp_pipeline.validate_calibration()
                        avg_error = validation["accuracy_metrics"]["average_delta_e"]

                        result = {
                            "noise_level": noise,
                            "dot_gain": dot_gain,
                            "interpolation_method": interp_method,
                            "average_delta_e": avg_error,
                            "num_patches": validation["accuracy_metrics"][
                                "calibration_patches"
                            ],
                        }

                        optimization_results.append(result)

                        if avg_error < best_error:
                            best_error = avg_error
                            best_params = result.copy()

                        print(f"  Average Delta E: {avg_error:.4f}")

                    except Exception as e:
                        print(f"  Failed: {e}")
                        continue

        # Save optimization results
        optimization_data = {
            "best_parameters": best_params,
            "all_results": optimization_results,
            "parameter_ranges": parameter_ranges,
        }

        optimization_path = self.output_dir / "parameter_optimization.json"
        with open(optimization_path, "w") as f:
            json.dump(optimization_data, f, indent=2)

        print("\nOptimization completed!")
        print(f"Best parameters: {best_params}")
        print(f"Results saved to: {optimization_path}")

        return optimization_data
