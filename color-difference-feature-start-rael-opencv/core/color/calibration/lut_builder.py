#!/usr/bin/env python

"""
Lookup Table Builder Module

This module builds multi-dimensional lookup tables (LUTs) for Lab-to-CMYK
conversion using measurement data. It implements interpolation algorithms
for colors that fall between measured points.
"""

import pickle

import numpy as np
from scipy.interpolate import LinearNDInterpolator, griddata
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist


class LookupTableBuilder:
    """
    Builds multi-dimensional lookup tables for Lab-to-CMYK conversion

    This class creates the complex, multi-dimensional lookup table that maps
    Lab values to CMYK values. It uses sophisticated interpolation algorithms
    to find the correct CMYK value for Lab colors that weren't directly measured.
    """

    def __init__(self, interpolation_method: str = "linear"):
        """
        Initialize LUT builder

        Args:
            interpolation_method: Interpolation method ('linear', 'cubic', 'nearest')
        """
        self.interpolation_method = interpolation_method
        self.cmyk_data = None
        self.lab_data = None
        self.lut_data = None
        self.interpolators = {}
        self.gamut_hull = None

        # LUT resolution settings
        self.lut_resolution = {
            "L": 33,  # L* axis resolution (0-100)
            "a": 33,  # a* axis resolution (-128 to +127)
            "b": 33,  # b* axis resolution (-128 to +127)
        }

    def build_lut(self, measurement_dataset: dict) -> None:
        """
        Build lookup table from measurement dataset

        Args:
            measurement_dataset: Dataset with CMYK inputs and Lab measurements
        """
        cmyk_inputs = np.array(measurement_dataset["cmyk_inputs"])
        lab_measurements = np.array(measurement_dataset["lab_measurements"])

        self.cmyk_data = cmyk_inputs
        self.lab_data = lab_measurements

        print(f"Building LUT from {len(cmyk_inputs)} measurement points...")

        # Build interpolators for each CMYK channel
        self._build_interpolators()

        # Create gamut boundary
        self._build_gamut_boundary()

        # Store the data for interpolation
        self.lut_data = {
            "cmyk_points": cmyk_inputs,
            "lab_points": lab_measurements,
            "interpolation_method": self.interpolation_method,
            "lut_resolution": self.lut_resolution,
            "gamut_boundary": self.gamut_hull,
        }

        print("LUT building completed successfully!")

    def _build_interpolators(self) -> None:
        """Build interpolators for each CMYK channel"""
        self.interpolators = {}

        for channel in range(4):
            channel_name = ["C", "M", "Y", "K"][channel]

            try:
                if self.interpolation_method == "linear":
                    # Use LinearNDInterpolator for better performance
                    interpolator = LinearNDInterpolator(
                        self.lab_data, self.cmyk_data[:, channel], fill_value=0.0
                    )
                else:
                    # Use griddata for other methods
                    def interpolator(points):
                        return griddata(
                            self.lab_data,
                            self.cmyk_data[:, channel],
                            points,
                            method=self.interpolation_method,
                            fill_value=0.0,
                        )

                self.interpolators[channel_name] = interpolator

            except Exception as e:
                print(
                    f"Warning: Failed to create interpolator for channel {channel_name}: {e}"
                )

                # Fallback to nearest neighbor
                def interpolator(points):
                    return griddata(
                        self.lab_data,
                        self.cmyk_data[:, channel],
                        points,
                        method="nearest",
                        fill_value=0.0,
                    )

                self.interpolators[channel_name] = interpolator

    def _build_gamut_boundary(self) -> None:
        """Build gamut boundary for out-of-gamut detection"""
        try:
            # Create convex hull of measured Lab points
            self.gamut_hull = Delaunay(self.lab_data)
        except Exception as e:
            print(f"Warning: Could not build gamut boundary: {e}")
            self.gamut_hull = None

    def is_in_gamut(self, lab_points: np.ndarray) -> np.ndarray:
        """
        Check if Lab points are within the printer gamut

        Args:
            lab_points: Lab values to check, shape (N, 3)

        Returns:
            Boolean array indicating if points are in gamut
        """
        if self.gamut_hull is None:
            # If no gamut boundary, assume all points are in gamut
            return np.ones(len(lab_points), dtype=bool)

        try:
            # Use Delaunay triangulation to check if points are inside convex hull
            return self.gamut_hull.find_simplex(lab_points) >= 0
        except Exception:
            return np.ones(len(lab_points), dtype=bool)

    def find_gamut_boundary_point(self, lab_point: np.ndarray) -> np.ndarray:
        """
        Find the closest point on the gamut boundary for out-of-gamut colors

        Args:
            lab_point: Lab point outside gamut, shape (3,)

        Returns:
            Closest Lab point on gamut boundary
        """
        if self.gamut_hull is None or self.lab_data is None:
            return lab_point

        # Find closest measured point
        distances = cdist([lab_point], self.lab_data)[0]
        closest_idx = np.argmin(distances)

        return self.lab_data[closest_idx]

    def interpolate_cmyk(
        self, target_lab: np.ndarray, _recursion_depth: int = 0
    ) -> np.ndarray:
        """
        Interpolate CMYK values for given Lab coordinates

        Args:
            target_lab: Lab values to convert, shape (N, 3) or (3,)

        Returns:
            Interpolated CMYK values, shape (N, 4) or (4,)
        """
        if self.cmyk_data is None or self.lab_data is None:
            raise ValueError("LUT not built. Call build_lut() first.")

        # Safety check to prevent infinite recursion
        if _recursion_depth > 3:
            # Return a safe fallback CMYK value
            if target_lab.ndim == 1:
                return np.array([0.5, 0.5, 0.5, 0.1])  # Default grayish color
            else:
                return np.full((len(target_lab), 4), [0.5, 0.5, 0.5, 0.1])

        # Ensure target_lab is 2D
        if target_lab.ndim == 1:
            target_lab = target_lab.reshape(1, -1)
            single_point = True
        else:
            single_point = False

        # Initialize result array
        cmyk_results = np.zeros((len(target_lab), 4))

        # Check gamut for all points
        in_gamut = self.is_in_gamut(target_lab)

        # Process in-gamut points
        if np.any(in_gamut):
            in_gamut_points = target_lab[in_gamut]

            for channel in range(4):
                channel_name = ["C", "M", "Y", "K"][channel]

                try:
                    if (
                        self.interpolation_method == "linear"
                        and channel_name in self.interpolators
                    ):
                        # Use pre-built LinearNDInterpolator
                        interpolated = self.interpolators[channel_name](in_gamut_points)
                    else:
                        # Use griddata
                        interpolated = griddata(
                            self.lab_data,
                            self.cmyk_data[:, channel],
                            in_gamut_points,
                            method=self.interpolation_method,
                            fill_value=0.0,
                        )

                    cmyk_results[in_gamut, channel] = interpolated

                except Exception as e:
                    print(
                        f"Warning: Interpolation failed for channel {channel}, using nearest neighbor: {e}"
                    )
                    interpolated = griddata(
                        self.lab_data,
                        self.cmyk_data[:, channel],
                        in_gamut_points,
                        method="nearest",
                        fill_value=0.0,
                    )
                    cmyk_results[in_gamut, channel] = interpolated

        # Process out-of-gamut points
        if np.any(~in_gamut):
            out_of_gamut_points = target_lab[~in_gamut]

            for i, lab_point in enumerate(out_of_gamut_points):
                # Find closest gamut boundary point
                boundary_point = self.find_gamut_boundary_point(lab_point)

                # Interpolate at boundary point
                boundary_cmyk = self.interpolate_cmyk(
                    boundary_point.reshape(1, -1), _recursion_depth + 1
                )

                # Apply gamut mapping (simple approach: use boundary CMYK)
                out_of_gamut_idx = np.where(~in_gamut)[0][i]
                cmyk_results[out_of_gamut_idx] = boundary_cmyk[0]

        # Ensure valid CMYK range
        cmyk_results = np.clip(cmyk_results, 0, 1)

        # Apply ink limiting
        cmyk_results = self._apply_ink_limiting(cmyk_results)

        if single_point:
            return cmyk_results[0]
        else:
            return cmyk_results

    def _apply_ink_limiting(self, cmyk_values: np.ndarray) -> np.ndarray:
        """
        Apply total ink limiting to prevent excessive ink coverage

        Args:
            cmyk_values: CMYK values, shape (N, 4)

        Returns:
            CMYK values with ink limiting applied
        """
        # Maximum total ink coverage (typical printing limit)
        max_total_ink = 3.5

        # Calculate total ink for each point
        total_ink = np.sum(cmyk_values, axis=1)

        # Find points that exceed the limit
        excessive_points = total_ink > max_total_ink

        if np.any(excessive_points):
            # Scale down excessive points proportionally
            scale_factors = max_total_ink / total_ink[excessive_points]
            cmyk_values[excessive_points] *= scale_factors.reshape(-1, 1)

        return cmyk_values

    def create_3d_lut_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Create a regular 3D grid in Lab space for LUT generation

        Returns:
            Tuple of (lab_grid_points, cmyk_grid_values)
        """
        # Create regular grid in Lab space
        L_range = np.linspace(0, 100, self.lut_resolution["L"])
        a_range = np.linspace(-128, 127, self.lut_resolution["a"])
        b_range = np.linspace(-128, 127, self.lut_resolution["b"])

        # Create meshgrid
        L_grid, a_grid, b_grid = np.meshgrid(L_range, a_range, b_range, indexing="ij")

        # Flatten to get all grid points
        lab_grid_points = np.column_stack(
            [L_grid.flatten(), a_grid.flatten(), b_grid.flatten()]
        )

        # Interpolate CMYK values for all grid points
        print(f"Interpolating CMYK values for {len(lab_grid_points)} grid points...")
        cmyk_grid_values = self.interpolate_cmyk(lab_grid_points)

        return lab_grid_points, cmyk_grid_values

    def validate_lut_accuracy(self, test_fraction: float = 0.1) -> dict:
        """
        Validate LUT accuracy using a subset of measurement data

        Args:
            test_fraction: Fraction of data to use for testing

        Returns:
            Validation results
        """
        if self.cmyk_data is None or self.lab_data is None:
            raise ValueError("LUT not built. Call build_lut() first.")

        # Select random subset for testing
        n_test = max(1, int(len(self.lab_data) * test_fraction))
        test_indices = np.random.choice(len(self.lab_data), n_test, replace=False)

        test_lab = self.lab_data[test_indices]
        test_cmyk_actual = self.cmyk_data[test_indices]

        # Predict CMYK using LUT
        test_cmyk_predicted = self.interpolate_cmyk(test_lab)

        # Calculate errors
        cmyk_errors = np.abs(test_cmyk_predicted - test_cmyk_actual)

        # Calculate statistics
        results = {
            "num_test_points": n_test,
            "mean_absolute_error": {
                "C": float(np.mean(cmyk_errors[:, 0])),
                "M": float(np.mean(cmyk_errors[:, 1])),
                "Y": float(np.mean(cmyk_errors[:, 2])),
                "K": float(np.mean(cmyk_errors[:, 3])),
                "total": float(np.mean(cmyk_errors)),
            },
            "max_absolute_error": {
                "C": float(np.max(cmyk_errors[:, 0])),
                "M": float(np.max(cmyk_errors[:, 1])),
                "Y": float(np.max(cmyk_errors[:, 2])),
                "K": float(np.max(cmyk_errors[:, 3])),
                "total": float(np.max(cmyk_errors)),
            },
            "rms_error": {
                "C": float(np.sqrt(np.mean(cmyk_errors[:, 0] ** 2))),
                "M": float(np.sqrt(np.mean(cmyk_errors[:, 1] ** 2))),
                "Y": float(np.sqrt(np.mean(cmyk_errors[:, 2] ** 2))),
                "K": float(np.sqrt(np.mean(cmyk_errors[:, 3] ** 2))),
                "total": float(np.sqrt(np.mean(cmyk_errors**2))),
            },
        }

        return results

    def optimize_interpolation_method(self, methods: list[str] = None) -> str:
        """
        Test different interpolation methods and return the best one

        Args:
            methods: List of methods to test

        Returns:
            Best interpolation method name
        """
        if methods is None:
            methods = ["linear", "cubic", "nearest"]

        best_method = self.interpolation_method
        best_error = float("inf")

        for method in methods:
            try:
                self.interpolation_method = method
                self._build_interpolators()

                validation = self.validate_lut_accuracy()
                total_error = validation["mean_absolute_error"]["total"]

                print(f"Method {method}: MAE = {total_error:.4f}")

                if total_error < best_error:
                    best_error = total_error
                    best_method = method

            except Exception as e:
                print(f"Method {method} failed: {e}")

        # Restore best method
        self.interpolation_method = best_method
        self._build_interpolators()

        print(f"Best interpolation method: {best_method} (MAE = {best_error:.4f})")
        return best_method

    def save_lut(self, filename: str) -> None:
        """
        Save lookup table to file

        Args:
            filename: Path to save the LUT
        """
        if self.lut_data is None:
            raise ValueError("No LUT data to save")

        with open(filename, "wb") as f:
            pickle.dump(self.lut_data, f)

        print(f"LUT saved to {filename}")

    def load_lut(self, filename: str) -> None:
        """
        Load lookup table from file

        Args:
            filename: Path to load the LUT from
        """
        with open(filename, "rb") as f:
            self.lut_data = pickle.load(f)

        self.cmyk_data = self.lut_data["cmyk_points"]
        self.lab_data = self.lut_data["lab_points"]
        self.interpolation_method = self.lut_data["interpolation_method"]

        if "lut_resolution" in self.lut_data:
            self.lut_resolution = self.lut_data["lut_resolution"]

        if "gamut_boundary" in self.lut_data:
            self.gamut_hull = self.lut_data["gamut_boundary"]

        # Rebuild interpolators
        self._build_interpolators()

        print(f"LUT loaded from {filename}")

    def export_lut_as_icc_data(self, filename: str) -> None:
        """
        Export LUT data in a format suitable for ICC profile creation

        Args:
            filename: Path to save ICC-compatible data
        """
        lab_grid, cmyk_grid = self.create_3d_lut_grid()

        # Prepare ICC-compatible data structure
        icc_data = {
            "input_channels": 3,  # Lab
            "output_channels": 4,  # CMYK
            "grid_points": self.lut_resolution,
            "input_table": lab_grid,
            "output_table": cmyk_grid,
            "color_space": "Lab",
            "device_space": "CMYK",
            "interpolation": self.interpolation_method,
            "creation_info": {
                "measurement_points": len(self.lab_data),
                "gamut_mapping": "boundary_projection",
                "ink_limiting": "enabled",
            },
        }

        with open(filename, "wb") as f:
            pickle.dump(icc_data, f)

        print(f"ICC-compatible LUT data exported to {filename}")

    def analyze_lut_coverage(self) -> dict:
        """
        Analyze the coverage and quality of the LUT

        Returns:
            Analysis results
        """
        if self.lab_data is None:
            return {}

        # Calculate Lab space coverage
        lab_ranges = {
            "L_range": [
                float(np.min(self.lab_data[:, 0])),
                float(np.max(self.lab_data[:, 0])),
            ],
            "a_range": [
                float(np.min(self.lab_data[:, 1])),
                float(np.max(self.lab_data[:, 1])),
            ],
            "b_range": [
                float(np.min(self.lab_data[:, 2])),
                float(np.max(self.lab_data[:, 2])),
            ],
        }

        # Calculate CMYK space coverage
        cmyk_ranges = {
            "C_range": [
                float(np.min(self.cmyk_data[:, 0])),
                float(np.max(self.cmyk_data[:, 0])),
            ],
            "M_range": [
                float(np.min(self.cmyk_data[:, 1])),
                float(np.max(self.cmyk_data[:, 1])),
            ],
            "Y_range": [
                float(np.min(self.cmyk_data[:, 2])),
                float(np.max(self.cmyk_data[:, 2])),
            ],
            "K_range": [
                float(np.min(self.cmyk_data[:, 3])),
                float(np.max(self.cmyk_data[:, 3])),
            ],
        }

        # Calculate point density
        lab_volume = np.prod([r[1] - r[0] for r in lab_ranges.values()])
        point_density = len(self.lab_data) / lab_volume if lab_volume > 0 else 0

        # Calculate uniformity
        if len(self.lab_data) > 1:
            distances = cdist(self.lab_data, self.lab_data)
            np.fill_diagonal(distances, np.inf)  # Ignore self-distances
            min_distances = np.min(distances, axis=1)
            uniformity = {
                "mean_nearest_distance": float(np.mean(min_distances)),
                "std_nearest_distance": float(np.std(min_distances)),
                "uniformity_ratio": float(
                    np.std(min_distances) / np.mean(min_distances)
                ),
            }
        else:
            uniformity = {
                "mean_nearest_distance": 0,
                "std_nearest_distance": 0,
                "uniformity_ratio": 0,
            }

        return {
            "measurement_points": len(self.lab_data),
            "lab_coverage": lab_ranges,
            "cmyk_coverage": cmyk_ranges,
            "point_density": point_density,
            "uniformity": uniformity,
            "interpolation_method": self.interpolation_method,
            "lut_resolution": self.lut_resolution,
            "has_gamut_boundary": self.gamut_hull is not None,
        }
