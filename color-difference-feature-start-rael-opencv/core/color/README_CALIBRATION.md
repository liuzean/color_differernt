# CMYK Calibration System

This module implements a comprehensive CMYK calibration system based on ICC profile workflow standards. It provides tools for creating target charts, simulating measurement data, building lookup tables, and performing calibrated color conversions.

## Overview

The CMYK calibration system follows the standard workflow for printer characterization:

1. **Target Chart Generation**: Creates digital CMYK target charts with various color patches
2. **Measurement Simulation**: Simulates spectrophotometer measurements (since we don't have real hardware)
3. **Lookup Table Building**: Creates multi-dimensional lookup tables for Lab-to-CMYK conversion
4. **Color Space Calibration**: Provides calibrated color conversion functionality

## Key Features

- **Comprehensive Target Charts**: Generates charts with primaries, gradations, overprints, grays, and gamut sampling
- **Realistic Measurement Simulation**: Includes dot gain, printer characteristics, and measurement noise
- **Advanced Interpolation**: Uses scipy's griddata for robust CMYK interpolation
- **ICC-Profile Workflow**: Follows standard color management practices
- **Validation Tools**: Built-in validation and accuracy assessment
- **Save/Load Functionality**: Persistent storage of calibration data

## Installation Requirements

The calibration system uses the existing project dependencies:
- OpenCV for color space conversions
- NumPy for array operations
- SciPy for interpolation
- Matplotlib for visualization (optional)

## Quick Start

### Basic Calibration

```python
from core.color.calibration import CMYKCalibrationPipeline

# Create and run calibration
pipeline = CMYKCalibrationPipeline(output_dir="my_calibration")
colorspace = pipeline.run_calibration(
    noise_level=0.2,
    dot_gain=0.15,
    colorspace_name="My Printer Profile"
)

# Use calibrated conversion
import numpy as np
rgb_image = np.array([[[255, 0, 0]]], dtype=np.uint8)  # Red pixel
cmyk_calibrated = colorspace.rgb_to_cmyk_calibrated(rgb_image)
print(f"Calibrated CMYK: {cmyk_calibrated[0, 0]}")
```

### Validation

```python
# Validate calibration accuracy
validation_results = pipeline.validate_calibration()
print(f"Average Delta E: {validation_results['accuracy_metrics']['average_delta_e']:.2f}")
```

## Module Components

### 1. CMYKTargetChart

Creates standardized target charts for calibration.

```python
from core.color.calibration import CMYKTargetChart

chart = CMYKTargetChart(patch_size=50, grid_spacing=10)
patches = chart.generate_complete_target_chart()
chart_image = chart.create_chart_image()

print(f"Generated {len(patches)} patches")
```

**Patch Types Generated:**
- **Primary patches**: 100% C, M, Y, K
- **Gradation patches**: 10%, 20%, ..., 90% for each ink
- **Overprint patches**: Combinations like C+M, C+Y, etc.
- **Gray patches**: K-only and CMY composite grays
- **Gamut sampling**: Various CMYK combinations

### 2. MeasurementSimulator

Simulates spectrophotometer measurements with realistic printer characteristics.

```python
from core.color.calibration import MeasurementSimulator

simulator = MeasurementSimulator(noise_level=0.3, dot_gain=0.15)
lab_measurements = simulator.simulate_measurement(cmyk_patches)
```

**Simulation Features:**
- **Dot gain modeling**: Simulates ink spreading in midtones
- **Measurement noise**: Adds realistic sensor noise
- **Printer characteristics**: Models different ink densities and hue shifts

### 3. LookupTableBuilder

Builds multi-dimensional lookup tables for color conversion.

```python
from core.color.calibration import LookupTableBuilder

lut_builder = LookupTableBuilder(interpolation_method='linear')
lut_builder.build_lut(measurement_dataset)
cmyk_result = lut_builder.interpolate_cmyk(lab_colors)
```

**Interpolation Methods:**
- **Linear**: Fast, good for most applications
- **Cubic**: Higher quality, slower
- **Nearest**: Fallback method

### 4. CMYKColorSpace

Manages calibrated color space with conversion functions.

```python
from core.color.calibration import CMYKColorSpace

colorspace = CMYKColorSpace("My Printer")
colorspace.calibrate(measurement_dataset)

# Convert colors
cmyk_result = colorspace.rgb_to_cmyk_calibrated(rgb_image)
cmyk_direct = colorspace.lab_to_cmyk(lab_image)
```

### 5. CMYKCalibrationPipeline

Complete end-to-end calibration workflow.

```python
from core.color.calibration import CMYKCalibrationPipeline

pipeline = CMYKCalibrationPipeline(output_dir="calibration_results")
colorspace = pipeline.run_calibration(
    noise_level=0.2,      # Measurement noise (0-1)
    dot_gain=0.15,        # Dot gain simulation (0-1)
    colorspace_name="Custom Profile"
)
```

## Configuration Options

### Noise Level (0.0 - 1.0)
Controls measurement noise simulation:
- `0.0`: Perfect measurements
- `0.1`: High-end spectrophotometer
- `0.2`: Standard measurement device
- `0.5`: Consumer-grade device

### Dot Gain (0.0 - 0.3)
Simulates ink spreading:
- `0.05`: High-quality press
- `0.15`: Standard offset printing
- `0.25`: Digital printing

## Output Files

The calibration pipeline generates several files:

1. **`target_chart.png`**: Visual representation of the target chart
2. **`target_chart_definition.json`**: CMYK values for each patch
3. **`measurements.json`**: Simulated Lab measurements
4. **`{ColorSpace_Name}.pkl`**: Calibrated color space data

## Usage Examples

### Example 1: Basic Calibration

```python
# Simple calibration with default settings
from core.color.calibration import CMYKCalibrationPipeline

pipeline = CMYKCalibrationPipeline()
colorspace = pipeline.run_calibration(colorspace_name="Basic Profile")

# Test conversion
import numpy as np
test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
cmyk_result = colorspace.rgb_to_cmyk_calibrated(test_image)
```

### Example 2: High-Quality Calibration

```python
# High-quality calibration with low noise
pipeline = CMYKCalibrationPipeline(output_dir="high_quality_cal")
colorspace = pipeline.run_calibration(
    noise_level=0.1,      # Low noise
    dot_gain=0.08,        # Minimal dot gain
    colorspace_name="High Quality Press"
)

# Validate results
validation = pipeline.validate_calibration()
print(f"Calibration quality: {validation['accuracy_metrics']['average_delta_e']:.2f} Delta E")
```

### Example 3: Custom Target Chart

```python
# Create custom target chart
from core.color.calibration import CMYKTargetChart, MeasurementSimulator, CMYKColorSpace

chart = CMYKTargetChart(patch_size=30)
patches = chart.generate_complete_target_chart()

# Add custom patches
custom_patches = [(0.2, 0.4, 0.6, 0.1), (0.8, 0.3, 0.1, 0.2)]
all_patches = patches + custom_patches

# Simulate measurements
simulator = MeasurementSimulator()
dataset = {
    'cmyk_inputs': all_patches,
    'lab_measurements': simulator.simulate_measurement(all_patches)
}

# Create custom color space
colorspace = CMYKColorSpace("Custom Profile")
colorspace.calibrate(dataset)
```

## Validation and Quality Assessment

The system provides comprehensive validation tools:

```python
# Run validation
validation_results = pipeline.validate_calibration()

# Access metrics
metrics = validation_results['accuracy_metrics']
print(f"Average Delta E: {metrics['average_delta_e']:.2f}")
print(f"Test colors: {metrics['num_test_colors']}")
print(f"Calibration patches: {metrics['calibration_patches']}")

# Individual color results
for result in validation_results['test_colors']:
    rgb = result['input_rgb']
    delta_e = result['delta_e']
    print(f"RGB{rgb} -> Delta E: {delta_e:.2f}")
```

## Performance Characteristics

- **Target Chart Generation**: ~0.1s for 282 patches
- **Measurement Simulation**: ~0.2s for 282 patches
- **LUT Building**: ~0.1s for 282 patches
- **Color Conversion**: ~0.01s for 100x100 image
- **Memory Usage**: ~44KB per calibrated color space

## Quality Guidelines

### Delta E Interpretation
- **< 1.0**: Excellent match (imperceptible difference)
- **1.0 - 3.0**: Good match (barely perceptible)
- **3.0 - 5.0**: Acceptable match (noticeable under scrutiny)
- **> 5.0**: Poor match (clearly visible difference)

### Recommended Settings
- **High-end printing**: `noise_level=0.1, dot_gain=0.08`
- **Standard printing**: `noise_level=0.2, dot_gain=0.15`
- **Consumer printing**: `noise_level=0.3, dot_gain=0.20`

## Troubleshooting

### Common Issues

1. **High Delta E values**: Increase number of calibration patches or reduce noise level
2. **Interpolation warnings**: Ensure sufficient patch coverage in target gamut
3. **Memory errors**: Reduce image size or patch count for large datasets

### Debugging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration with Existing Code

The calibration system integrates seamlessly with existing CMYK functionality:

```python
# Use with existing optimization
from core.color.cmyk_optimized import optimize_cmyk_until_acceptable

# Apply calibration before optimization
calibrated_cmyk = colorspace.rgb_to_cmyk_calibrated(sample_image)
optimized_result = optimize_cmyk_until_acceptable(
    reference_image,
    calibrated_cmyk  # Use calibrated as starting point
)
```

## Running the Demo

To see the complete system in action:

```bash
python core/color/calibration_demo.py
```

This will generate example calibrations and demonstrate all features.

## Testing

Run the comprehensive test suite:

```bash
python -m pytest tests/test_cmyk_calibration.py -v
```

Tests cover:
- Target chart generation
- Measurement simulation
- LUT building and interpolation
- Color space calibration
- Complete pipeline workflow
- Save/load functionality

## Files and Dependencies

### Core Files
- `calibration.py`: Main calibration system
- `calibration_demo.py`: Demonstration script
- `test_cmyk_calibration.py`: Test suite

### Dependencies
- `difference.py`: Color difference calculations
- `cmyk_optimized.py`: Optimized CMYK functions
- OpenCV, NumPy, SciPy

## Future Enhancements

Potential improvements for real-world usage:
- Support for real spectrophotometer data import
- ICC profile export functionality
- Advanced gamut mapping algorithms
- Multi-media calibration support
- Web-based calibration interface

---

For more information or support, refer to the test files and demo script for practical examples.
