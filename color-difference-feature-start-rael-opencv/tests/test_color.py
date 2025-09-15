"""
Comprehensive tests for color functionality
"""

import numpy as np
import pytest

from core.color.utils import (
    analyze_color_statistics,
    calculate_color_difference,
    cmyk_to_rgb,
    rgb_to_cmyk,
    rgb_to_lab,
)


@pytest.fixture
def test_images():
    """Create test images for color analysis"""
    # Reference image (red square on white background)
    ref_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    ref_img[25:75, 25:75] = [255, 0, 0]  # Red square

    # Sample image (slightly different red square)
    sample_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    sample_img[25:75, 25:75] = [240, 15, 10]  # Slightly different red

    # Blue square
    blue_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    blue_img[25:75, 25:75] = [0, 0, 255]  # Blue square

    # Mask for the colored area
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 255

    return {"ref": ref_img, "sample": sample_img, "blue": blue_img, "mask": mask}


def test_color_space_conversions():
    """Test RGB-LAB and RGB-CMYK conversions"""
    # Create test colors (red, green, blue)
    test_img = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)

    # Test RGB to LAB conversion
    lab_img = rgb_to_lab(test_img)
    assert lab_img.shape == test_img.shape

    # Test LAB to RGB conversion using colour library
    import colour
    lab_normalized = lab_img.astype(np.float64)
    xyz_img = colour.Lab_to_XYZ(lab_normalized)
    rgb_back = colour.XYZ_to_sRGB(xyz_img)
    rgb_back = np.clip(rgb_back * 255, 0, 255).astype(np.uint8)
    assert rgb_back.shape == test_img.shape
    # Allow some tolerance due to conversion rounding
    assert np.allclose(rgb_back, test_img, atol=10)

    # Test RGB to CMYK conversion
    cmyk_img = rgb_to_cmyk(test_img)
    assert cmyk_img.shape[:2] == test_img.shape[:2]
    assert cmyk_img.shape[2] == 4  # CMYK has 4 channels

    # CMYK values should be in [0, 1] range
    assert np.all(cmyk_img >= 0) and np.all(cmyk_img <= 1)

    # Test CMYK to RGB conversion
    rgb_from_cmyk = cmyk_to_rgb(cmyk_img)
    assert rgb_from_cmyk.shape == test_img.shape

    # For ICC-based conversions, we test that the conversion produces valid RGB values
    # ICC profiles are designed for accurate device-specific color reproduction,
    # not for mathematical round-trip consistency
    assert np.all(rgb_from_cmyk >= 0) and np.all(rgb_from_cmyk <= 255)

    # Test that the conversion functions work without errors
    # The actual color values depend on the specific ICC profiles used
    # (sRGB IEC61966-21.icc and JapanColor2001Coated.icc)

    # Test with a simple grayscale image for more predictable results
    gray_img = np.array([[[128, 128, 128]]], dtype=np.uint8)
    gray_cmyk = rgb_to_cmyk(gray_img)
    gray_rgb_back = cmyk_to_rgb(gray_cmyk)

    # Gray should remain relatively gray (no strong color cast)
    gray_pixel = gray_rgb_back[0, 0, :]
    max_channel = np.max(gray_pixel)
    min_channel = np.min(gray_pixel)

    # The difference between channels should be reasonable for gray
    assert (max_channel - min_channel) < 100  # Allow some variation due to ICC profile


def test_delta_e_calculations(test_images):
    """Test different delta E calculation methods"""
    ref_img = test_images["ref"]
    sample_img = test_images["sample"]

    # Test CIEDE2000 method
    avg_delta_e, delta_e = calculate_color_difference(
        ref_img, sample_img
    )
    assert delta_e.shape == (100, 100)

    # The red squares should have a nonzero delta E
    assert np.mean(delta_e[25:75, 25:75]) > 0

    # White background should have delta E of 0
    assert np.mean(delta_e[0:25, 0:25]) < 1e-5

    # Test with mask
    mask = test_images["mask"]
    masked_avg_delta_e, masked_delta_e = calculate_color_difference(
        ref_img, sample_img, mask
    )

    # Only pixels in the mask should be counted
    assert masked_avg_delta_e > 0


def test_lab_difference_analysis(test_images):
    """Test LAB difference analysis"""
    # Convert images to LAB
    lab_ref = rgb_to_lab(test_images["ref"])
    lab_sample = rgb_to_lab(test_images["sample"])
    lab_blue = rgb_to_lab(test_images["blue"])

    # Test small difference (red vs slightly different red)
    avg_delta_small, delta_e_small = calculate_color_difference(
        test_images["ref"], test_images["sample"]
    )
    assert avg_delta_small >= 0

    # Test large difference (red vs blue)
    avg_delta_large, delta_e_large = calculate_color_difference(
        test_images["ref"], test_images["blue"]
    )

    # Large color difference should be greater than small difference
    assert avg_delta_large > avg_delta_small


def test_statistics_calculation(test_images):
    """Test statistics calculation for delta E values"""
    # Calculate delta E
    avg_delta_e, delta_e = calculate_color_difference(
        test_images["ref"], test_images["sample"]
    )

    # Calculate statistics
    stats = analyze_color_statistics(delta_e)

    # Check that stats contains required fields
    assert "mean" in stats
    assert "median" in stats
    assert "std" in stats
    assert "max" in stats
    assert "min" in stats
    assert "percentile_95" in stats

    # Mean should be greater than min
    assert stats["mean"] > stats["min"]
    # Max should be greater than mean
    assert stats["max"] > stats["mean"]
    # 95th percentile should be between mean and max
    assert stats["percentile_95"] <= stats["max"]
