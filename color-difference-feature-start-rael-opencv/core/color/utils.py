#!/usr/bin/env python

"""
Color Utilities Module

Provides general color space conversion functions and color difference calculations.
This module centralizes all color operations to avoid code duplication.
Uses ICC-based color management for accurate CMYK/RGB conversions.
"""

from enum import Enum

import colour
import cv2
import numpy as np

from .icc_trans import cmyk_to_srgb_array, srgb_to_cmyk_array


class ColorSpace(Enum):
    """Supported color spaces for conversion"""

    RGB = "rgb"
    LAB = "lab"
    CMYK = "cmyk"


def convert_color(
    image: np.ndarray,
    from_space: ColorSpace | str,
    to_space: ColorSpace | str,
    **kwargs,
) -> np.ndarray:
    """
    General color space conversion function

    Args:
        image: Input image array
        from_space: Source color space
        to_space: Target color space
        **kwargs: Additional conversion parameters

    Returns:
        Converted image array
    """
    # Normalize input spaces
    if isinstance(from_space, str):
        from_space = ColorSpace(from_space.lower())
    if isinstance(to_space, str):
        to_space = ColorSpace(to_space.lower())

    # If same color space, return copy
    if from_space == to_space:
        return image.copy()

    # Define conversion mapping
    conversion_map = {
        (ColorSpace.RGB, ColorSpace.LAB): rgb_to_lab,
        (ColorSpace.RGB, ColorSpace.CMYK): rgb_to_cmyk,
        (ColorSpace.CMYK, ColorSpace.RGB): cmyk_to_rgb,
    }

    conversion_key = (from_space, to_space)

    if conversion_key in conversion_map:
        return conversion_map[conversion_key](image, **kwargs)
    else:
        # Multi-step conversion through RGB if needed
        if from_space != ColorSpace.RGB:
            intermediate = convert_color(image, from_space, ColorSpace.RGB, **kwargs)
            return convert_color(intermediate, ColorSpace.RGB, to_space, **kwargs)
        else:
            raise ValueError(
                f"Conversion from {from_space.value} to {to_space.value} not supported"
            )


def rgb_to_lab(rgb_image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to LAB color space using colour library

    Args:
        rgb_image: RGB image array, values in [0, 255] or [0, 1]

    Returns:
        LAB image array
    """
    # Ensure correct input format
    if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Input must be a 3-channel RGB image")

    # Normalize to [0, 1] if needed
    if rgb_image.dtype == np.uint8 or rgb_image.max() > 1.0:
        rgb_normalized = rgb_image.astype(np.float64) / 255.0
    else:
        rgb_normalized = rgb_image.astype(np.float64)

    # Convert to XYZ first, then to LAB using colour library
    xyz = colour.sRGB_to_XYZ(rgb_normalized)
    lab = colour.XYZ_to_Lab(xyz)

    return lab


def rgb_to_cmyk(rgb_image: np.ndarray, **kwargs) -> np.ndarray:
    """
    Convert RGB image to CMYK color space using ICC profiles

    Args:
        rgb_image: RGB image array, values in [0, 255]
        **kwargs: Additional conversion parameters (passed to ICC transformation)

    Returns:
        CMYK image array, values in [0, 1]
    """
    # Ensure correct input format
    if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Input must be a 3-channel RGB image")

    # Convert RGB to BGR for ICC transformation (as expected by icc_trans.py)
    bgr_image = rgb_image[..., ::-1]

    try:
        # Use ICC-based conversion
        cmyk_array, _ = srgb_to_cmyk_array(bgr_image, **kwargs)

        # Normalize to [0, 1] range
        if cmyk_array.dtype == np.uint8:
            cmyk_normalized = cmyk_array.astype(np.float64) / 255.0
        else:
            cmyk_normalized = cmyk_array.astype(np.float64)
            if cmyk_normalized.max() > 1.0:
                cmyk_normalized = cmyk_normalized / 255.0

        return cmyk_normalized
    except Exception as e:
        raise ValueError(f"ICC-based CMYK conversion failed: {e}")


def cmyk_to_rgb(cmyk_image: np.ndarray, **kwargs) -> np.ndarray:
    """
    Convert CMYK image to RGB color space using ICC profiles

    Args:
        cmyk_image: CMYK image array, values in [0, 1]
        **kwargs: Additional conversion parameters (passed to ICC transformation)

    Returns:
        RGB image array, values in [0, 255]
    """
    # Ensure correct input format
    if len(cmyk_image.shape) != 3 or cmyk_image.shape[2] != 4:
        raise ValueError("Input must be a 4-channel CMYK image")

    # Convert to uint8 if needed
    if cmyk_image.dtype != np.uint8:
        if cmyk_image.max() <= 1.0:
            cmyk_uint8 = (cmyk_image * 255.0).astype(np.uint8)
        else:
            cmyk_uint8 = np.clip(cmyk_image, 0, 255).astype(np.uint8)
    else:
        cmyk_uint8 = cmyk_image

    try:
        # Use ICC-based conversion
        rgb_array, _ = cmyk_to_srgb_array(cmyk_uint8, **kwargs)

        # Convert BGR to RGB
        rgb_image = rgb_array[..., ::-1]

        # Ensure uint8 format
        if rgb_image.dtype != np.uint8:
            rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

        return rgb_image
    except Exception as e:
        raise ValueError(f"ICC-based CMYK conversion failed: {e}")


def calculate_color_difference(
    image1: np.ndarray,
    image2: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """
    Calculate color difference between two images using CIEDE2000

    Args:
        image1: First image (RGB)
        image2: Second image (RGB)
        mask: Optional mask for analysis region

    Returns:
        Tuple of (average_delta_e, delta_e_map)
    """
    # Ensure images have same shape
    if image1.shape != image2.shape:
        min_h = min(image1.shape[0], image2.shape[0])
        min_w = min(image1.shape[1], image2.shape[1])
        image1 = image1[:min_h, :min_w]
        image2 = image2[:min_h, :min_w]

    # Convert to LAB
    lab1 = rgb_to_lab(image1)
    lab2 = rgb_to_lab(image2)

    # Calculate delta E using colour library's CIEDE2000 implementation
    delta_e_map = colour.delta_E(lab1, lab2, method="CIE 2000")

    # Apply mask if provided
    if mask is not None:
        if mask.shape[:2] != delta_e_map.shape[:2]:
            mask = cv2.resize(mask, (delta_e_map.shape[1], delta_e_map.shape[0]))

        valid_pixels = delta_e_map[mask > 0]
        avg_delta_e = float(np.mean(valid_pixels)) if len(valid_pixels) > 0 else 0.0
    else:
        avg_delta_e = float(np.mean(delta_e_map))

    return avg_delta_e, delta_e_map


def analyze_color_statistics(
    delta_e_map: np.ndarray, mask: np.ndarray | None = None
) -> dict:
    """
    Calculate color difference statistics

    Args:
        delta_e_map: Color difference map
        mask: Optional mask for analysis region

    Returns:
        Dictionary with statistical metrics
    """
    if mask is not None:
        if mask.shape[:2] != delta_e_map.shape[:2]:
            mask = cv2.resize(mask, (delta_e_map.shape[1], delta_e_map.shape[0]))
        valid_pixels = delta_e_map[mask > 0]
    else:
        valid_pixels = delta_e_map.flatten()

    if len(valid_pixels) == 0:
        return {}

    return {
        "mean": float(np.mean(valid_pixels)),
        "median": float(np.median(valid_pixels)),
        "std": float(np.std(valid_pixels)),
        "min": float(np.min(valid_pixels)),
        "max": float(np.max(valid_pixels)),
        "percentile_95": float(np.percentile(valid_pixels, 95)),
        "percentile_99": float(np.percentile(valid_pixels, 99)),
        "pixel_count": len(valid_pixels),
    }


def create_color_difference_visualization(
    delta_e_map: np.ndarray,
    max_delta_e: float = None,
    colormap: int = cv2.COLORMAP_VIRIDIS,
) -> np.ndarray:
    """
    Create visualization of color difference map

    Args:
        delta_e_map: Color difference map
        max_delta_e: Maximum value for normalization
        colormap: OpenCV colormap

    Returns:
        Colored visualization of the difference map
    """
    if max_delta_e is None:
        max_delta_e = np.max(delta_e_map)

    # Normalize to 0-255
    if max_delta_e > 0:
        normalized = np.clip(delta_e_map / max_delta_e * 255.0, 0, 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(delta_e_map, dtype=np.uint8)

    # Apply colormap
    colored_map = cv2.applyColorMap(normalized, colormap)

    return colored_map
