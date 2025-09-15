"""
Advanced Color Analysis for Individual Color Blocks
"""

import colorsys

import cv2
import numpy as np
from sklearn.cluster import KMeans

# Import CMYK conversion from existing color system
try:
    from ..color.icc_trans import srgb_to_cmyk_array
except ImportError:
    srgb_to_cmyk_array = None


def rgb_to_cmyk_icc(rgb_tuple: tuple[int, int, int]) -> tuple[int, int, int, int]:
    """
    Convert RGB color to CMYK using ICC profiles.

    Args:
        rgb_tuple: RGB color tuple (0-255 range)

    Returns:
        CMYK tuple (0-100 range) or (0, 0, 0, 0) if conversion fails
    """
    if srgb_to_cmyk_array is None:
        # Fallback calculation if ICC conversion not available
        return rgb_to_cmyk_simple(rgb_tuple)

    try:
        # Create a small RGB image for conversion
        r, g, b = rgb_tuple
        rgb_array = np.array([[[b, g, r]]], dtype=np.uint8)  # BGR format for OpenCV

        # Convert using ICC profiles
        cmyk_array, _ = srgb_to_cmyk_array(rgb_array)

        # Extract CMYK values (0-255 range from ICC) and convert to percentages
        if cmyk_array.size >= 4:
            c, m, y, k = cmyk_array[0, 0, :]
            # Convert from 0-255 to 0-100 percentage
            return (
                int(c * 100 / 255),
                int(m * 100 / 255),
                int(y * 100 / 255),
                int(k * 100 / 255),
            )
        else:
            return rgb_to_cmyk_simple(rgb_tuple)

    except Exception:
        # Fallback to simple calculation
        return rgb_to_cmyk_simple(rgb_tuple)


def rgb_to_cmyk_simple(rgb_tuple: tuple[int, int, int]) -> tuple[int, int, int, int]:
    """
    Simple RGB to CMYK conversion (fallback method).

    Args:
        rgb_tuple: RGB color tuple (0-255 range)

    Returns:
        CMYK tuple (0-100 range)
    """
    r, g, b = [x / 255.0 for x in rgb_tuple]

    # Find the maximum of RGB
    k = 1 - max(r, g, b)

    if k == 1:
        return (0, 0, 0, 100)

    c = (1 - r - k) / (1 - k)
    m = (1 - g - k) / (1 - k)
    y = (1 - b - k) / (1 - k)

    return (int(c * 100), int(m * 100), int(y * 100), int(k * 100))


def shrink_image(
    image: np.ndarray, target_size: tuple[int, int] = (50, 50)
) -> np.ndarray:
    """
    Shrink image to target size for color analysis.

    Args:
        image: Input image array (BGR or RGB)
        target_size: Target (width, height) for resizing

    Returns:
        Resized image array
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def get_dominant_colors(
    image: np.ndarray, k: int = 5, image_processing_size: tuple[int, int] = (25, 25)
) -> list[tuple[tuple[int, int, int], float]]:
    """
    Extract dominant colors from an image using K-means clustering.

    Args:
        image: Input image array (BGR format)
        k: Number of dominant colors to extract
        image_processing_size: Size to resize image for faster processing

    Returns:
        List of (RGB_color, percentage) tuples, sorted by dominance
    """
    # Resize image for faster processing
    image_resized = cv2.resize(
        image, image_processing_size, interpolation=cv2.INTER_AREA
    )

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

    # Reshape image to be a list of pixels
    pixels = image_rgb.reshape((-1, 3))

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)

    # Get cluster centers (dominant colors) and labels
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Count the frequency of each cluster
    (unique, counts) = np.unique(labels, return_counts=True)

    # Calculate percentages
    total_pixels = len(pixels)
    color_percentages = []

    for i, count in enumerate(counts):
        percentage = count / total_pixels
        rgb_color = tuple(colors[unique[i]].astype(int))
        color_percentages.append((rgb_color, percentage))

    # Sort by percentage (most dominant first)
    color_percentages.sort(key=lambda x: x[1], reverse=True)

    return color_percentages


def analyze_single_block_color(
    block_image: np.ndarray,
    shrink_size: tuple[int, int] = (30, 30),
    colorbar_id: int = None,
    block_id: int = None,
) -> dict:
    """
    Comprehensive color analysis of a single color block.

    Args:
        block_image: Individual color block image (BGR format)
        shrink_size: Size to shrink image for analysis
        colorbar_id: ID of parent colorbar
        block_id: ID of this block within the colorbar

    Returns:
        Dictionary with comprehensive color analysis
    """
    if block_image.size == 0:
        return {"error": "Empty block image"}

    # Shrink image for analysis
    shrunken = shrink_image(block_image, shrink_size)

    # Convert to RGB for analysis
    rgb_image = cv2.cvtColor(shrunken, cv2.COLOR_BGR2RGB)

    # Get dominant colors
    dominant_colors = get_dominant_colors(
        block_image, k=3, image_processing_size=shrink_size
    )

    # Calculate average color
    avg_color_bgr = np.mean(shrunken.reshape(-1, 3), axis=0)
    avg_color_rgb = avg_color_bgr[::-1]  # Convert BGR to RGB

    # Convert to other color spaces
    avg_color_hsv = colorsys.rgb_to_hsv(
        avg_color_rgb[0] / 255, avg_color_rgb[1] / 255, avg_color_rgb[2] / 255
    )
    avg_color_hsv = (
        int(avg_color_hsv[0] * 360),
        int(avg_color_hsv[1] * 100),
        int(avg_color_hsv[2] * 100),
    )

    # Calculate color variance (how uniform the color is)
    color_variance = np.var(rgb_image.reshape(-1, 3), axis=0)
    color_std = np.std(rgb_image.reshape(-1, 3), axis=0)

    # Determine if it's a solid color or gradient
    is_solid = np.all(color_std < 20)  # Low standard deviation means solid color

    # Get the most dominant color
    primary_color = dominant_colors[0][0] if dominant_colors else (0, 0, 0)
    primary_percentage = dominant_colors[0][1] if dominant_colors else 0.0

    # Convert primary color to CMYK
    primary_color_cmyk = rgb_to_cmyk_icc(primary_color)

    # Convert average color to CMYK
    avg_color_rgb_int = tuple(avg_color_rgb.astype(int))
    avg_color_cmyk = rgb_to_cmyk_icc(avg_color_rgb_int)

    return {
        "colorbar_id": colorbar_id,
        "block_id": block_id,
        "primary_color_rgb": primary_color,
        "primary_color_cmyk": primary_color_cmyk,
        "primary_color_percentage": primary_percentage,
        "average_color_rgb": avg_color_rgb_int,
        "average_color_cmyk": avg_color_cmyk,
        "average_color_hsv": avg_color_hsv,
        "dominant_colors": dominant_colors[:3],  # Top 3 colors
        "color_variance": color_variance.tolist(),
        "color_std": color_std.tolist(),
        "is_solid_color": is_solid,
        "block_size": block_image.shape[:2],
        "shrunken_size": shrunken.shape[:2],
        "total_pixels": block_image.shape[0] * block_image.shape[1],
    }


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex color string."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def format_color_analysis_report(analysis: dict, block_identifier: str = None) -> str:
    """
    Format color analysis results into a readable report.

    Args:
        analysis: Color analysis dictionary
        block_identifier: Block identifier string

    Returns:
        Formatted report string
    """
    if "error" in analysis:
        return f"Block {block_identifier}: {analysis['error']}"

    if block_identifier is None:
        colorbar_id = analysis.get("colorbar_id", "?")
        block_id = analysis.get("block_id", "?")
        block_identifier = f"{colorbar_id}.{block_id}"

    report = f"🎨 Block {block_identifier} Color Analysis:\n"

    # Primary color with RGB, CMYK, and hex
    primary_rgb = analysis["primary_color_rgb"]
    primary_cmyk = analysis["primary_color_cmyk"]
    primary_hex = rgb_to_hex(primary_rgb)
    report += f"  • Primary Color: RGB{primary_rgb} ({primary_hex}) - {analysis['primary_color_percentage']:.1%}\n"
    report += f"    └─ CMYK: C={primary_cmyk[0]}% M={primary_cmyk[1]}% Y={primary_cmyk[2]}% K={primary_cmyk[3]}%\n"

    # Average color with RGB, CMYK, HSV, and hex
    avg_rgb = analysis["average_color_rgb"]
    avg_cmyk = analysis["average_color_cmyk"]
    avg_hex = rgb_to_hex(avg_rgb)
    avg_hsv = analysis["average_color_hsv"]
    report += f"  • Average Color: RGB{avg_rgb} ({avg_hex})\n"
    report += f"    └─ CMYK: C={avg_cmyk[0]}% M={avg_cmyk[1]}% Y={avg_cmyk[2]}% K={avg_cmyk[3]}%\n"
    report += f"    └─ HSV: H={avg_hsv[0]}° S={avg_hsv[1]}% V={avg_hsv[2]}%\n"

    # Color uniformity
    if analysis["is_solid_color"]:
        report += "  • Type: Solid color (uniform)\n"
    else:
        report += "  • Type: Gradient/mixed colors\n"

    # Dominant colors with CMYK
    report += "  • Dominant Colors:\n"
    for i, (color, percentage) in enumerate(analysis["dominant_colors"][:2]):
        hex_color = rgb_to_hex(color)
        cmyk_color = rgb_to_cmyk_icc(color)
        report += f"    {i + 1}. RGB{color} ({hex_color}) - {percentage:.1%}\n"
        report += f"       └─ CMYK: C={cmyk_color[0]}% M={cmyk_color[1]}% Y={cmyk_color[2]}% K={cmyk_color[3]}%\n"

    # Size info
    report += f"  • Original Size: {analysis['block_size']}\n"
    report += f"  • Analysis Size: {analysis['shrunken_size']}\n"

    return report


def analyze_colorbar_blocks(
    colorbar_blocks: list[np.ndarray],
    shrink_size: tuple[int, int] = (30, 30),
    colorbar_id: int = None,
) -> list[dict]:
    """
    Analyze colors in multiple blocks from a colorbar.

    Args:
        colorbar_blocks: List of individual color block images (BGR format)
        shrink_size: Size to shrink each block for analysis
        colorbar_id: ID of the parent colorbar

    Returns:
        List of color analysis dictionaries
    """
    analyses = []

    for i, block in enumerate(colorbar_blocks):
        analysis = analyze_single_block_color(
            block, shrink_size, colorbar_id=colorbar_id, block_id=i + 1
        )
        analyses.append(analysis)

    return analyses
