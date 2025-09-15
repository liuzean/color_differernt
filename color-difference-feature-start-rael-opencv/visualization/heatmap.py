"""
Simplified heatmap visualization for color difference analysis
Minimal styling for use with Gradio interface
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import logging

import cv2
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class SimpleHeatmapGenerator:
    """Simplified heatmap generator with minimal styling"""

    def __init__(self, dpi: int = 100):
        self.dpi = dpi
        plt.style.use("default")

    def create_heatmap(
        self,
        delta_e_map: np.ndarray,
        title: str = None,
        cmap: str = "viridis",
        max_delta_e: float = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """
        Create a simplified heatmap of color differences

        Args:
            delta_e_map: Color difference map
            title: Optional title
            cmap: Colormap to use
            max_delta_e: Maximum delta E for scale

        Returns:
            Tuple of (image_array, statistics)
        """
        if delta_e_map is None or delta_e_map.size == 0:
            logger.error("Empty delta E map")
            return None, {}

        try:
            # Calculate statistics
            stats = {
                "mean": float(np.mean(delta_e_map)),
                "max": float(np.max(delta_e_map)),
                "min": float(np.min(delta_e_map)),
                "std": float(np.std(delta_e_map)),
            }

            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)

            # Set scale
            vmax = max_delta_e if max_delta_e is not None else stats["max"]

            # Create heatmap
            im = ax.imshow(delta_e_map, cmap=cmap, vmin=0, vmax=vmax)

            # Minimal styling
            ax.set_xticks([])
            ax.set_yticks([])

            if title:
                ax.set_title(title, fontsize=12, pad=10)

            # Simple colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label("ΔE", rotation=270, labelpad=15)

            plt.tight_layout()

            # Convert to array
            canvas = fig.canvas
            canvas.draw()
            buf = canvas.buffer_rgba()
            image_array = np.asarray(buf)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

            plt.close(fig)

            return image_array, stats

        except Exception as e:
            logger.error(f"Heatmap creation failed: {e}")
            return None, {}

    def create_comparison_heatmap(
        self,
        delta_e_map1: np.ndarray,
        delta_e_map2: np.ndarray,
        titles: tuple[str, str] = None,
        cmap: str = "viridis",
    ) -> tuple[np.ndarray, dict]:
        """
        Create side-by-side comparison heatmaps

        Args:
            delta_e_map1: First delta E map
            delta_e_map2: Second delta E map
            titles: Optional titles for each map
            cmap: Colormap to use

        Returns:
            Tuple of (image_array, comparison_stats)
        """
        if delta_e_map1 is None or delta_e_map2 is None:
            logger.error("Invalid delta E maps for comparison")
            return None, {}

        try:
            # Calculate combined scale
            vmax = max(np.max(delta_e_map1), np.max(delta_e_map2))

            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)

            # First heatmap
            ax1.imshow(delta_e_map1, cmap=cmap, vmin=0, vmax=vmax)
            ax1.set_xticks([])
            ax1.set_yticks([])
            if titles and titles[0]:
                ax1.set_title(titles[0], fontsize=12)

            # Second heatmap
            im2 = ax2.imshow(delta_e_map2, cmap=cmap, vmin=0, vmax=vmax)
            ax2.set_xticks([])
            ax2.set_yticks([])
            if titles and titles[1]:
                ax2.set_title(titles[1], fontsize=12)

            # Shared colorbar
            fig.colorbar(im2, ax=[ax1, ax2], shrink=0.8, label="ΔE")

            plt.tight_layout()

            # Convert to array
            canvas = fig.canvas
            canvas.draw()
            buf = canvas.buffer_rgba()
            image_array = np.asarray(buf)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

            plt.close(fig)

            # Calculate comparison statistics
            stats = {
                "map1_mean": float(np.mean(delta_e_map1)),
                "map2_mean": float(np.mean(delta_e_map2)),
                "map1_max": float(np.max(delta_e_map1)),
                "map2_max": float(np.max(delta_e_map2)),
                "difference": float(np.mean(delta_e_map1) - np.mean(delta_e_map2)),
            }

            return image_array, stats

        except Exception as e:
            logger.error(f"Comparison heatmap creation failed: {e}")
            return None, {}


# Backward compatibility functions
def create_heatmap(
    delta_e_map: np.ndarray,
    title: str = "Color Difference Heatmap",
    max_delta_e: float = None,
    colormap: str = "viridis",
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Create a simple heatmap visualization

    Args:
        delta_e_map: Color difference map
        title: Title for the heatmap
        max_delta_e: Maximum delta E for scale
        colormap: Matplotlib colormap name

    Returns:
        Tuple of (image_array, statistics)
    """
    generator = SimpleHeatmapGenerator()
    return generator.create_heatmap(delta_e_map, title, colormap, max_delta_e)


def create_comparison_heatmap(
    delta_e_map1: np.ndarray,
    delta_e_map2: np.ndarray,
    title1: str = "Map 1",
    title2: str = "Map 2",
    colormap: str = "viridis",
) -> tuple[np.ndarray, dict]:
    """
    Create side-by-side comparison heatmaps

    Args:
        delta_e_map1: First delta E map
        delta_e_map2: Second delta E map
        title1: Title for first map
        title2: Title for second map
        colormap: Matplotlib colormap name

    Returns:
        Tuple of (image_array, comparison_statistics)
    """
    generator = SimpleHeatmapGenerator()
    return generator.create_comparison_heatmap(
        delta_e_map1, delta_e_map2, (title1, title2), colormap
    )


# Legacy API compatibility functions
def generate_heatmap(
    delta_e_map: np.ndarray,
    max_value: float = None,
    colormap: int = cv2.COLORMAP_VIRIDIS,
) -> np.ndarray:
    """
    Legacy function for generating simple heatmaps (OpenCV style)

    Args:
        delta_e_map: Color difference map
        max_value: Maximum value for scaling
        colormap: OpenCV colormap constant

    Returns:
        BGR image array
    """
    if delta_e_map is None:
        logger.error("Invalid delta E map")
        return None

    if max_value is None:
        max_value = np.max(delta_e_map)

    if max_value <= 0:
        max_value = 1.0

    # Normalize to 0-255
    normalized = np.clip(delta_e_map / max_value * 255.0, 0, 255).astype(np.uint8)

    # Apply colormap
    heatmap = cv2.applyColorMap(normalized, colormap)

    return heatmap


def overlay_heatmap(
    image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.6
) -> np.ndarray:
    """
    Overlay heatmap on original image

    Args:
        image: Original image (BGR)
        heatmap: Heatmap image (BGR)
        alpha: Transparency for heatmap

    Returns:
        Overlayed image
    """
    if image is None or heatmap is None:
        logger.error("Invalid input images")
        return None

    # Ensure same dimensions
    if image.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Ensure BGR format
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Blend images
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    return overlay


def create_heatmap_with_colorbar(
    delta_e_map: np.ndarray,
    title: str = "Color Difference Heatmap",
    cmap: str = "viridis",
    vmax: float = None,
) -> np.ndarray:
    """
    Legacy function for creating heatmap with colorbar

    Args:
        delta_e_map: Color difference map
        title: Chart title
        cmap: Colormap name
        vmax: Maximum value

    Returns:
        BGR image array
    """
    image_array, _ = create_heatmap(delta_e_map, title, vmax, cmap)
    if image_array is not None:
        return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    return None


def highlight_regions(
    delta_e_map: np.ndarray, threshold: float, image: np.ndarray | None = None
) -> np.ndarray:
    """
    Highlight regions above threshold

    Args:
        delta_e_map: Color difference map
        threshold: Threshold value
        image: Optional base image

    Returns:
        Image with highlighted regions
    """
    if delta_e_map is None:
        logger.error("Invalid delta E map")
        return None

    # Create mask for regions above threshold
    mask = (delta_e_map > threshold).astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare result image
    if image is not None:
        if len(image.shape) == 2:
            highlighted = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            highlighted = image.copy()
    else:
        highlighted = np.zeros(
            (delta_e_map.shape[0], delta_e_map.shape[1], 3), dtype=np.uint8
        )

    # Draw contours
    cv2.drawContours(highlighted, contours, -1, (0, 0, 255), 2)

    # Add semi-transparent overlay
    overlay = highlighted.copy()
    for cnt in contours:
        cv2.fillPoly(overlay, [cnt], (0, 0, 255))

    highlighted = cv2.addWeighted(highlighted, 0.7, overlay, 0.3, 0)

    return highlighted
