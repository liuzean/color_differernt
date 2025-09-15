"""
Simplified block-wise color difference analysis visualization
Minimal styling for use with Gradio interface
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import logging

import cv2
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class SimpleBlockwiseVisualizer:
    """Simplified block-wise analysis visualizer with minimal styling"""

    def __init__(self, dpi: int = 100):
        self.dpi = dpi
        plt.style.use("default")

    def analyze_blocks(
        self, delta_e_map: np.ndarray, block_size: tuple[int, int] = (32, 32)
    ) -> dict:
        """
        Analyze delta E map in blocks and return statistics

        Args:
            delta_e_map: Color difference map
            block_size: Size of each block (height, width)

        Returns:
            Dictionary containing block analysis results
        """
        if delta_e_map is None or delta_e_map.size == 0:
            logger.error("Empty delta E map")
            return {}

        try:
            h, w = delta_e_map.shape[:2]
            block_h, block_w = block_size

            # Calculate number of blocks
            n_blocks_h = h // block_h
            n_blocks_w = w // block_w

            if n_blocks_h == 0 or n_blocks_w == 0:
                logger.warning("Image too small for block analysis")
                return {}

            # Initialize result arrays
            block_means = np.zeros((n_blocks_h, n_blocks_w))
            block_maxes = np.zeros((n_blocks_h, n_blocks_w))
            block_stds = np.zeros((n_blocks_h, n_blocks_w))

            # Analyze each block
            for i in range(n_blocks_h):
                for j in range(n_blocks_w):
                    y1, y2 = i * block_h, (i + 1) * block_h
                    x1, x2 = j * block_w, (j + 1) * block_w

                    block = delta_e_map[y1:y2, x1:x2]

                    block_means[i, j] = np.mean(block)
                    block_maxes[i, j] = np.max(block)
                    block_stds[i, j] = np.std(block)

            return {
                "block_means": block_means,
                "block_maxes": block_maxes,
                "block_stds": block_stds,
                "block_size": block_size,
                "grid_shape": (n_blocks_h, n_blocks_w),
                "total_blocks": n_blocks_h * n_blocks_w,
                "overall_mean": float(np.mean(block_means)),
                "overall_max": float(np.max(block_maxes)),
                "problematic_blocks": int(
                    np.sum(block_means > 5.0)
                ),  # Blocks with mean > 5.0
            }

        except Exception as e:
            logger.error(f"Block analysis failed: {e}")
            return {}

    def create_block_visualization(
        self,
        block_analysis: dict,
        metric: str = "mean",
        title: str = None,
        cmap: str = "viridis",
    ) -> tuple[np.ndarray, dict]:
        """
        Create block-wise visualization

        Args:
            block_analysis: Results from analyze_blocks
            metric: Which metric to visualize ('mean', 'max', 'std')
            title: Optional title
            cmap: Colormap to use

        Returns:
            Tuple of (image_array, visualization_stats)
        """
        if not block_analysis or f"block_{metric}s" not in block_analysis:
            logger.error(f"Invalid block analysis data for metric: {metric}")
            return None, {}

        try:
            data = block_analysis[f"block_{metric}s"]

            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)

            # Create visualization
            im = ax.imshow(data, cmap=cmap, interpolation="nearest")

            # Minimal styling
            ax.set_xticks([])
            ax.set_yticks([])

            if title:
                ax.set_title(title, fontsize=12, pad=10)
            else:
                ax.set_title(f"Block {metric.capitalize()}", fontsize=12, pad=10)

            # Simple colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label(f"ΔE {metric}", rotation=270, labelpad=15)

            plt.tight_layout()

            # Convert to array
            canvas = fig.canvas
            canvas.draw()
            buf = canvas.buffer_rgba()
            image_array = np.asarray(buf)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

            plt.close(fig)

            # Calculate visualization statistics
            stats = {
                "metric": metric,
                "min_value": float(np.min(data)),
                "max_value": float(np.max(data)),
                "mean_value": float(np.mean(data)),
                "std_value": float(np.std(data)),
                "grid_shape": data.shape,
            }

            return image_array, stats

        except Exception as e:
            logger.error(f"Block visualization failed: {e}")
            return None, {}

    def create_comparison_blocks(
        self,
        delta_e_map1: np.ndarray,
        delta_e_map2: np.ndarray,
        block_size: tuple[int, int] = (32, 32),
        titles: tuple[str, str] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Create side-by-side block comparison

        Args:
            delta_e_map1: First delta E map
            delta_e_map2: Second delta E map
            block_size: Size of each block
            titles: Optional titles for each map

        Returns:
            Tuple of (image_array, comparison_stats)
        """
        if delta_e_map1 is None or delta_e_map2 is None:
            logger.error("Invalid delta E maps for comparison")
            return None, {}

        try:
            # Analyze both maps
            analysis1 = self.analyze_blocks(delta_e_map1, block_size)
            analysis2 = self.analyze_blocks(delta_e_map2, block_size)

            if not analysis1 or not analysis2:
                logger.error("Failed to analyze blocks for comparison")
                return None, {}

            # Get block means
            blocks1 = analysis1["block_means"]
            blocks2 = analysis2["block_means"]

            # Calculate combined scale
            vmax = max(np.max(blocks1), np.max(blocks2))

            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)

            # First block visualization
            ax1.imshow(
                blocks1, cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest"
            )
            ax1.set_xticks([])
            ax1.set_yticks([])
            if titles and titles[0]:
                ax1.set_title(titles[0], fontsize=12)
            else:
                ax1.set_title("Blocks 1", fontsize=12)

            # Second block visualization
            im2 = ax2.imshow(
                blocks2, cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest"
            )
            ax2.set_xticks([])
            ax2.set_yticks([])
            if titles and titles[1]:
                ax2.set_title(titles[1], fontsize=12)
            else:
                ax2.set_title("Blocks 2", fontsize=12)

            # Shared colorbar
            fig.colorbar(im2, ax=[ax1, ax2], shrink=0.8, label="Mean ΔE")

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
                "map1_mean": analysis1["overall_mean"],
                "map2_mean": analysis2["overall_mean"],
                "map1_max": analysis1["overall_max"],
                "map2_max": analysis2["overall_max"],
                "map1_problematic": analysis1["problematic_blocks"],
                "map2_problematic": analysis2["problematic_blocks"],
                "difference": analysis1["overall_mean"] - analysis2["overall_mean"],
                "block_size": block_size,
            }

            return image_array, stats

        except Exception as e:
            logger.error(f"Block comparison failed: {e}")
            return None, {}


# Backward compatibility functions
def analyze_blockwise(
    delta_e_map: np.ndarray, block_size: tuple[int, int] = (32, 32)
) -> dict:
    """
    Analyze color differences in blocks

    Args:
        delta_e_map: Color difference map
        block_size: Size of each analysis block

    Returns:
        Dictionary containing block analysis results
    """
    visualizer = SimpleBlockwiseVisualizer()
    return visualizer.analyze_blocks(delta_e_map, block_size)


def create_block_heatmap(
    delta_e_map: np.ndarray,
    block_size: tuple[int, int] = (32, 32),
    metric: str = "mean",
    title: str = "Block Analysis",
    colormap: str = "viridis",
) -> tuple[np.ndarray, dict]:
    """
    Create block-wise heatmap visualization

    Args:
        delta_e_map: Color difference map
        block_size: Size of analysis blocks
        metric: Metric to visualize ('mean', 'max', 'std')
        title: Title for the visualization
        colormap: Matplotlib colormap name

    Returns:
        Tuple of (image_array, visualization_stats)
    """
    visualizer = SimpleBlockwiseVisualizer()
    analysis = visualizer.analyze_blocks(delta_e_map, block_size)
    if not analysis:
        return None, {}
    return visualizer.create_block_visualization(analysis, metric, title, colormap)


def compare_block_analysis(
    delta_e_map1: np.ndarray,
    delta_e_map2: np.ndarray,
    block_size: tuple[int, int] = (32, 32),
    title1: str = "Analysis 1",
    title2: str = "Analysis 2",
) -> tuple[np.ndarray, dict]:
    """
    Compare two delta E maps using block analysis

    Args:
        delta_e_map1: First delta E map
        delta_e_map2: Second delta E map
        block_size: Size of analysis blocks
        title1: Title for first analysis
        title2: Title for second analysis

    Returns:
        Tuple of (comparison_image, comparison_stats)
    """
    visualizer = SimpleBlockwiseVisualizer()
    return visualizer.create_comparison_blocks(
        delta_e_map1, delta_e_map2, block_size, (title1, title2)
    )


# Legacy API compatibility functions
def analyze_blocks(
    delta_e_map: np.ndarray,
    block_size: tuple[int, int] = (32, 32),
    mask: np.ndarray | None = None,
) -> dict:
    """
    Legacy function for block analysis with different return format

    Args:
        delta_e_map: Color difference map
        block_size: Block size (height, width)
        mask: Optional mask (ignored in simplified version)

    Returns:
        Dictionary with block coordinates as keys
    """
    if delta_e_map is None:
        logger.error("Invalid delta E map")
        return {}

    h, w = delta_e_map.shape[:2]
    block_h, block_w = block_size

    rows = h // block_h
    cols = w // block_w

    if rows == 0 or cols == 0:
        return {}

    results = {}

    for r in range(rows):
        for c in range(cols):
            start_y = r * block_h
            start_x = c * block_w
            end_y = min(start_y + block_h, h)
            end_x = min(start_x + block_w, w)

            block = delta_e_map[start_y:end_y, start_x:end_x]

            # Apply mask if provided
            if mask is not None:
                block_mask = mask[start_y:end_y, start_x:end_x]
                if np.sum(block_mask) == 0:
                    continue
                values = block[block_mask > 0]
            else:
                values = block.flatten()

            if len(values) == 0:
                continue

            results[(r, c)] = {
                "mean": float(np.mean(values)),
                "max": float(np.max(values)),
                "min": float(np.min(values)),
                "std": float(np.std(values)),
                "x_range": (start_x, end_x),
                "y_range": (start_y, end_y),
                "pixel_count": len(values),
            }

    return results


def visualize_block_heatmap(
    delta_e_map: np.ndarray,
    block_size: tuple[int, int] = (32, 32),
    mask: np.ndarray | None = None,
    title: str = "Block Analysis",
) -> np.ndarray:
    """
    Legacy function for creating block heatmap

    Args:
        delta_e_map: Color difference map
        block_size: Block size
        mask: Optional mask (ignored)
        title: Chart title

    Returns:
        BGR image array
    """
    image_array, _ = create_block_heatmap(delta_e_map, block_size, "mean", title)
    if image_array is not None:
        return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    return None


def overlay_block_boundaries(
    image: np.ndarray,
    blocks_info: dict,
    threshold: float = None,
    show_values: bool = False,
    color_scheme: str = "red",
) -> np.ndarray:
    """
    Legacy function for overlaying block boundaries (simplified)

    Args:
        image: Base image
        blocks_info: Block information dictionary
        threshold: Optional threshold (ignored)
        show_values: Whether to show values (ignored)
        color_scheme: Color scheme (ignored)

    Returns:
        Image with simple block grid overlay
    """
    if image is None or not blocks_info:
        return image

    result = image.copy()

    # Ensure BGR format
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    # Draw simple grid based on block info
    color = (0, 0, 255)  # Red

    for (_r, _c), block_data in blocks_info.items():
        if "x_range" in block_data and "y_range" in block_data:
            x_start, x_end = block_data["x_range"]
            y_start, y_end = block_data["y_range"]

            # Draw rectangle
            cv2.rectangle(result, (x_start, y_start), (x_end - 1, y_end - 1), color, 1)

    return result
