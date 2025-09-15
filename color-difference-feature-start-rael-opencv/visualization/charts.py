"""
Simplified charts visualization for color difference analysis
Minimal styling for use with Gradio interface
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import logging

import cv2
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class SimpleChartsGenerator:
    """Simplified charts generator with minimal styling"""

    def __init__(self, dpi: int = 100):
        self.dpi = dpi
        plt.style.use("default")

    def create_histogram(
        self, delta_e_map: np.ndarray, title: str = None, bins: int = 50
    ) -> tuple[np.ndarray, dict]:
        """
        Create histogram of delta E values

        Args:
            delta_e_map: Color difference map
            title: Optional title
            bins: Number of histogram bins

        Returns:
            Tuple of (image_array, histogram_stats)
        """
        if delta_e_map is None or delta_e_map.size == 0:
            logger.error("Empty delta E map")
            return None, {}

        try:
            # Flatten the data
            values = delta_e_map.flatten()

            # Calculate statistics
            stats = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values),
            }

            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)

            # Create histogram
            n, bins_edges, patches = ax.hist(
                values, bins=bins, alpha=0.7, color="skyblue", edgecolor="black"
            )

            # Simple styling
            if title:
                ax.set_title(title, fontsize=12, pad=10)
            else:
                ax.set_title("Delta E Distribution", fontsize=12, pad=10)

            ax.set_xlabel("ΔE Values", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.grid(True, alpha=0.3)

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
            logger.error(f"Histogram creation failed: {e}")
            return None, {}

    def create_comparison_chart(
        self,
        data1: np.ndarray,
        data2: np.ndarray,
        labels: tuple[str, str] = None,
        title: str = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Create comparison histogram

        Args:
            data1: First dataset
            data2: Second dataset
            labels: Labels for the datasets
            title: Optional title

        Returns:
            Tuple of (image_array, comparison_stats)
        """
        if data1 is None or data2 is None:
            logger.error("Invalid data for comparison")
            return None, {}

        try:
            # Flatten data
            values1 = data1.flatten()
            values2 = data2.flatten()

            # Calculate statistics
            stats = {
                "data1_mean": float(np.mean(values1)),
                "data2_mean": float(np.mean(values2)),
                "data1_std": float(np.std(values1)),
                "data2_std": float(np.std(values2)),
                "difference": float(np.mean(values1) - np.mean(values2)),
            }

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)

            # Create overlapping histograms
            ax.hist(
                values1,
                bins=50,
                alpha=0.7,
                color="blue",
                label=labels[0] if labels else "Data 1",
            )
            ax.hist(
                values2,
                bins=50,
                alpha=0.7,
                color="red",
                label=labels[1] if labels else "Data 2",
            )

            # Simple styling
            if title:
                ax.set_title(title, fontsize=12, pad=10)
            else:
                ax.set_title("Comparison Chart", fontsize=12, pad=10)

            ax.set_xlabel("ΔE Values", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)

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
            logger.error(f"Comparison chart creation failed: {e}")
            return None, {}

    def create_statistics_chart(
        self, statistics_list: list[dict], labels: list[str] = None, title: str = None
    ) -> tuple[np.ndarray, dict]:
        """
        Create bar chart of statistics

        Args:
            statistics_list: List of statistics dictionaries
            labels: Labels for each statistics set
            title: Optional title

        Returns:
            Tuple of (image_array, chart_stats)
        """
        if not statistics_list:
            logger.error("Empty statistics list")
            return None, {}

        try:
            # Extract common metrics
            metrics = ["mean", "max", "std"]
            n_sets = len(statistics_list)

            if labels is None:
                labels = [f"Set {i + 1}" for i in range(n_sets)]

            # Prepare data
            data = {metric: [] for metric in metrics}
            for stats in statistics_list:
                for metric in metrics:
                    data[metric].append(stats.get(metric, 0))

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)

            # Create grouped bar chart
            x = np.arange(len(labels))
            width = 0.25

            colors = ["skyblue", "lightcoral", "lightgreen"]
            for i, metric in enumerate(metrics):
                ax.bar(
                    x + i * width,
                    data[metric],
                    width,
                    label=metric.capitalize(),
                    color=colors[i],
                )

            # Simple styling
            if title:
                ax.set_title(title, fontsize=12, pad=10)
            else:
                ax.set_title("Statistics Comparison", fontsize=12, pad=10)

            ax.set_xlabel("Datasets", fontsize=10)
            ax.set_ylabel("ΔE Values", fontsize=10)
            ax.set_xticks(x + width)
            ax.set_xticklabels(labels)
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

            plt.tight_layout()

            # Convert to array
            canvas = fig.canvas
            canvas.draw()
            buf = canvas.buffer_rgba()
            image_array = np.asarray(buf)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

            plt.close(fig)

            # Calculate chart statistics
            chart_stats = {
                "n_datasets": n_sets,
                "metrics": metrics,
                "max_values": {metric: max(data[metric]) for metric in metrics},
                "min_values": {metric: min(data[metric]) for metric in metrics},
            }

            return image_array, chart_stats

        except Exception as e:
            logger.error(f"Statistics chart creation failed: {e}")
            return None, {}


# Backward compatibility functions
def create_histogram(
    delta_e_map: np.ndarray, title: str = "Delta E Histogram", bins: int = 50
) -> tuple[np.ndarray, dict]:
    """
    Create histogram of delta E values

    Args:
        delta_e_map: Color difference map
        title: Chart title
        bins: Number of histogram bins

    Returns:
        Tuple of (image_array, statistics)
    """
    generator = SimpleChartsGenerator()
    return generator.create_histogram(delta_e_map, title, bins)


def create_comparison_histogram(
    data1: np.ndarray,
    data2: np.ndarray,
    label1: str = "Data 1",
    label2: str = "Data 2",
    title: str = "Comparison",
) -> tuple[np.ndarray, dict]:
    """
    Create comparison histogram of two datasets

    Args:
        data1: First dataset
        data2: Second dataset
        label1: Label for first dataset
        label2: Label for second dataset
        title: Chart title

    Returns:
        Tuple of (image_array, comparison_stats)
    """
    generator = SimpleChartsGenerator()
    return generator.create_comparison_chart(data1, data2, (label1, label2), title)


def create_stats_comparison(
    statistics_list: list[dict],
    labels: list[str] = None,
    title: str = "Statistics Comparison",
) -> tuple[np.ndarray, dict]:
    """
    Create bar chart comparing statistics from multiple datasets

    Args:
        statistics_list: List of statistics dictionaries
        labels: Labels for each dataset
        title: Chart title

    Returns:
        Tuple of (image_array, chart_statistics)
    """
    generator = SimpleChartsGenerator()
    return generator.create_statistics_chart(statistics_list, labels, title)


# Legacy API compatibility functions
def histogram_delta_e(
    delta_e_map: np.ndarray,
    bins: int = 50,
    title: str = "Delta E Histogram",
    max_value: float = None,
    threshold: float = None,
) -> np.ndarray:
    """
    Legacy function for creating histogram

    Args:
        delta_e_map: Color difference map
        bins: Number of bins
        title: Chart title
        max_value: Maximum value (ignored)
        threshold: Threshold value (ignored)

    Returns:
        BGR image array
    """
    image_array, _ = create_histogram(delta_e_map, title, bins)
    if image_array is not None:
        return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    return None


def stats_box_chart(stats: dict[str, float], title: str = "Statistics") -> np.ndarray:
    """
    Legacy function for creating stats chart

    Args:
        stats: Statistics dictionary
        title: Chart title

    Returns:
        BGR image array (simplified version)
    """
    # Create a simple bar chart from stats
    try:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

        # Extract key metrics
        metrics = ["mean", "median", "std", "max"]
        values = [stats.get(metric, 0) for metric in metrics]

        # Create bar chart
        bars = ax.bar(metrics, values, color="lightblue", edgecolor="black")

        # Simple styling
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Values", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, value in zip(bars, values, strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()

        # Convert to array
        canvas = fig.canvas
        canvas.draw()
        buf = canvas.buffer_rgba()
        image_array = np.asarray(buf)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

        plt.close(fig)

        # Convert to BGR for legacy compatibility
        return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    except Exception as e:
        logger.error(f"Stats chart creation failed: {e}")
        return None


def create_composite_analysis(
    delta_e_map: np.ndarray,
    stats: dict[str, float],
    image: np.ndarray | None = None,
    title: str = "Analysis",
) -> np.ndarray:
    """
    Legacy function for creating composite analysis (simplified)

    Args:
        delta_e_map: Color difference map
        stats: Statistics dictionary
        image: Optional base image (ignored)
        title: Chart title

    Returns:
        BGR image array
    """
    # Create simple histogram as composite analysis
    image_array, _ = create_histogram(delta_e_map, title)
    if image_array is not None:
        return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    return None


def create_comparison_chart(
    before_delta_e: np.ndarray,
    after_delta_e: np.ndarray,
    title: str = "Comparison",
    bins: int = 40,
) -> np.ndarray:
    """
    Legacy function for creating comparison chart

    Args:
        before_delta_e: Before data
        after_delta_e: After data
        title: Chart title
        bins: Number of bins (ignored)

    Returns:
        BGR image array
    """
    image_array, _ = create_comparison_histogram(
        before_delta_e, after_delta_e, "Before", "After", title
    )
    if image_array is not None:
        return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    return None
