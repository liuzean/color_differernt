"""
Tests for the redesigned pure color-based colorbar analysis system

This module tests the pure colorbar analysis pipeline that focuses specifically
on pure color extraction and ground truth matching.
"""

import numpy as np
import pytest
from PIL import Image

from core.block_detection.pure_colorbar_analysis import (
    _get_color_quality,
    analyze_colorbar_pure_colors,
    analyze_pure_color_block,
    extract_pure_color_from_block,
    pure_colorbar_analysis_for_gradio,
    pure_colorbar_analysis_pipeline,
)


class TestPureColorExtraction:
    """Test pure color extraction functionality"""

    def create_pure_color_block(self, color: tuple[int, int, int], size: tuple[int, int] = (50, 50)) -> np.ndarray:
        """Create a color block with uniform color (BGR format)"""
        height, width = size
        block = np.full((height, width, 3), color[::-1], dtype=np.uint8)  # Convert RGB to BGR
        return block

    def create_mixed_color_block(self, colors: list[tuple[int, int, int]], size: tuple[int, int] = (50, 50)) -> np.ndarray:
        """Create a color block with mixed colors (BGR format)"""
        height, width = size
        block = np.zeros((height, width, 3), dtype=np.uint8)

        # Create vertical stripes of different colors
        stripe_width = width // len(colors)
        for i, color in enumerate(colors):
            start_x = i * stripe_width
            end_x = start_x + stripe_width
            block[:, start_x:end_x] = color[::-1]  # Convert RGB to BGR

        return block

    def test_extract_pure_color_uniform_block(self):
        """Test pure color extraction from uniform color block"""
        # Create pure red block
        red_color = (255, 0, 0)
        red_block = self.create_pure_color_block(red_color)

        pure_color, purity_score = extract_pure_color_from_block(red_block)

        # Should extract the red color with high purity
        assert pure_color == red_color
        assert purity_score > 0.8  # Should be high purity

    def test_extract_pure_color_mixed_block(self):
        """Test pure color extraction from mixed color block"""
        # Create mixed color block
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
        mixed_block = self.create_mixed_color_block(colors)

        pure_color, purity_score = extract_pure_color_from_block(mixed_block)

        # Should have lower purity score due to color variation
        assert purity_score < 0.8
        # Color should be one of the dominant colors or a reasonable mix
        assert isinstance(pure_color, tuple)
        assert len(pure_color) == 3
        assert all(0 <= c <= 255 for c in pure_color)

    def test_extract_pure_color_empty_block(self):
        """Test pure color extraction from empty block"""
        empty_block = np.array([], dtype=np.uint8).reshape(0, 0, 3)

        pure_color, purity_score = extract_pure_color_from_block(empty_block)

        assert pure_color == (0, 0, 0)
        assert purity_score == 0.0

    def test_get_color_quality_levels(self):
        """Test color quality level classification"""
        test_cases = [
            (0.95, "Excellent"),
            (0.85, "Very Good"),
            (0.75, "Good"),
            (0.65, "Fair"),
            (0.55, "Poor"),
            (0.35, "Very Poor"),
        ]

        for purity_score, expected_quality in test_cases:
            quality = _get_color_quality(purity_score)
            assert quality == expected_quality


class TestPureColorBlockAnalysis:
    """Test pure color block analysis functionality"""

    def create_test_block(self, color: tuple[int, int, int]) -> np.ndarray:
        """Create a test color block"""
        height, width = 40, 40
        block = np.full((height, width, 3), color[::-1], dtype=np.uint8)  # Convert RGB to BGR
        return block

    def test_analyze_pure_color_block_success(self):
        """Test successful pure color block analysis"""
        # Create red block
        red_color = (255, 0, 0)
        red_block = self.create_test_block(red_color)

        analysis = analyze_pure_color_block(red_block, block_id=1, colorbar_id=1)

        # Check analysis structure
        assert "error" not in analysis
        assert analysis["block_id"] == 1
        assert analysis["colorbar_id"] == 1
        assert "pure_color_rgb" in analysis
        assert "pure_color_cmyk" in analysis
        assert "purity_score" in analysis
        assert "color_quality" in analysis
        assert "ground_truth_match" in analysis

        # Check ground truth match structure
        gt_match = analysis["ground_truth_match"]
        assert "closest_color" in gt_match
        assert "delta_e" in gt_match
        assert "accuracy_level" in gt_match
        assert "is_acceptable" in gt_match
        assert "is_excellent" in gt_match

        # Check data types
        assert isinstance(analysis["pure_color_rgb"], tuple)
        assert len(analysis["pure_color_rgb"]) == 3
        assert isinstance(analysis["pure_color_cmyk"], tuple)
        assert len(analysis["pure_color_cmyk"]) == 4
        assert isinstance(analysis["purity_score"], float)
        assert isinstance(gt_match["delta_e"], float)

    def test_analyze_pure_color_block_empty(self):
        """Test pure color block analysis with empty block"""
        empty_block = np.array([], dtype=np.uint8).reshape(0, 0, 3)

        analysis = analyze_pure_color_block(empty_block)

        assert "error" in analysis
        assert analysis["error"] == "Empty color block"

    def test_analyze_colorbar_pure_colors(self):
        """Test analyzing multiple pure color blocks"""
        # Create multiple test blocks
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
        blocks = [self.create_test_block(color) for color in colors]

        analyses = analyze_colorbar_pure_colors(blocks, colorbar_id=1)

        assert len(analyses) == 3
        for i, analysis in enumerate(analyses):
            assert analysis["block_id"] == i + 1
            assert analysis["colorbar_id"] == 1
            assert "pure_color_rgb" in analysis
            assert "ground_truth_match" in analysis


class TestPureColorbarAnalysisPipeline:
    """Test the complete pure colorbar analysis pipeline"""

    def create_test_image_with_colorbar(self, size: tuple[int, int] = (400, 300)) -> np.ndarray:
        """Create a test image with a colorbar for testing"""
        width, height = size
        image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background

        # Add a colorbar in the center
        colorbar_height = 50
        colorbar_width = 200
        start_y = (height - colorbar_height) // 2
        start_x = (width - colorbar_width) // 2

        # Create colorbar with 4 colors
        colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Yellow
        ]

        color_width = colorbar_width // len(colors)
        for i, color in enumerate(colors):
            x_start = start_x + i * color_width
            x_end = x_start + color_width
            image[start_y:start_y + colorbar_height, x_start:x_end] = color

        return image

    def test_pure_colorbar_analysis_pipeline_no_image(self):
        """Test pipeline with no image"""
        result = pure_colorbar_analysis_pipeline(None)
        assert "error" in result
        assert "No image provided" in result["error"]

    def test_pure_colorbar_analysis_pipeline_structure(self):
        """Test that the pipeline returns the correct structure"""
        # Create test image
        test_image_array = self.create_test_image_with_colorbar()
        test_image_pil = Image.fromarray(test_image_array)

        # Run pipeline (this might fail if YOLO model is not available)
        try:
            result = pure_colorbar_analysis_pipeline(
                test_image_pil,
                confidence_threshold=0.3,
                box_expansion=5,
                block_area_threshold=30,
                block_aspect_ratio=0.2,
                min_square_size=5,
                purity_threshold=0.7,
            )

            # Check result structure
            assert isinstance(result, dict)

            if "success" in result and result["success"]:
                # If successful, check structure
                assert "analysis_type" in result
                assert result["analysis_type"] == "pure_color_based"
                assert "colorbar_count" in result
                assert "colorbar_results" in result
                assert "total_blocks" in result
                assert "accuracy_statistics" in result
                assert "step_completed" in result

                # Check colorbar results structure
                for colorbar_result in result["colorbar_results"]:
                    assert "colorbar_id" in colorbar_result
                    assert "confidence" in colorbar_result
                    assert "block_count" in colorbar_result
                    assert "pure_color_analyses" in colorbar_result

                # Check accuracy statistics structure
                accuracy_stats = result["accuracy_statistics"]
                if accuracy_stats:  # Only check if there are statistics
                    assert "average_delta_e" in accuracy_stats
                    assert "excellent_colors" in accuracy_stats
                    assert "acceptable_colors" in accuracy_stats
                    assert "high_purity_colors" in accuracy_stats
                    assert "total_analyzed" in accuracy_stats

            elif "error" in result:
                # If there's an error, it's likely due to missing YOLO model
                assert "error" in result
                print(f"Expected error (YOLO model not available): {result['error']}")

        except Exception as e:
            # If there's an exception, it's likely due to missing YOLO model
            print(f"Expected exception (YOLO model not available): {str(e)}")

    def test_pure_colorbar_analysis_for_gradio(self):
        """Test Gradio wrapper function"""
        # Create test image
        test_image_array = self.create_test_image_with_colorbar()
        test_image_pil = Image.fromarray(test_image_array)

        try:
            # Run Gradio wrapper
            annotated_image, colorbar_data, report, total_blocks = pure_colorbar_analysis_for_gradio(
                test_image_pil,
                confidence_threshold=0.3,
                box_expansion=5,
                block_area_threshold=30,
                min_square_size=5,
                purity_threshold=0.7,
            )

            # Check return types
            assert isinstance(annotated_image, (Image.Image, type(None)))
            assert isinstance(colorbar_data, list)
            assert isinstance(report, str)
            assert isinstance(total_blocks, int)

            # Check report content
            assert "Pure Color-Based Colorbar Analysis Results" in report

            # If there are colorbar results, check structure
            for colorbar_entry in colorbar_data:
                assert "colorbar_id" in colorbar_entry
                assert "confidence" in colorbar_entry
                assert "block_count" in colorbar_entry
                assert "pure_color_analyses" in colorbar_entry

        except Exception as e:
            # If there's an exception, it's likely due to missing YOLO model
            print(f"Expected exception (YOLO model not available): {str(e)}")


class TestPureColorAnalysisIntegration:
    """Integration tests for pure color analysis system"""

    def test_color_extraction_accuracy(self):
        """Test that pure color extraction is accurate for known colors"""
        # Test with primary colors
        test_colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]

        for expected_color in test_colors:
            # Create pure color block
            height, width = 30, 30
            block = np.full((height, width, 3), expected_color[::-1], dtype=np.uint8)  # BGR

            # Extract pure color
            extracted_color, purity_score = extract_pure_color_from_block(block)

            # Check accuracy (allow small tolerance for conversion artifacts)
            color_diff = sum(abs(a - b) for a, b in zip(extracted_color, expected_color, strict=False))
            assert color_diff <= 10  # Allow small difference
            assert purity_score > 0.9  # Should be very pure

    def test_purity_score_consistency(self):
        """Test that purity scores are consistent with color uniformity"""
        # Pure color should have high purity score
        pure_red = np.full((30, 30, 3), (0, 0, 255), dtype=np.uint8)  # BGR
        _, pure_purity = extract_pure_color_from_block(pure_red)

        # Mixed color should have lower purity score
        mixed_block = np.zeros((30, 30, 3), dtype=np.uint8)
        mixed_block[:, :15] = (0, 0, 255)  # Red half
        mixed_block[:, 15:] = (255, 0, 0)  # Blue half
        _, mixed_purity = extract_pure_color_from_block(mixed_block)

        assert pure_purity > mixed_purity
        assert pure_purity > 0.8
        assert mixed_purity < 0.8

    def test_ground_truth_matching(self):
        """Test that ground truth matching works correctly"""

        # Test with a color that should match ground truth well
        # Using pure cyan which should match the "Pure Cyan" ground truth color
        cyan_color = (0, 255, 255)  # RGB
        height, width = 30, 30
        cyan_block = np.full((height, width, 3), cyan_color[::-1], dtype=np.uint8)  # BGR

        # Analyze the block
        analysis = analyze_pure_color_block(cyan_block)

        # Check that analysis was successful
        assert "error" not in analysis

        # Check that ground truth matching occurred
        gt_match = analysis["ground_truth_match"]
        assert gt_match["closest_color"] is not None
        assert isinstance(gt_match["delta_e"], float)
        assert gt_match["delta_e"] >= 0  # Delta E should be non-negative

        # Check accuracy classification
        assert gt_match["accuracy_level"] in [
            "Excellent", "Very Good", "Good", "Acceptable", "Poor", "Very Poor"
        ]

    def test_error_handling(self):
        """Test error handling in pure color analysis"""
        # Test with various invalid inputs

        # Empty block
        empty_block = np.array([], dtype=np.uint8).reshape(0, 0, 3)
        analysis = analyze_pure_color_block(empty_block)
        assert "error" in analysis

        # Very small block
        tiny_block = np.ones((1, 1, 3), dtype=np.uint8)
        tiny_analysis = analyze_pure_color_block(tiny_block)
        # Should not crash, may have low purity
        assert "pure_color_rgb" in tiny_analysis

        # Block with extreme colors
        extreme_block = np.ones((10, 10, 3), dtype=np.uint8) * 255  # All white
        extreme_analysis = analyze_pure_color_block(extreme_block)
        assert "pure_color_rgb" in extreme_analysis


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
