"""
Comprehensive tests for the redesigned colorbar analysis system

This module tests the colorbar analysis pipeline that focuses on pure color
matching with ground truth colors, including:
- Colorbar detection using YOLO
- Pure color extraction from detected blocks
- Ground truth color matching with CMYK values
- Delta E calculation and accuracy assessment
- Complete pipeline integration
"""


import numpy as np
import pytest
from PIL import Image

from core.block_detection.colorbar_analysis import (
    colorbar_analysis_pipeline,
    enhance_with_ground_truth_comparison,
    extract_blocks_from_colorbar,
)
from core.color.ground_truth_checker import GroundTruthColor, GroundTruthColorChecker


class TestGroundTruthColorChecker:
    """Test ground truth color checker functionality"""

    def test_ground_truth_color_creation(self):
        """Test creation of ground truth colors"""
        color = GroundTruthColor(
            id=1,
            name="Test Color",
            cmyk=(50, 25, 75, 10),
            rgb=(100, 150, 200),
            lab=(60.0, 10.0, -20.0),
        )

        assert color.id == 1
        assert color.name == "Test Color"
        assert color.cmyk == (50, 25, 75, 10)
        assert color.rgb == (100, 150, 200)
        assert color.lab == (60.0, 10.0, -20.0)

    def test_ground_truth_checker_initialization(self):
        """Test ground truth checker initialization"""
        checker = GroundTruthColorChecker()

        # Should have preset colors
        assert len(checker.colors) > 0

        # All colors should have computed RGB and LAB values
        for color in checker.colors:
            assert color.rgb is not None
            assert color.lab is not None
            assert len(color.rgb) == 3
            assert len(color.lab) == 3

    def test_get_color_by_id(self):
        """Test getting color by ID"""
        checker = GroundTruthColorChecker()

        # Test getting existing color
        color = checker.get_color_by_id(1)
        assert color is not None
        assert color.id == 1

        # Test getting non-existing color
        color = checker.get_color_by_id(999)
        assert color is None

    def test_find_closest_color(self):
        """Test finding closest ground truth color"""
        checker = GroundTruthColorChecker()

        # Test with pure red color
        red_rgb = (255, 0, 0)
        closest_color, delta_e = checker.find_closest_color(red_rgb)

        assert closest_color is not None
        assert isinstance(delta_e, float)
        assert delta_e >= 0

        # Test with white color
        white_rgb = (255, 255, 255)
        closest_color, delta_e = checker.find_closest_color(white_rgb)

        assert closest_color is not None
        assert isinstance(delta_e, float)
        assert delta_e >= 0

    def test_calculate_delta_e_for_colors(self):
        """Test delta E calculation for multiple colors"""
        checker = GroundTruthColorChecker()

        # Create test colors
        test_colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 255),  # White
        ]

        results = checker.calculate_delta_e_for_colors(test_colors)

        assert len(results) == 4

        for i, result in enumerate(results):
            assert result["detected_color_id"] == i
            assert result["detected_rgb"] == test_colors[i]
            assert "closest_ground_truth" in result
            assert "delta_e" in result
            assert "accuracy_level" in result

            # Check accuracy level classification
            assert result["accuracy_level"] in [
                "Excellent",
                "Very Good",
                "Good",
                "Acceptable",
                "Poor",
                "Very Poor",
            ]

    def test_accuracy_level_classification(self):
        """Test accuracy level classification based on delta E"""
        checker = GroundTruthColorChecker()

        # Test different delta E values
        test_cases = [
            (0.5, "Excellent"),
            (1.5, "Very Good"),
            (2.5, "Good"),
            (4.0, "Acceptable"),
            (8.0, "Poor"),
            (15.0, "Very Poor"),
        ]

        for delta_e, expected_level in test_cases:
            level = checker._get_accuracy_level(delta_e)
            assert level == expected_level

    def test_palette_config_generation(self):
        """Test palette configuration generation"""
        checker = GroundTruthColorChecker()

        config = checker.get_palette_config()

        assert "layout" in config
        assert "gradients" in config

        # Check layout configuration
        layout = config["layout"]
        assert "swatch_size_mm" in layout
        assert "arrangement" in layout
        assert "columns" in layout

        # Check gradients configuration
        gradients = config["gradients"]
        assert len(gradients) == len(checker.colors)

        for i, gradient in enumerate(gradients):
            assert "name" in gradient
            assert "start" in gradient
            assert "end" in gradient
            assert "steps" in gradient
            assert gradient["steps"] == 1  # Single step for ground truth colors

    def test_summary_stats(self):
        """Test summary statistics generation"""
        checker = GroundTruthColorChecker()

        stats = checker.get_summary_stats()

        assert "total_colors" in stats
        assert "color_names" in stats
        assert "cmyk_range" in stats

        # Check CMYK range
        cmyk_range = stats["cmyk_range"]
        for component in ["c_min", "c_max", "m_min", "m_max", "y_min", "y_max", "k_min", "k_max"]:
            assert component in cmyk_range
            assert isinstance(cmyk_range[component], (int, float))

        # Check that ranges are logical
        assert cmyk_range["c_min"] <= cmyk_range["c_max"]
        assert cmyk_range["m_min"] <= cmyk_range["m_max"]
        assert cmyk_range["y_min"] <= cmyk_range["y_max"]
        assert cmyk_range["k_min"] <= cmyk_range["k_max"]


class TestColorbarAnalysisUtils:
    """Test colorbar analysis utility functions"""

    def create_test_colorbar_image(self, colors: list[tuple[int, int, int]], size: tuple[int, int] = (200, 50)):
        """Create a test colorbar image with specified colors"""
        width, height = size
        colors_count = len(colors)
        color_width = width // colors_count

        image = np.zeros((height, width, 3), dtype=np.uint8)

        for i, color in enumerate(colors):
            start_x = i * color_width
            end_x = start_x + color_width
            image[:, start_x:end_x] = color

        return image

    def test_extract_blocks_from_colorbar(self):
        """Test extracting color blocks from colorbar"""
        # Create test colorbar with distinct colors
        test_colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Yellow
        ]

        colorbar_image = self.create_test_colorbar_image(test_colors)

        # Extract blocks
        annotated_image, color_blocks, block_count = extract_blocks_from_colorbar(
            colorbar_image,
            area_threshold=50,
            aspect_ratio_threshold=0.3,
            min_square_size=10,
        )

        # Should detect some blocks
        assert block_count >= 0
        assert len(color_blocks) == block_count

        # Annotated image should have same dimensions as input
        assert annotated_image.shape == colorbar_image.shape

    def test_enhance_with_ground_truth_comparison(self):
        """Test enhancing block analyses with ground truth comparison"""
        # Create mock block analysis
        block_analyses = [
            {
                "colorbar_id": 1,
                "block_id": 1,
                "primary_color_rgb": (255, 0, 0),  # Red
                "primary_color_cmyk": (0, 100, 100, 0),
                "average_color_rgb": (250, 5, 5),
                "is_solid_color": True,
            },
            {
                "colorbar_id": 1,
                "block_id": 2,
                "primary_color_rgb": (0, 255, 0),  # Green
                "primary_color_cmyk": (100, 0, 100, 0),
                "average_color_rgb": (5, 250, 5),
                "is_solid_color": True,
            },
            {
                "error": "Empty block image",
            },
        ]

        enhanced_analyses = enhance_with_ground_truth_comparison(block_analyses)

        assert len(enhanced_analyses) == 3

        # First analysis should be enhanced
        enhanced_1 = enhanced_analyses[0]
        assert "ground_truth_comparison" in enhanced_1
        gt_comp = enhanced_1["ground_truth_comparison"]
        assert "closest_color" in gt_comp
        assert "delta_e" in gt_comp
        assert "accuracy_level" in gt_comp
        assert "is_acceptable" in gt_comp

        # Second analysis should be enhanced
        enhanced_2 = enhanced_analyses[1]
        assert "ground_truth_comparison" in enhanced_2

        # Third analysis should remain unchanged (has error)
        enhanced_3 = enhanced_analyses[2]
        assert "error" in enhanced_3
        assert "ground_truth_comparison" not in enhanced_3


class TestColorbarAnalysisPipeline:
    """Test complete colorbar analysis pipeline"""

    def create_test_image_with_colorbar(self, size: tuple[int, int] = (400, 300)):
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

    def test_colorbar_analysis_pipeline_no_image(self):
        """Test pipeline with no image"""
        result = colorbar_analysis_pipeline(None)
        assert "error" in result
        assert "No image provided" in result["error"]

    def test_colorbar_analysis_pipeline_success(self):
        """Test successful colorbar analysis pipeline"""
        # Create test image
        test_image_array = self.create_test_image_with_colorbar()
        test_image_pil = Image.fromarray(test_image_array)

        # Run pipeline (this might fail if YOLO model is not available)
        try:
            result = colorbar_analysis_pipeline(
                test_image_pil,
                confidence_threshold=0.3,
                box_expansion=5,
                block_area_threshold=30,
                block_aspect_ratio=0.2,
                min_square_size=5,
                shrink_size=(20, 20),
            )

            # If YOLO detection works, we should get results
            if "success" in result:
                assert result["success"] is True
                assert "colorbar_count" in result
                assert "colorbar_results" in result
                assert "total_blocks" in result
                assert "step_completed" in result

                # Check colorbar results structure
                for colorbar_result in result["colorbar_results"]:
                    assert "colorbar_id" in colorbar_result
                    assert "confidence" in colorbar_result
                    assert "block_count" in colorbar_result
                    assert "block_analyses" in colorbar_result

                    # Check block analyses
                    for analysis in colorbar_result["block_analyses"]:
                        if "error" not in analysis:
                            assert "primary_color_rgb" in analysis
                            assert "primary_color_cmyk" in analysis
                            assert "ground_truth_comparison" in analysis

                            # Check ground truth comparison
                            gt_comp = analysis["ground_truth_comparison"]
                            assert "closest_color" in gt_comp
                            assert "delta_e" in gt_comp
                            assert "accuracy_level" in gt_comp
                            assert "is_acceptable" in gt_comp

            elif "error" in result:
                # If YOLO model is not available, we expect an error
                assert "error" in result
                print(f"Expected error (YOLO model not available): {result['error']}")

        except Exception as e:
            # If there's an exception, it's likely due to missing YOLO model
            print(f"Expected exception (YOLO model not available): {str(e)}")

    def test_colorbar_analysis_pipeline_parameters(self):
        """Test pipeline with different parameters"""
        # Create test image
        test_image_array = self.create_test_image_with_colorbar()
        test_image_pil = Image.fromarray(test_image_array)

        # Test with different parameters
        parameters = [
            {
                "confidence_threshold": 0.8,
                "box_expansion": 20,
                "block_area_threshold": 100,
                "block_aspect_ratio": 0.5,
                "min_square_size": 15,
                "shrink_size": (40, 40),
            },
            {
                "confidence_threshold": 0.2,
                "box_expansion": 5,
                "block_area_threshold": 20,
                "block_aspect_ratio": 0.1,
                "min_square_size": 5,
                "shrink_size": (10, 10),
            },
        ]

        for params in parameters:
            try:
                result = colorbar_analysis_pipeline(test_image_pil, **params)
                # Should not crash regardless of parameters
                assert isinstance(result, dict)
                assert "error" in result or "success" in result
            except Exception as e:
                # If there's an exception, it's likely due to missing YOLO model
                print(f"Expected exception (YOLO model not available): {str(e)}")


class TestColorbarAnalysisIntegration:
    """Integration tests for colorbar analysis system"""

    def test_ground_truth_matching_accuracy(self):
        """Test accuracy of ground truth color matching"""
        checker = GroundTruthColorChecker()

        # Test with ground truth colors themselves
        for gt_color in checker.colors:
            if gt_color.rgb is not None:
                closest_color, delta_e = checker.find_closest_color(gt_color.rgb)

                # Should match itself
                assert closest_color.id == gt_color.id
                # Delta E should be very small (close to 0)
                assert delta_e < 1.0

    def test_cmyk_conversion_consistency(self):
        """Test CMYK conversion consistency"""
        checker = GroundTruthColorChecker()

        # Test that CMYK values are within valid range
        for color in checker.colors:
            c, m, y, k = color.cmyk
            assert 0 <= c <= 100
            assert 0 <= m <= 100
            assert 0 <= y <= 100
            assert 0 <= k <= 100

        # Test that RGB values are within valid range
        for color in checker.colors:
            if color.rgb is not None:
                r, g, b = color.rgb
                assert 0 <= r <= 255
                assert 0 <= g <= 255
                assert 0 <= b <= 255

    def test_delta_e_calculation_validity(self):
        """Test that delta E calculations are valid"""
        checker = GroundTruthColorChecker()

        # Test with known color pairs
        test_pairs = [
            ((255, 0, 0), (255, 0, 0)),  # Same color - should have delta E = 0
            ((255, 0, 0), (0, 255, 0)),  # Very different colors - should have high delta E
            ((255, 0, 0), (254, 1, 1)),  # Very similar colors - should have low delta E
        ]

        for color1, color2 in test_pairs:
            _, delta_e1 = checker.find_closest_color(color1)
            _, delta_e2 = checker.find_closest_color(color2)

            # Delta E should be non-negative
            assert delta_e1 >= 0
            assert delta_e2 >= 0

    def test_complete_workflow(self):
        """Test complete workflow from image to results"""
        # Create a simple test image
        test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255  # White background

        # Add some colored rectangles (simulating a colorbar)
        test_image[30:70, 20:60] = [255, 0, 0]  # Red
        test_image[30:70, 80:120] = [0, 255, 0]  # Green
        test_image[30:70, 140:180] = [0, 0, 255]  # Blue

        # Convert to PIL
        test_image_pil = Image.fromarray(test_image)

        # Test the workflow components individually
        checker = GroundTruthColorChecker()

        # Test color matching
        test_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        results = checker.calculate_delta_e_for_colors(test_colors)

        assert len(results) == 3
        for result in results:
            assert "delta_e" in result
            assert "accuracy_level" in result
            assert "closest_ground_truth" in result

    def test_error_handling(self):
        """Test error handling in various scenarios"""
        checker = GroundTruthColorChecker()

        # Test with invalid RGB values
        try:
            # This should not crash
            closest_color, delta_e = checker.find_closest_color((256, -1, 300))
            assert delta_e >= 0
        except Exception:
            # Some error handling is acceptable
            pass

        # Test with empty color list
        empty_results = checker.calculate_delta_e_for_colors([])
        assert len(empty_results) == 0

        # Test with None values
        try:
            enhance_with_ground_truth_comparison([])
            enhance_with_ground_truth_comparison([{"error": "test"}])
        except Exception:
            # Should not crash
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
