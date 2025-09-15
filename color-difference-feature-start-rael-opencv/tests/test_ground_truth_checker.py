#!/usr/bin/env python3

"""
Test Ground Truth Color Checker Module

Tests for the ground truth color checker functionality including
delta E calculations using ICC-based color conversion.
"""

from core.color.ground_truth_checker import GroundTruthColorChecker


class TestGroundTruthColorChecker:
    """Test the ground truth color checker functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        self.checker = GroundTruthColorChecker()

    def test_color_initialization(self):
        """Test that ground truth colors are initialized correctly"""
        colors = self.checker.get_all_colors()
        assert len(colors) == 7, "Should have 7 standard colors"

        # Test specific colors
        cyan = self.checker.get_color_by_id(1)
        assert cyan is not None, "Cyan color should exist"
        assert cyan.name == "Pure Cyan", "Cyan name should be correct"
        assert cyan.cmyk == (100, 0, 0, 0), "Cyan CMYK should be correct"
        assert cyan.rgb is not None, "Cyan RGB should be computed"
        assert cyan.lab is not None, "Cyan LAB should be computed"

    def test_delta_e_calculation_cyan(self):
        """Test delta E calculation for cyan color"""
        # User's reported colors
        detected_cyan = (1, 158, 230)  # Detected cyan
        ground_truth_cyan = (0, 161, 232)  # Ground truth cyan

        # Get ground truth cyan color
        gt_cyan = self.checker.get_color_by_id(1)
        assert gt_cyan is not None, "Ground truth cyan should exist"

        # Calculate delta E between detected and ground truth
        delta_e = self.checker._calculate_single_color_delta_e(detected_cyan, gt_cyan)

        print(f"Detected cyan: {detected_cyan}")
        print(f"Ground truth cyan RGB: {gt_cyan.rgb}")
        print(f"Ground truth cyan CMYK: {gt_cyan.cmyk}")
        print(f"Delta E: {delta_e}")

        # Delta E should be low for such similar colors
        assert delta_e < 10.0, f"Delta E should be low for similar colors, got {delta_e}"

    def test_delta_e_calculation_direct_comparison(self):
        """Test delta E calculation between the two similar cyan colors"""
        # User's reported colors
        detected_cyan = (1, 158, 230)
        ground_truth_cyan = (0, 161, 232)

        # Create a temporary ground truth color for comparison
        from core.color.ground_truth_checker import GroundTruthColor
        temp_gt = GroundTruthColor(id=999, name="Test Cyan", cmyk=(100, 0, 0, 0))
        temp_gt.rgb = ground_truth_cyan

        # Calculate delta E
        delta_e = self.checker._calculate_single_color_delta_e(detected_cyan, temp_gt)

        print(f"Direct comparison - Detected: {detected_cyan}, GT: {ground_truth_cyan}")
        print(f"Delta E: {delta_e}")

        # These colors are very similar, delta E should be low
        assert delta_e < 5.0, f"Delta E should be very low for such similar colors, got {delta_e}"

    def test_fixed_order_comparison(self):
        """Test fixed order comparison functionality"""
        # Test with some detected colors
        detected_colors = [
            (1, 158, 230),   # Detected cyan
            (220, 50, 150),  # Detected magenta
            (255, 255, 0),   # Detected yellow
        ]

        results = self.checker.calculate_delta_e_fixed_order(detected_colors)

        assert len(results) == 3, "Should have 3 results"

        # Check cyan result
        cyan_result = results[0]
        assert cyan_result["detected_color_id"] == 0
        assert cyan_result["detected_rgb"] == (1, 158, 230)
        assert cyan_result["expected_ground_truth"]["name"] == "Pure Cyan"

        print(f"Cyan delta E: {cyan_result['delta_e']}")
        print(f"Cyan accuracy: {cyan_result['accuracy_level']}")

        # Delta E should be reasonable for cyan
        assert cyan_result["delta_e"] < 20.0, f"Cyan delta E should be reasonable, got {cyan_result['delta_e']}"

    def test_closest_color_matching(self):
        """Test finding closest color functionality"""
        # Test with a cyan-like color
        test_color = (0, 160, 230)

        closest_color, delta_e = self.checker.find_closest_color(test_color)

        assert closest_color is not None, "Should find a closest color"
        assert closest_color.name == "Pure Cyan", "Should match to cyan"

        print(f"Closest color: {closest_color.name}")
        print(f"Delta E: {delta_e}")

        # Should find cyan as closest
        assert delta_e < 15.0, f"Delta E to closest cyan should be reasonable, got {delta_e}"

    def test_accuracy_levels(self):
        """Test accuracy level classification"""
        # Test with various delta E values
        assert self.checker._get_accuracy_level(1.0) == "Acceptable"
        assert self.checker._get_accuracy_level(2.9) == "Acceptable"
        assert self.checker._get_accuracy_level(3.1) == "Poor"
        assert self.checker._get_accuracy_level(10.0) == "Poor"

    def test_summary_stats(self):
        """Test summary statistics"""
        stats = self.checker.get_summary_stats()

        assert stats["total_colors"] == 7
        assert "Pure Cyan" in stats["color_names"]
        assert "Pure Magenta" in stats["color_names"]
        assert "Pure Yellow" in stats["color_names"]
        assert "Pure Black" in stats["color_names"]

        # Check CMYK ranges
        cmyk_range = stats["cmyk_range"]
        assert cmyk_range["c_min"] == 0
        assert cmyk_range["c_max"] == 100
        assert cmyk_range["m_min"] == 0
        assert cmyk_range["m_max"] == 100


if __name__ == "__main__":
    # Run basic tests
    test_checker = TestGroundTruthColorChecker()
    test_checker.setup_method()

    print("=== Testing Ground Truth Color Checker ===")

    try:
        test_checker.test_color_initialization()
        print("✓ Color initialization test passed")

        test_checker.test_delta_e_calculation_cyan()
        print("✓ Delta E calculation test passed")

        test_checker.test_delta_e_calculation_direct_comparison()
        print("✓ Direct comparison test passed")

        test_checker.test_fixed_order_comparison()
        print("✓ Fixed order comparison test passed")

        test_checker.test_closest_color_matching()
        print("✓ Closest color matching test passed")

        test_checker.test_accuracy_levels()
        print("✓ Accuracy level test passed")

        test_checker.test_summary_stats()
        print("✓ Summary stats test passed")

        print("\n=== All tests passed! ===")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
