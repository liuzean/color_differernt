# core/color/ground_truth_checker.py

"""
Ground Truth Color Checker
- Manages standard color definitions (ground truth).
- Provides functionality to find the closest standard color to a given color.
- Calculates Delta E 2000 for color difference assessment.
"""

from dataclasses import dataclass, field
import numpy as np

# 从项目内部的utils导入正确的函数
from .utils import calculate_color_difference, cmyk_to_rgb, rgb_to_lab

# 11组标准色卡数据
standard_cards_data = {
    "card_001": {
        "id": "card_001",
        "colors": [
            {"position": 1, "cmyk": (100, 0, 0, 0)},
            {"position": 2, "cmyk": (0, 100, 0, 0)},
            {"position": 3, "cmyk": (0, 0, 100, 0)},
            {"position": 4, "cmyk": (0, 0, 0, 100)},
            {"position": 5, "cmyk": (0, 55, 100, 0)},
            {"position": 6, "cmyk": (90, 0, 100, 0)},
            {"position": 7, "cmyk": (80, 100, 0, 0)},
        ],
    },
    "card_002": {
        "id": "card_002",
        "colors": [
            {"position": 1, "cmyk": (90, 0, 0, 0)},
            {"position": 2, "cmyk": (0, 90, 0, 0)},
            {"position": 3, "cmyk": (0, 0, 90, 0)},
            {"position": 4, "cmyk": (0, 0, 0, 90)},
            {"position": 5, "cmyk": (0, 50, 90, 0)},
            {"position": 6, "cmyk": (81, 0, 90, 0)},
            {"position": 7, "cmyk": (72, 90, 0, 0)},
        ],
    },
    "card_003": {
        "id": "card_003",
        "colors": [
            {"position": 1, "cmyk": (80, 0, 0, 0)},
            {"position": 2, "cmyk": (0, 80, 0, 0)},
            {"position": 3, "cmyk": (0, 0, 80, 0)},
            {"position": 4, "cmyk": (0, 0, 0, 80)},
            {"position": 5, "cmyk": (0, 44, 80, 0)},
            {"position": 6, "cmyk": (72, 0, 80, 0)},
            {"position": 7, "cmyk": (64, 80, 0, 0)},
        ],
    },
    "card_004": {
        "id": "card_004",
        "colors": [
            {"position": 1, "cmyk": (70, 0, 0, 0)},
            {"position": 2, "cmyk": (0, 70, 0, 0)},
            {"position": 3, "cmyk": (0, 0, 70, 0)},
            {"position": 4, "cmyk": (0, 0, 0, 70)},
            {"position": 5, "cmyk": (0, 38, 70, 0)},
            {"position": 6, "cmyk": (63, 0, 70, 0)},
            {"position": 7, "cmyk": (56, 70, 0, 0)},
        ],
    },
    "card_005": {
        "id": "card_005",
        "colors": [
            {"position": 1, "cmyk": (60, 0, 0, 0)},
            {"position": 2, "cmyk": (0, 60, 0, 0)},
            {"position": 3, "cmyk": (0, 0, 60, 0)},
            {"position": 4, "cmyk": (0, 0, 0, 60)},
            {"position": 5, "cmyk": (0, 33, 60, 0)},
            {"position": 6, "cmyk": (54, 0, 60, 0)},
            {"position": 7, "cmyk": (48, 60, 0, 0)},
        ],
    },
    "card_006": {
        "id": "card_006",
        "colors": [
            {"position": 1, "cmyk": (50, 0, 0, 0)},
            {"position": 2, "cmyk": (0, 50, 0, 0)},
            {"position": 3, "cmyk": (0, 0, 50, 0)},
            {"position": 4, "cmyk": (0, 0, 0, 50)},
            {"position": 5, "cmyk": (0, 28, 50, 0)},
            {"position": 6, "cmyk": (45, 0, 50, 0)},
            {"position": 7, "cmyk": (40, 50, 0, 0)},
        ],
    },
    "card_007": {
        "id": "card_007",
        "colors": [
            {"position": 1, "cmyk": (40, 0, 0, 0)},
            {"position": 2, "cmyk": (0, 40, 0, 0)},
            {"position": 3, "cmyk": (0, 0, 40, 0)},
            {"position": 4, "cmyk": (0, 0, 0, 40)},
            {"position": 5, "cmyk": (0, 22, 40, 0)},
            {"position": 6, "cmyk": (36, 0, 40, 0)},
            {"position": 7, "cmyk": (32, 40, 0, 0)},
        ],
    },
    "card_008": {
        "id": "card_008",
        "colors": [
            {"position": 1, "cmyk": (30, 0, 0, 0)},
            {"position": 2, "cmyk": (0, 30, 0, 0)},
            {"position": 3, "cmyk": (0, 0, 30, 0)},
            {"position": 4, "cmyk": (0, 0, 0, 30)},
            {"position": 5, "cmyk": (0, 16, 30, 0)},
            {"position": 6, "cmyk": (27, 0, 30, 0)},
            {"position": 7, "cmyk": (24, 30, 0, 0)},
        ],
    },
    "card_009": {
        "id": "card_009",
        "colors": [
            {"position": 1, "cmyk": (20, 0, 0, 0)},
            {"position": 2, "cmyk": (0, 20, 0, 0)},
            {"position": 3, "cmyk": (0, 0, 20, 0)},
            {"position": 4, "cmyk": (0, 0, 0, 20)},
            {"position": 5, "cmyk": (0, 11, 20, 0)},
            {"position": 6, "cmyk": (18, 0, 20, 0)},
            {"position": 7, "cmyk": (16, 20, 0, 0)},
        ],
    },
    "card_010": {
        "id": "card_010",
        "colors": [
            {"position": 1, "cmyk": (10, 0, 0, 0)},
            {"position": 2, "cmyk": (0, 10, 0, 0)},
            {"position": 3, "cmyk": (0, 0, 10, 0)},
            {"position": 4, "cmyk": (0, 0, 0, 10)},
            {"position": 5, "cmyk": (0, 6, 10, 0)},
            {"position": 6, "cmyk": (9, 0, 10, 0)},
            {"position": 7, "cmyk": (8, 10, 0, 0)},
        ],
    },
    "card_011": {
        "id": "card_011",
        "colors": [
            {"position": 1, "cmyk": (0, 0, 0, 0)},
            {"position": 2, "cmyk": (0, 0, 0, 0)},
            {"position": 3, "cmyk": (0, 0, 0, 0)},
            {"position": 4, "cmyk": (0, 0, 0, 0)},
            {"position": 5, "cmyk": (0, 0, 0, 0)},
            {"position": 6, "cmyk": (0, 0, 0, 0)},
            {"position": 7, "cmyk": (0, 0, 0, 0)},
        ],
    },
}


@dataclass
class GroundTruthColor:
    """Represents a standard ground truth color."""
    id: str
    name: str
    cmyk: tuple[int, int, int, int]
    rgb: tuple[int, int, int]
    lab: np.ndarray = field(default_factory=lambda: np.zeros(3))
    position: int = 0


class GroundTruthChecker:
    """Manages and compares against a set of standard ground truth colors."""

    def __init__(self):
        self.standard_cards = self._create_standard_color_cards()
        self.standard_colors: list[GroundTruthColor] = self.standard_cards.get("card_001", [])

    def _create_standard_color_cards(self) -> dict[str, list[GroundTruthColor]]:
        """Loads all standard color cards from the predefined data."""
        cards = {}
        for card_id, card_data in standard_cards_data.items():
            color_list = []
            for color_info in card_data["colors"]:
                cmyk = color_info["cmyk"]
                position = color_info["position"]
                
                cmyk_norm = np.array([c/100.0 for c in cmyk]).reshape(1, 1, 4)
                rgb_array = cmyk_to_rgb(cmyk_norm)
                rgb = tuple(rgb_array[0, 0])

                lab_array = rgb_to_lab(rgb_array)
                lab = lab_array[0, 0]

                color_name = f"C{cmyk[0]} M{cmyk[1]} Y{cmyk[2]} K{cmyk[3]}"
                gt_color = GroundTruthColor(
                    id=f"{card_id}_p{position}",
                    name=color_name,
                    cmyk=cmyk,
                    rgb=rgb,
                    lab=lab,
                    position=position,
                )
                color_list.append(gt_color)
            cards[card_id] = color_list
        return cards

    def _get_accuracy_level(self, delta_e: float) -> str:
        """Returns a descriptive accuracy level based on the Delta E value."""
        if delta_e < 1.0:
            return "Excellent"
        if delta_e < 2.0:
            return "Very Good"
        if delta_e < 3.0:
            return "Good"
        if delta_e < 5.0:
            return "Fair"
        if delta_e < 10.0:
            return "Poor"
        return "Very Poor"

    # [新函数] 内部辅助函数，用于计算色差总和
    def _calculate_total_delta_e(
        self,
        detected_colors_rgb: list[tuple[int, int, int]],
        standard_colors: list[GroundTruthColor],
    ) -> float:
        """Calculates the total deltaE sum based on the number of detected colors."""
        
        n_detected = len(detected_colors_rgb)
        
        # Case 1: More than 7 blocks detected -> Invalid
        if n_detected > 7:
            return float("inf") # Return infinity to mark as invalid

        # Case 2: Exactly 7 blocks detected -> Try forward and reverse matching
        if n_detected == 7:
            total_delta_e_forward = 0
            total_delta_e_reverse = 0
            
            sorted_standard = sorted(standard_colors, key=lambda c: c.position)
            
            for i in range(7):
                # Forward match
                img1 = np.array([[detected_colors_rgb[i]]], dtype=np.uint8)
                img2_fwd = np.array([[sorted_standard[i].rgb]], dtype=np.uint8)
                delta_e_fwd, _ = calculate_color_difference(img1, img2_fwd)
                total_delta_e_forward += delta_e_fwd
                
                # Reverse match
                img2_rev = np.array([[sorted_standard[6 - i].rgb]], dtype=np.uint8)
                delta_e_rev, _ = calculate_color_difference(img1, img2_rev)
                total_delta_e_reverse += delta_e_rev
            
            return min(total_delta_e_forward, total_delta_e_reverse)

        # Case 3: Less than 7 blocks detected -> Find best match for each
        if n_detected < 7:
            total_delta_e = 0
            for detected_rgb in detected_colors_rgb:
                min_delta_e_for_block = float("inf")
                img1 = np.array([[detected_rgb]], dtype=np.uint8)
                
                for gt_color in standard_colors:
                    img2 = np.array([[gt_color.rgb]], dtype=np.uint8)
                    delta_e, _ = calculate_color_difference(img1, img2)
                    if delta_e < min_delta_e_for_block:
                        min_delta_e_for_block = delta_e
                
                total_delta_e += min_delta_e_for_block
            return total_delta_e
            
        return float("inf") # Should not be reached

    # 原函数名: find_best_card_for_colorbar
    def find_best_card_for_colorbar_new(
        self, detected_colors_rgb: list[tuple[int, int, int]]
    ) -> dict | None:
        """
        [新逻辑] Finds the best matching standard card for a list of detected colors
        based on the new matching rules.
        """
        if not detected_colors_rgb:
            return None
        
        # If more than 7 colors are detected, return a specific status
        if len(detected_colors_rgb) > 7:
            return {"best_card_id": "INVALID_DETECTION", "results": []}

        card_scores = {}
        for card_id, standard_colors in self.standard_cards.items():
            card_scores[card_id] = self._calculate_total_delta_e(
                detected_colors_rgb, standard_colors
            )
        
        if not card_scores:
            return None

        # Find the card with the minimum total delta E
        best_card_id = min(card_scores, key=card_scores.get)
        
        # --- Generate final detailed results based on the best card ---
        best_card_colors = sorted(self.standard_cards[best_card_id], key=lambda c: c.position)
        results = []

        # If exactly 7, we need to determine if forward or reverse was better to report correctly
        if len(detected_colors_rgb) == 7:
            total_fwd = 0
            total_rev = 0
            for i in range(7):
                img1 = np.array([[detected_colors_rgb[i]]], dtype=np.uint8)
                img2_fwd = np.array([[best_card_colors[i].rgb]], dtype=np.uint8)
                delta_fwd, _ = calculate_color_difference(img1, img2_fwd)
                total_fwd += delta_fwd
                
                img2_rev = np.array([[best_card_colors[6 - i].rgb]], dtype=np.uint8)
                delta_rev, _ = calculate_color_difference(img1, img2_rev)
                total_rev += delta_rev

            is_reverse_match = total_rev < total_fwd
            
            for i in range(7):
                gt_color = best_card_colors[6 - i] if is_reverse_match else best_card_colors[i]
                img1 = np.array([[detected_colors_rgb[i]]], dtype=np.uint8)
                img2 = np.array([[gt_color.rgb]], dtype=np.uint8)
                delta_e, _ = calculate_color_difference(img1, img2)
                results.append({
                    "detected_rgb": detected_colors_rgb[i],
                    "closest_ground_truth": gt_color,
                    "delta_e": delta_e,
                    "accuracy_level": self._get_accuracy_level(delta_e),
                })
        
        # If less than 7, find the best match for each individually against the best card
        elif len(detected_colors_rgb) < 7:
            for detected_rgb in detected_colors_rgb:
                min_delta_e = float("inf")
                best_gt_color = None
                img1 = np.array([[detected_rgb]], dtype=np.uint8)

                for gt_color in best_card_colors:
                    img2 = np.array([[gt_color.rgb]], dtype=np.uint8)
                    delta_e, _ = calculate_color_difference(img1, img2)
                    if delta_e < min_delta_e:
                        min_delta_e = delta_e
                        best_gt_color = gt_color
                
                results.append({
                    "detected_rgb": detected_rgb,
                    "closest_ground_truth": best_gt_color,
                    "delta_e": min_delta_e,
                    "accuracy_level": self._get_accuracy_level(min_delta_e),
                })

        return {"best_card_id": best_card_id, "results": results}
    
    # 保留旧函数以兼容
    find_best_card_for_colorbar = find_best_card_for_colorbar_new

    def find_closest_color(
        self, rgb_color: tuple[int, int, int]
    ) -> tuple[GroundTruthColor | None, float]:
        """
        [兼容性保留] Finds the closest standard color to a given RGB color from the default list.
        # WARNING: This function is deprecated...
        """
        min_delta_e = float("inf")
        closest_color = None
        detected_rgb_image = np.array([[rgb_color]], dtype=np.uint8)
        for gt_color in self.standard_colors:
            gt_rgb_image = np.array([[gt_color.rgb]], dtype=np.uint8)
            delta_e, _ = calculate_color_difference(detected_rgb_image, gt_rgb_image)
            if delta_e < min_delta_e:
                min_delta_e = delta_e
                closest_color = gt_color
        return closest_color, min_delta_e

# Singleton instance for easy access across the application
ground_truth_checker = GroundTruthChecker()