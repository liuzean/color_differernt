"""
Ground Truth Color Checker - 标准色卡匹配系统

重新设计为支持多个标准色卡的完整色板匹配系统。
从原有的单一颜色匹配升级为整套色卡匹配，支持11套标准色卡。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from PIL import Image
import cv2


@dataclass
class StandardColorBlock:
    """标准色块数据类"""
    position: int
    cmyk: Tuple[int, int, int, int]
    rgb: Tuple[int, int, int] = None
    lab: Tuple[float, float, float] = None
    
    def __post_init__(self):
        if self.rgb is None:
            self.rgb = self._cmyk_to_rgb(self.cmyk)
        if self.lab is None:
            self.lab = self._rgb_to_lab(self.rgb)
    
    def _cmyk_to_rgb(self, cmyk: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
        """CMYK转RGB"""
        c, m, y, k = [x/100.0 for x in cmyk]
        r = 255 * (1-c) * (1-k)
        g = 255 * (1-m) * (1-k)
        b = 255 * (1-y) * (1-k)
        return (int(r), int(g), int(b))
    
    def _rgb_to_lab(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """RGB转Lab色彩空间的简化实现"""
        try:
            # 尝试使用现有的颜色转换工具
            from .utils import rgb_to_lab
            return rgb_to_lab(rgb)
        except:
            # 简化实现作为回退
            r, g, b = [x/255.0 for x in rgb]
            return (r*100, g*100-50, b*100-50)


@dataclass
class StandardColorCard:
    """标准色卡数据类"""
    id: str
    colors: List[StandardColorBlock]
    name: str = ""
    description: str = ""
    
    def __post_init__(self):
        # 将字典转换为StandardColorBlock对象
        if self.colors and isinstance(self.colors[0], dict):
            self.colors = [StandardColorBlock(**color_data) for color_data in self.colors]
        
        # 自动生成名称和描述
        if not self.name:
            self.name = f"标准色卡 {self.id}"
        if not self.description:
            self.description = f"包含{len(self.colors)}个标准颜色的色卡"


@dataclass
class ColorBlockComparison:
    """单个色块比较结果"""
    position: int
    detected_rgb: Tuple[int, int, int]
    detected_cmyk: Tuple[int, int, int, int]
    standard_block: StandardColorBlock
    delta_e: float
    accuracy_level: str
    is_excellent: bool
    is_acceptable: bool
    is_missing: bool = False


@dataclass
class ColorCardMatchResult:
    """色卡匹配结果数据类"""
    matched_card: StandardColorCard
    total_delta_e: float
    average_delta_e: float
    block_comparisons: List[ColorBlockComparison]
    match_quality: str
    excellent_blocks: int
    acceptable_blocks: int
    poor_blocks: int
    missing_blocks: int
    color_count: int


# 向后兼容的旧数据结构
@dataclass
class GroundTruthColor:
    """向后兼容的标准颜色数据类"""
    id: int
    name: str
    cmyk: Tuple[float, float, float, float]
    rgb: Tuple[int, int, int] = None
    lab: Tuple[float, float, float] = None


class GroundTruthChecker:
    """标准色卡匹配器 - 支持11套标准色卡的完整匹配系统"""
    
    def __init__(self):
        self.standard_cards = self._load_standard_color_cards()
        # 向后兼容：从第一套色卡生成旧格式的颜色列表
        self.colors = self._create_legacy_colors()
    
    def _load_standard_color_cards(self) -> Dict[str, StandardColorCard]:
        """加载重组后的标准色卡数据"""

        standard_cards_data = {
            "card_001": {
                "id": "card_001",
                "colors": [
                    {"position": 1, "cmyk": (100, 0, 0, 0)},   # 色板1-1
                    {"position": 2, "cmyk": (0, 100, 0, 0)},   # 色板2-1
                    {"position": 3, "cmyk": (0, 0, 100, 0)},   # 色板3-1
                    {"position": 4, "cmyk": (0, 0, 0, 100)},   # 色板4-1
                    {"position": 5, "cmyk": (0, 55, 100, 0)},  # 色板5-1
                    {"position": 6, "cmyk": (90, 0, 100, 0)},  # 色板6-1
                    {"position": 7, "cmyk": (80, 100, 0, 0)},  # 色板7-1
                ]
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
                ]
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
                ]
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
                ]
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
                ]
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
                ]
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
                ]
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
                ]
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
                ]
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
                ]
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
                ]
            }
        }

        cards = {}
        for card_id, card_data in standard_cards_data.items():
            cards[card_id] = StandardColorCard(**card_data)

        return cards
    
    def _create_legacy_colors(self) -> List[GroundTruthColor]:
        """从第一套标准色卡创建向后兼容的颜色列表"""
        legacy_colors = []
        if "card_001" in self.standard_cards:
            card_001 = self.standard_cards["card_001"]
            for i, color_block in enumerate(card_001.colors):
                legacy_color = GroundTruthColor(
                    id=i + 1,
                    name=f"Standard Color {i + 1}",
                    cmyk=tuple(float(x) for x in color_block.cmyk),
                    rgb=color_block.rgb,
                    lab=color_block.lab
                )
                legacy_colors.append(legacy_color)
        return legacy_colors
    
    def calculate_delta_e(self, rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]) -> float:
        """计算两个RGB颜色之间的Delta E差异"""
        try:
            # 尝试使用专业的颜色转换工具
            from .utils import calculate_color_difference
            return calculate_color_difference(rgb1, rgb2)
        except ImportError:
            # 简化的Delta E计算作为回退
            r1, g1, b1 = rgb1
            r2, g2, b2 = rgb2
            
            # 使用欧几里得距离的归一化版本作为简化Delta E
            delta_e = np.sqrt((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2) / np.sqrt(3) / 255 * 100
            return delta_e
    
    def compare_colorbar_with_card(
        self, 
        detected_colors: List[Tuple[int, int, int]], 
        detected_cmyk_colors: List[Tuple[int, int, int, int]],
        card: StandardColorCard
    ) -> ColorCardMatchResult:
        """
        将检测到的色板与单个标准色卡进行完整比较
        
        Args:
            detected_colors: 检测到的RGB颜色列表
            detected_cmyk_colors: 检测到的CMYK颜色列表  
            card: 标准色卡
            
        Returns:
            ColorCardMatchResult: 完整的匹配结果
        """
        block_comparisons = []
        total_delta_e = 0.0
        excellent_count = 0
        acceptable_count = 0
        poor_count = 0
        missing_count = 0
        
        # 处理颜色数量不匹配的情况
        max_blocks = max(len(detected_colors), len(card.colors))
        
        for i in range(max_blocks):
            if i < len(detected_colors) and i < len(card.colors):
                # 正常比较
                detected_rgb = detected_colors[i]
                detected_cmyk = detected_cmyk_colors[i] if i < len(detected_cmyk_colors) else (0, 0, 0, 0)
                standard_block = card.colors[i]
                
                delta_e = self.calculate_delta_e(detected_rgb, standard_block.rgb)
                total_delta_e += delta_e
                
                accuracy_level = self._get_accuracy_level(delta_e)
                is_excellent = delta_e < 1.0
                is_acceptable = delta_e < 3.0
                
                if is_excellent:
                    excellent_count += 1
                elif is_acceptable:
                    acceptable_count += 1
                else:
                    poor_count += 1
                
                comparison = ColorBlockComparison(
                    position=i + 1,
                    detected_rgb=detected_rgb,
                    detected_cmyk=detected_cmyk,
                    standard_block=standard_block,
                    delta_e=delta_e,
                    accuracy_level=accuracy_level,
                    is_excellent=is_excellent,
                    is_acceptable=is_acceptable,
                    is_missing=False
                )
                
            elif i < len(card.colors):
                # 缺失的色块（标准色卡有，但检测结果没有）
                standard_block = card.colors[i]
                missing_penalty = 50.0  # 缺失色块的惩罚Delta E值
                total_delta_e += missing_penalty
                missing_count += 1
                
                comparison = ColorBlockComparison(
                    position=i + 1,
                    detected_rgb=(128, 128, 128),  # 灰色表示缺失
                    detected_cmyk=(0, 0, 0, 50),
                    standard_block=standard_block,
                    delta_e=missing_penalty,
                    accuracy_level="Missing",
                    is_excellent=False,
                    is_acceptable=False,
                    is_missing=True
                )
            
            else:
                # 多余的色块（检测结果有，但标准色卡没有）
                # 这种情况暂时忽略，不影响匹配评分
                continue
            
            block_comparisons.append(comparison)
        
        # 计算平均Delta E
        color_count = len(block_comparisons)
        avg_delta_e = total_delta_e / color_count if color_count > 0 else float('inf')
        
        # 确定整体匹配质量
        match_quality = self._get_match_quality(avg_delta_e, excellent_count, color_count)
        
        return ColorCardMatchResult(
            matched_card=card,
            total_delta_e=total_delta_e,
            average_delta_e=avg_delta_e,
            block_comparisons=block_comparisons,
            match_quality=match_quality,
            excellent_blocks=excellent_count,
            acceptable_blocks=acceptable_count,
            poor_blocks=poor_count,
            missing_blocks=missing_count,
            color_count=color_count
        )
    
    def find_best_matching_card(
        self, 
        detected_colors: List[Tuple[int, int, int]],
        detected_cmyk_colors: List[Tuple[int, int, int, int]] = None
    ) -> Tuple[Optional[ColorCardMatchResult], Dict[str, ColorCardMatchResult]]:
        """
        找到与检测颜色最匹配的标准色卡
        
        Args:
            detected_colors: 检测到的RGB颜色列表
            detected_cmyk_colors: 检测到的CMYK颜色列表（可选）
            
        Returns:
            (最佳匹配结果, 所有匹配结果字典)
        """
        if not detected_colors:
            return None, {}
        
        # 如果没有提供CMYK，使用默认值
        if detected_cmyk_colors is None:
            detected_cmyk_colors = [(0, 0, 0, 0)] * len(detected_colors)
        
        best_match = None
        min_total_delta_e = float('inf')
        all_matches = {}
        
        # 与每个标准色卡进行比较
        for card_id, card in self.standard_cards.items():
            match_result = self.compare_colorbar_with_card(
                detected_colors, detected_cmyk_colors, card
            )
            all_matches[card_id] = match_result
            
            # 找到总Delta E最小的色卡
            if match_result.total_delta_e < min_total_delta_e:
                min_total_delta_e = match_result.total_delta_e
                best_match = match_result
        
        return best_match, all_matches
    
    def _get_accuracy_level(self, delta_e: float) -> str:
        """根据Delta E值获取准确度级别"""
        if delta_e < 1.0:
            return "Excellent"
        elif delta_e < 3.0:
            return "Acceptable"
        elif delta_e < 6.0:
            return "Noticeable"
        else:
            return "Poor"
    
    def _get_match_quality(self, avg_delta_e: float, excellent_count: int, total_count: int) -> str:
        """根据平均Delta E和优秀色块比例获取整体匹配质量"""
        excellent_ratio = excellent_count / total_count if total_count > 0 else 0
        
        if avg_delta_e < 1.5 and excellent_ratio >= 0.7:
            return "Excellent Match"
        elif avg_delta_e < 3.0 and excellent_ratio >= 0.5:
            return "Good Match"
        elif avg_delta_e < 5.0:
            return "Fair Match"
        else:
            return "Poor Match"
    
    def get_all_cards_info(self) -> Dict[str, Dict[str, Any]]:
        """获取所有标准色卡的详细信息"""
        cards_info = {}
        for card_id, card in self.standard_cards.items():
            cards_info[card_id] = {
                "id": card.id,
                "name": card.name,
                "description": card.description,
                "color_count": len(card.colors),
                "colors": [
                    {
                        "position": block.position,
                        "cmyk": block.cmyk,
                        "rgb": block.rgb,
                        "lab": block.lab
                    }
                    for block in card.colors
                ]
            }
        return cards_info
    
    def generate_card_reference_image(self, card_id: str, block_size: int = 100) -> Optional[Image.Image]:
        """为指定色卡生成参考图像"""
        if card_id not in self.standard_cards:
            return None
        
        card = self.standard_cards[card_id]
        colors = [block.rgb for block in card.colors]
        
        # 创建水平排列的色块图像
        width = len(colors) * block_size
        height = block_size
        
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        for i, color in enumerate(colors):
            x_start = i * block_size
            x_end = (i + 1) * block_size
            image[:, x_start:x_end] = color[::-1]  # BGR format for OpenCV
        
        # 转换为PIL Image (RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_rgb)
    
    # === 向后兼容方法 ===
    
    def find_closest_color(self, rgb_color: Tuple[int, int, int]) -> Tuple[Dict[str, Any], float]:
        """
        向后兼容的单色匹配方法
        从card_001中选择最接近的颜色
        """
        if "card_001" not in self.standard_cards:
            return {}, float('inf')
        
        card_001 = self.standard_cards["card_001"]
        min_delta_e = float('inf')
        closest_color = None
        
        for block in card_001.colors:
            delta_e = self.calculate_delta_e(rgb_color, block.rgb)
            if delta_e < min_delta_e:
                min_delta_e = delta_e
                closest_color = {
                    "name": f"Standard Color {block.position}",
                    "cmyk": block.cmyk,
                    "rgb": block.rgb,
                    "position": block.position
                }
        
        return closest_color, min_delta_e
    
    def generate_reference_chart(self) -> Image.Image:
        """生成所有标准色卡的参考图表"""
        chart_images = []
        
        for card_id in sorted(self.standard_cards.keys()):
            card_image = self.generate_card_reference_image(card_id, block_size=80)
            if card_image:
                chart_images.append(card_image)
        
        if not chart_images:
            return None
        
        # 垂直堆叠所有色卡图像
        total_width = chart_images[0].width
        total_height = sum(img.height for img in chart_images) + len(chart_images) * 10  # 10px间距
        
        combined_image = Image.new('RGB', (total_width, total_height), 'white')
        
        y_offset = 0
        for img in chart_images:
            combined_image.paste(img, (0, y_offset))
            y_offset += img.height + 10
        
        return combined_image
    
    def get_palette_yaml(self) -> str:
        """获取所有色卡的YAML格式配置"""
        yaml_content = "# Standard Color Cards Configuration\n"
        yaml_content += "# Generated from GroundTruthChecker\n\n"
        yaml_content += "standard_color_cards:\n"
        
        for card_id, card in self.standard_cards.items():
            yaml_content += f"  {card_id}:\n"
            yaml_content += f"    name: \"{card.name}\"\n"
            yaml_content += f"    description: \"{card.description}\"\n"
            yaml_content += f"    colors:\n"
            
            for block in card.colors:
                yaml_content += f"      - position: {block.position}\n"
                yaml_content += f"        cmyk: {block.cmyk}\n"
                yaml_content += f"        rgb: {block.rgb}\n"
            yaml_content += "\n"
        
        return yaml_content


# 创建全局实例
ground_truth_checker = GroundTruthChecker()


# 向后兼容的类别名
class GroundTruthColorChecker(GroundTruthChecker):
    """向后兼容的类别名"""
    pass