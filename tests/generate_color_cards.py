import os
import sys
from PIL import Image
import numpy as np

# 将项目根目录添加到Python路径中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from core.color.utils import cmyk_to_rgb

# --- START: 从 ground_truth_checker.py 复制的 standard_cards_data ---
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
# --- END: 数据复制区域 ---

def generate_color_cards_image(output_basename="color_cards"):
    """
    直接从源数据生成 PNG 和 BMP 图像。
    """
    # 尺寸定义
    DPI = 96
    INCH_TO_CM = 2.54
    px_per_cm = DPI / INCH_TO_CM
    px_per_mm = px_per_cm / 10
    swatch_size_px = int(8 * px_per_mm)
    swatch_spacing_px = int(2 * px_per_mm)
    card_spacing_px = int(10 * px_per_mm)
    padding_px = int(10 * px_per_mm)
    num_cards = len(standard_cards_data)
    swatches_per_card = 7
    card_width_px = (swatches_per_card * swatch_size_px) + ((swatches_per_card - 1) * swatch_spacing_px)
    total_width = card_width_px + (2 * padding_px)
    total_height = (num_cards * swatch_size_px) + ((num_cards - 1) * card_spacing_px) + (2 * padding_px)

    # 创建一个 8-bit sRGB 画布 (用于 PNG 和 BMP)
    srgb_8bit_canvas = np.full((total_height, total_width, 3), 255, dtype=np.uint8)

    # 循环并直接填充画布
    card_data_list = sorted(standard_cards_data.values(), key=lambda x: x['id'])
    start_x = padding_px
    current_y = padding_px
    for card_data in card_data_list:
        color_info_list = sorted(card_data["colors"], key=lambda c: c['position'])
        current_x = start_x
        for color_info in color_info_list:
            # 获取源颜色值
            cmyk_tuple = color_info["cmyk"]
            cmyk_norm = np.array([c / 100.0 for c in cmyk_tuple]).reshape(1, 1, 4)
            
            # 转换得到 8-bit sRGB 颜色
            srgb_8bit_value = cmyk_to_rgb(cmyk_norm)[0, 0]

            # 计算坐标并填充画布
            top_left_x = current_x
            top_left_y = current_y
            bottom_right_x = top_left_x + swatch_size_px
            bottom_right_y = top_left_y + swatch_size_px

            srgb_8bit_canvas[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = srgb_8bit_value
            
            current_x += swatch_size_px + swatch_spacing_px
        current_y += swatch_size_px + card_spacing_px

    # --- 从最终的画布数组保存文件 ---
    output_dir = os.path.join(project_root, "test_output", "generate_color_cards_output")
    os.makedirs(output_dir, exist_ok=True)

    # 从数组创建 Pillow Image 对象
    image_srgb_8bit = Image.fromarray(srgb_8bit_canvas, mode='RGB')

    # 保存 PNG (8-bit sRGB)
    png_path = os.path.join(output_dir, f"{output_basename}.png")
    image_srgb_8bit.save(png_path)
    print(f"8-bit PNG 已保存: {os.path.abspath(png_path)}")

    # 保存 BMP (8-bit sRGB)
    bmp_path = os.path.join(output_dir, f"{output_basename}.bmp")
    image_srgb_8bit.save(bmp_path)
    print(f"8-bit BMP 已保存: {os.path.abspath(bmp_path)}")


if __name__ == "__main__":
    generate_color_cards_image()