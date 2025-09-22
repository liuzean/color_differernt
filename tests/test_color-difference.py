import re
import sys
import os
import numpy as np

# --- 路径调整 ---
# 为了确保脚本在 'tests/' 目录下运行时能够找到 'core' 模块，
# 我们需要将项目的根目录添加到 Python 的模块搜索路径中。
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# --- 核心模块导入 ---
from core.color import utils as color_utils
from core.color import icc_trans


def parse_cmyk_string(cmyk_string: str) -> tuple[int, int, int, int] | None:
    """
    解析类似 "C89 M67 Y20 K0" 的字符串为 CMYK 元组 (0-100 范围)。
    """
    match = re.match(
        r"C(\d{1,3})\s*M(\d{1,3})\s*Y(\d{1,3})\s*K(\d{1,3})",
        cmyk_string.strip(),
        re.IGNORECASE,
    )
    if not match:
        print(f"错误：无法解析CMYK字符串 '{cmyk_string}'")
        return None

    c, m, y, k = map(int, match.groups())

    if not all(0 <= val <= 100 for val in (c, m, y, k)):
        print(f"错误：CMYK值必须在 0-100 之间。输入值为 C{c} M{m} Y{y} K{k}")
        return None

    return c, m, y, k


def convert_cmyk_to_rgb(cmyk_tuple: tuple[int, int, int]) -> tuple[int, int, int]:
    """
    将单个CMYK元组转换为RGB元组。
    """
    # 将CMYK值放入一个 1x1 像素的 numpy 数组中以匹配函数输入格式
    # 注意：CMYK元组应为4个值，已修正函数签名
    cmyk_pixel_array = np.array([[cmyk_tuple]], dtype=np.uint8)
    # 调用项目中的核心转换函数
    rgb_array, _ = icc_trans.cmyk_to_srgb_array(cmyk_pixel_array)
    # 从返回的数组中提取出RGB元组
    return tuple(rgb_array[0][0])


def main():
    """
    主函数，执行两个CMYK值之间的色差计算流程。
    """
    # --- 输入 ---
    # 定义你的 Detected 和 Standard CMYK 值
    detected_cmyk_str = "C19 M40 Y16 K0"
    standard_cmyk_str = "C0 M0 Y0 K0"

    # --- 流程开始 ---
    print("开始计算色差...")
    print(f"检测色 (Detected): '{detected_cmyk_str}'")
    print(f"标准色 (Standard): '{standard_cmyk_str}'")
    print("-" * 30)

    # 1. 解析两个 CMYK 字符串
    detected_cmyk = parse_cmyk_string(detected_cmyk_str)
    standard_cmyk = parse_cmyk_string(standard_cmyk_str)

    if detected_cmyk is None or standard_cmyk is None:
        return

    # 2. 将两个 CMYK 值分别转换为 RGB
    detected_rgb = convert_cmyk_to_rgb(detected_cmyk)
    standard_rgb = convert_cmyk_to_rgb(standard_cmyk)

    print(f"检测色转换后的RGB值: {detected_rgb}")
    print(f"标准色转换后的RGB值: {standard_rgb}")

    # 3. 计算两个 RGB 值之间的色差
    #    *** 已更正为原始代码中的正确函数名 ***
    #    注意：原始代码的函数返回 avg_delta_e 和一个 delta_e_map，我们这里只取前者。
    avg_delta_e, _ = color_utils.calculate_color_difference(
        np.array([[detected_rgb]], dtype=np.uint8),
        np.array([[standard_rgb]], dtype=np.uint8),
    )


    # --- 输出结果 ---
    print("\n" + "=" * 30)
    print("       最终计算结果")
    print("=" * 30)
    print(f"色差值 (Delta E 2000): {avg_delta_e:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    main()