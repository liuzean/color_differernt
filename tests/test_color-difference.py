# -*- coding: utf-8 -*-


"""
独立测试脚本：用于检验当前代码是否能成功调用 core.color.utils.calculate_color_difference。并且检测界面显示的 ΔE 值与该脚本计算的 ΔE 值一致。
- 使用 core.color.utils.calculate_color_difference 计算 ΔE（仅此实现）
- 输入：Detected 与 Standard 各自提供 cmyk 或 rgb 或 lab（三选一）
  * CMYK(0-100) -> /100 -> shape(1,1,4) -> cmyk_to_rgb -> 得到 RGB(1,1,3)
  * RGB(0-255)  -> 直接构造 1x1x3 图像
  * LAB 不支持（calculate_color_difference 仅接受 RGB 图像）
- CMYK 输出统一为 0-100 的整数；同时打印双方 LAB
运行(Windows PowerShell):
  python tests/test-color-difference.py
"""

from __future__ import annotations
import os
import sys
import numpy as np

# 将项目根目录加入 sys.path，确保可导入 core.*
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 项目内函数（与主流程一致）
from core.color.utils import cmyk_to_rgb, rgb_to_lab
try:
    from core.color.utils import calculate_color_difference  # 仅使用该实现
except Exception:
    print("错误：未找到 core.color.utils.calculate_color_difference。")
    sys.exit(2)


def _cmyk_percent_to_rgb_img(cmyk_percent) -> np.ndarray:
    """
    CMYK 百分比(0-100) -> /100 -> shape=(1,1,4) -> cmyk_to_rgb -> 1x1x3 uint8 RGB图像
    """
    cmyk_norm = np.array([float(c) / 100.0 for c in cmyk_percent], dtype=float).reshape(1, 1, 4)
    rgb_arr = cmyk_to_rgb(cmyk_norm)  # 期望 shape=(1,1,3)
    rgb = np.array(rgb_arr[0, 0], dtype=np.float32)
    rgb_uint8 = np.clip(np.round(rgb), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    return rgb_uint8


def _rgb_tuple_to_img(rgb_255) -> np.ndarray:
    """
    单一 RGB 三元组(0-255) -> 1x1x3 uint8 图像
    """
    r, g, b = [int(x) for x in rgb_255]
    return np.array([[[r, g, b]]], dtype=np.uint8)


def _rgb_img_to_lab_tuple(rgb_img_1x1: np.ndarray) -> tuple[float, float, float]:
    """
    1x1x3 RGB 图像 -> LAB 三元组(tuple, 保留两位小数)
    """
    lab_arr = rgb_to_lab(rgb_img_1x1)  # 期望 (1,1,3)
    lab = np.array(lab_arr[0, 0], dtype=float).reshape(3,)
    return tuple(round(float(v), 2) for v in lab)


def to_rgb_img_from_any(color: dict, role: str) -> tuple[np.ndarray, dict]:
    """
    将输入颜色(cmyk/rgb/lab 三选一)转换为 1x1 的 RGB 图像，供 calculate_color_difference 使用。
    返回 (rgb_img(1,1,3), trace_dict)
    """
    trace = {"input": dict(color), "path": []}

    if "rgb" in color and color["rgb"] is not None:
        img = _rgb_tuple_to_img(color["rgb"])
        trace["path"].append("RGB->RGB(img 1x1)")
        trace["rgb"] = tuple(int(x) for x in img.reshape(3,))
        return img, trace

    if "cmyk" in color and color["cmyk"] is not None:
        img = _cmyk_percent_to_rgb_img(color["cmyk"])
        trace["path"].append("CMYK(0-100)->/100->cmyk_to_rgb->RGB(img 1x1)")
        trace["cmyk_percent"] = tuple(int(round(float(x))) for x in color["cmyk"])
        trace["rgb"] = tuple(int(x) for x in img.reshape(3,))
        return img, trace

    if "lab" in color and color["lab"] is not None:
        print(f"错误：{role} 仅提供了 LAB。calculate_color_difference 只接受 RGB 图像，请改为提供 RGB 或 CMYK。")
        sys.exit(3)

    raise ValueError(f"{role}: 请提供 cmyk 或 rgb 或 lab 之一。")


def main():
    """
    仅设置一个检测色(Detected)与一个标准色(Standard)；各自三选一(cmyk/rgb/lab)。
    """
    # 可根据需要修改
    Detected = {
        # "cmyk": [0, 50, 100, 0],
        "rgb": [163, 41, 70],
        # "lab": [70.0, 30.0, 70.0],
    }
    Standard = {
        "cmyk": [0, 100, 0, 0],
        # "rgb": [230, 90, 35],
        # "lab": [72.0, 35.0, 68.0],
    }

    det_img, det_trace = to_rgb_img_from_any(Detected, "Detected")
    std_img, std_trace = to_rgb_img_from_any(Standard, "Standard")

    # 仅调用 calculate_color_difference；成功则输出成功提示
    try:
        avg_delta_e, delta_e_map = calculate_color_difference(det_img, std_img)
        print("调用calculate_color_difference成功")
    except Exception as e:
        import traceback
        print("错误：calculate_color_difference 调用失败。详细信息：")
        traceback.print_exception(type(e), e, e.__traceback__)
        sys.exit(2)

    # 计算并打印 LAB
    det_lab = _rgb_img_to_lab_tuple(det_img)
    std_lab = _rgb_img_to_lab_tuple(std_img)

    def _fmt_rgb_img(img):
        return tuple(int(x) for x in img.reshape(3,))

    print("=" * 60)
    print("Detected 转换路径:", " -> ".join(det_trace["path"]))
    print("Detected RGB:", det_trace.get("rgb", _fmt_rgb_img(det_img)))
    if "cmyk_percent" in det_trace:
        print("Detected CMYK(0-100%):", det_trace["cmyk_percent"])
    print("Detected LAB:", det_lab)

    print("-" * 60)
    print("Standard 转换路径:", " -> ".join(std_trace["path"]))
    print("Standard RGB:", std_trace.get("rgb", _fmt_rgb_img(std_img)))
    if "cmyk_percent" in std_trace:
        print("Standard CMYK(0-100%):", std_trace["cmyk_percent"])
    print("Standard LAB:", std_lab)

    print("-" * 60)
    print(f"ΔE: {float(avg_delta_e):.2f}")
    print("使用 ΔE 算法：calculate_color_difference")
    print("=" * 60)


if __name__ == "__main__":
    main()