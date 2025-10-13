# -*- coding: utf-8 -*-
"""
独立测试脚本：
- 依据当前界面 Colorbar Analysis 的同一路径与参数，进行色彩空间转换与色差计算
- 支持为 Detected 与 Standard 各自提供 cmyk 或 rgb 或 lab（三选一）
- 转换严格复用项目中的工具函数与相同的形状/归一化：
  * CMYK(0-100) -> 归一化到 0-1 -> 形状 (1,1,4) -> core.color.utils.cmyk_to_rgb
  * RGB(0-255)  -> 形状 (1,1,3) -> core.color.utils.rgb_to_lab
- ΔE 优先使用项目内与主流程一致的方法；若未找到则回退 CIE76
运行方式(Windows PowerShell):
  python tests/test_color-difference.py
"""

from __future__ import annotations
import os
import sys
import math
import numpy as np

# 将项目根目录加入 sys.path，确保可导入 core.*
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 与主流程一致的工具函数
from core.color.utils import cmyk_to_rgb, rgb_to_lab  # 必需：Colorbar Analysis 使用的路径

# 尝试导入项目中用于 ΔE 的实现（与 Colorbar Analysis 保持一致性）
def _select_delta_e():
    candidates = [
        ("core.color.utils", "delta_e"),        # 通用命名
        ("core.color.utils", "delta_e_cie76"),  # CIE76
        ("core.color.utils", "delta_e_76"),     # CIE76 另一命名
        ("core.color.utils", "ciede2000"),      # CIEDE2000
    ]
    for mod_path, name in candidates:
        try:
            mod = __import__(mod_path, fromlist=[name])
            fn = getattr(mod, name, None)
            if callable(fn):
                return fn
        except Exception:
            pass
    # 回退：CIE76
    def cie76(lab1, lab2):
        dL = float(lab1[0] - lab2[0])
        da = float(lab1[1] - lab2[1])
        db = float(lab1[2] - lab2[2])
        return math.sqrt(dL * dL + da * da + db * db)
    return cie76

DELTA_E_FN = _select_delta_e()


def to_lab_from_any(color: dict) -> tuple[np.ndarray, dict]:
    """
    将输入颜色(支持 cmyk/rgb/lab 三选一)转换为 LAB。
    返回 (lab_1d_array, trace_dict)
      - lab_1d_array: np.ndarray, shape=(3,)
      - trace_dict: 记录转换链路与中间值，便于核对
    约定：
      - cmyk: [C, M, Y, K] in 0-100
      - rgb:  [R, G, B] in 0-255
      - lab:  [L, a, b]
    """
    trace = {"input": dict(color), "path": []}

    if "lab" in color and color["lab"] is not None:
        lab = np.array(color["lab"], dtype=float).reshape(3,)
        trace["path"].append("LAB->LAB")
        trace["lab"] = lab
        return lab, trace

    if "rgb" in color and color["rgb"] is not None:
        # 与主流程一致：shape=(1,1,3)
        rgb = np.array(color["rgb"], dtype=np.uint8).reshape(1, 1, 3)
        lab_arr = rgb_to_lab(rgb)
        lab = np.array(lab_arr[0, 0], dtype=float).reshape(3,)
        trace["path"].append("RGB->LAB(rgb_to_lab)")
        trace["rgb"] = tuple(int(x) for x in rgb.reshape(3,))
        trace["lab"] = lab
        return lab, trace

    if "cmyk" in color and color["cmyk"] is not None:
        # 与 GroundTruthChecker 初始化一致：0-100 -> 0-1，shape=(1,1,4)
        cmyk_norm = np.array([c / 100.0 for c in color["cmyk"]], dtype=float).reshape(1, 1, 4)
        rgb_arr = cmyk_to_rgb(cmyk_norm)           # shape 应为 (1,1,3)
        lab_arr = rgb_to_lab(rgb_arr)              # shape (1,1,3)
        rgb = tuple(int(x) for x in rgb_arr[0, 0])
        lab = np.array(lab_arr[0, 0], dtype=float).reshape(3,)
        trace["path"].append("CMYK->RGB(cmyk_to_rgb)->LAB(rgb_to_lab)")
        trace["cmyk_norm"] = tuple(float(x) for x in cmyk_norm.reshape(4,))
        trace["rgb"] = rgb
        trace["lab"] = lab
        return lab, trace

    raise ValueError("请提供 cmyk 或 rgb 或 lab 之一，例如 {'rgb':[R,G,B]} 或 {'cmyk':[C,M,Y,K]} 或 {'lab':[L,a,b]}.")


def main():
    """
    仅设置一个检测色(Detected)与一个标准色(Standard)。
    可任选其一的色彩空间(cmyk/rgb/lab)进行输入。
    """
    # 在此设置你的测试输入：三选一即可
    Detected = {
        # "cmyk": [0, 50, 100, 0],
        "rgb": [200, 80, 40],
        # "lab": [70.0, 30.0, 70.0],
    }
    Standard = {
        "cmyk": [0, 60, 100, 0],
        # "rgb": [230, 90, 35],
        # "lab": [72.0, 35.0, 68.0],
    }

    det_lab, det_trace = to_lab_from_any(Detected)
    std_lab, std_trace = to_lab_from_any(Standard)

    delta_e = float(DELTA_E_FN(det_lab, std_lab))

    # 输出结果
    def _fmt_lab(x):
        return tuple(round(float(v), 2) for v in x.reshape(3,))

    print("=" * 60)
    print("Detected 转换路径:", " -> ".join(det_trace["path"]))
    if "rgb" in det_trace:
        print("Detected RGB:", det_trace["rgb"])
    if "cmyk_norm" in det_trace:
        print("Detected CMYK(norm 0-1):", det_trace["cmyk_norm"])
    print("Detected LAB:", _fmt_lab(det_trace["lab"]))

    print("-" * 60)
    print("Standard 转换路径:", " -> ".join(std_trace["path"]))
    if "rgb" in std_trace:
        print("Standard RGB:", std_trace["rgb"])
    if "cmyk_norm" in std_trace:
        print("Standard CMYK(norm 0-1):", std_trace["cmyk_norm"])
    print("Standard LAB:", _fmt_lab(std_trace["lab"]))

    print("-" * 60)
    print(f"ΔE: {delta_e:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()