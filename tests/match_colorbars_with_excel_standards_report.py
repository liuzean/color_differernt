# -*- coding: utf-8 -*-
#该文件直接计算比对的是界面中的检测色块的rgb值转lab后与excel中的rgb转lab值进行对比,以此用来检测打光的问题是什么

from __future__ import annotations
import os
import sys
import json
import math
from datetime import datetime
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

# 确保可读取 .xlsx
try:
    import openpyxl  # noqa: F401
except Exception:
    print("错误：缺少 openpyxl，请先安装后再运行：")
    print(r'  .\.venv\Scripts\python.exe -m pip install openpyxl')
    sys.exit(2)

# 项目根目录
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 项目内函数
from core.color.utils import calculate_color_difference, rgb_to_lab

# 默认输入
JSON_DEFAULT = os.path.join(ROOT, "Result Output", "analysis_2025-10-22_09-49-38.json")
EXCEL_PATH = os.path.join(ROOT, "data", "wumo2.xlsx")
GROUP_SIZE = 7  # 标准色板每组 7 块

# 输出目录
OUT_DIR = os.path.join(ROOT, "Result Output")


def _rgb_tuple_to_img(rgb_255: Tuple[int, int, int]) -> np.ndarray:
    r, g, b = [int(round(float(x))) for x in rgb_255[:3]]
    r, g, b = int(np.clip(r, 0, 255)), int(np.clip(g, 0, 255)), int(np.clip(b, 0, 255))
    return np.array([[[r, g, b]]], dtype=np.uint8)  # (1,1,3)


def _delta_e_of_pair(rgb_a: Tuple[int, int, int], rgb_b: Tuple[int, int, int]) -> float:
    img_a = _rgb_tuple_to_img(rgb_a)
    img_b = _rgb_tuple_to_img(rgb_b)
    de_avg, _ = calculate_color_difference(img_a, img_b)
    return float(de_avg)


def _rgb_to_lab_tuple(rgb_255: Tuple[int, int, int]) -> Tuple[float, float, float]:
    img = _rgb_tuple_to_img(rgb_255)
    lab_arr = rgb_to_lab(img)  # 期望 (1,1,3)
    L, a, b = map(float, np.array(lab_arr[0, 0], dtype=float).reshape(3,))
    return L, a, b


def _rgb_to_cmyk_tuple(rgb_255: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    """
    将 RGB(0-255) 转为 CMYK(0-100)，用于报告显示（设备无关近似）
    """
    r, g, b = [int(np.clip(int(x), 0, 255)) for x in rgb_255]
    if r == 0 and g == 0 and b == 0:
        return 0, 0, 0, 100
    rp, gp, bp = r / 255.0, g / 255.0, b / 255.0
    k = 1.0 - max(rp, gp, bp)
    c = (1.0 - rp - k) / (1.0 - k) if k < 1.0 else 0.0
    m = (1.0 - gp - k) / (1.0 - k) if k < 1.0 else 0.0
    y = (1.0 - bp - k) / (1.0 - k) if k < 1.0 else 0.0
    C = int(np.clip(round(c * 100), 0, 100))
    M = int(np.clip(round(m * 100), 0, 100))
    Y = int(np.clip(round(y * 100), 0, 100))
    K = int(np.clip(round(k * 100), 0, 100))
    return C, M, Y, K


def _parse_json_colorbars(path: str) -> List[dict]:
    """
    解析 JSON，返回列表：[ { colorbar_id, detected_rgbs } ]
    detected_rgbs: List[Tuple[int,int,int]]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    colorbars = []
    cbs = data.get("colorbar_results") or []
    for idx, cb in enumerate(cbs):
        cb_id = cb.get("colorbar_id") or cb.get("id") or f"cb_{idx+1}"
        entries = (
            cb.get("pure_color_analyses")
            or cb.get("color_analysis")
            or cb.get("items")
            or cb.get("results")
            or []
        )
        det_rgbs = []
        for it in entries:
            rgb = it.get("pure_color_rgb")
            if rgb is None:
                det = it.get("detected") or {}
                rgb = det.get("rgb") or it.get("rgb")
            if rgb is None:
                continue
            try:
                r, g, b = [int(round(float(x))) for x in rgb[:3]]
                det_rgbs.append((r, g, b))
            except Exception:
                continue

        if det_rgbs:
            colorbars.append({"colorbar_id": cb_id, "detected_rgbs": det_rgbs})

    return colorbars


def _load_standard_cards_from_excel(path: str, group_size: int = GROUP_SIZE) -> List[List[Tuple[int, int, int]]]:
    """
    从 youmo.xlsx 读取 F,G,H 列（R,G,B），第2行起为数据。
    每 group_size 行分为一组，不足一组的尾部行丢弃。
    """
    if not os.path.exists(path):
        print(f"错误：找不到标准色板文件：{path}")
        sys.exit(2)

    df = pd.read_excel(path, header=0, usecols="F:H", engine="openpyxl")
    df.columns = ["R", "G", "B"]
    for col in ["R", "G", "B"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["R", "G", "B"]).copy()
    df[["R", "G", "B"]] = (
        df[["R", "G", "B"]]
        .round()
        .clip(lower=0, upper=255)
        .astype(int)
    )

    rgbs = [tuple(int(v) for v in row) for row in df[["R", "G", "B"]].to_numpy()]
    cards = []
    for i in range(0, len(rgbs), group_size):
        chunk = rgbs[i:i + group_size]
        if len(chunk) == group_size:
            cards.append(chunk)
    if len(rgbs) % group_size != 0:
        dropped = len(rgbs) % group_size
        print(f"提示：标准色板尾部有 {dropped} 行不足 {group_size} 的数据，已忽略。")
    if not cards:
        print("错误：未从 Excel 解析到任何完整的标准色板组。")
        sys.exit(2)
    return cards


def _match_to_card(det_rgbs: List[Tuple[int, int, int]],
                   std_card: List[Tuple[int, int, int]]) -> Tuple[float, List[Dict]]:
    """
    返回 (总色差, 明细列表)
    明细列表每项包含：
      {
        "det_rgb": (r,g,b),
        "std_rgb": (r,g,b),  # 对应到的最小ΔE标准色块
        "de": float,
        "det_lab": (L,a,b),
        "std_lab": (L,a,b),
        "det_cmyk": (C,M,Y,K),
        "std_cmyk": (C,M,Y,K),
      }
    """
    details = []
    total = 0.0
    for det_rgb in det_rgbs:
        best_std = None
        best_de = float("inf")
        for std_rgb in std_card:
            de = _delta_e_of_pair(det_rgb, std_rgb)
            if de < best_de:
                best_de = de
                best_std = std_rgb
        if best_std is None or math.isinf(best_de):
            continue

        det_lab = _rgb_to_lab_tuple(det_rgb)
        std_lab = _rgb_to_lab_tuple(best_std)
        det_cmyk = _rgb_to_cmyk_tuple(det_rgb)
        std_cmyk = _rgb_to_cmyk_tuple(best_std)

        details.append({
            "det_rgb": det_rgb,
            "std_rgb": best_std,
            "de": float(best_de),
            "det_lab": det_lab,
            "std_lab": std_lab,
            "det_cmyk": det_cmyk,
            "std_cmyk": std_cmyk,
        })
        total += float(best_de)

    return float(total), details


def _find_best_card_for_colorbar(det_rgbs: List[Tuple[int, int, int]],
                                 std_cards: List[List[Tuple[int, int, int]]]) -> Tuple[int, float, List[Dict]]:
    """
    返回 (最佳标准卡索引(0-based), 最小总色差, 最佳卡的匹配明细)
    """
    best_idx = -1
    best_score = float("inf")
    best_details: List[Dict] = []
    for i, card in enumerate(std_cards):
        score, details = _match_to_card(det_rgbs, card)
        if score < best_score:
            best_score = score
            best_idx = i
            best_details = details
    return best_idx, float(best_score), best_details


def _fmt_rgb(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"RGB=({r:3d},{g:3d},{b:3d})"


def _fmt_cmyk(cmyk: Tuple[int, int, int, int]) -> str:
    C, M, Y, K = cmyk
    return f"CMYK=({C:3d},{M:3d},{Y:3d},{K:3d})"


def _fmt_lab(lab: Tuple[float, float, float]) -> str:
    L, a, b = lab
    return f"LAB=({L:6.2f}, {a:6.2f}, {b:6.2f})"


def main():
    # 允许通过命令行覆盖 JSON 路径
    json_path = sys.argv[1] if len(sys.argv) > 1 else JSON_DEFAULT

    if not os.path.exists(json_path):
        print(f"错误：找不到记录文件：{json_path}")
        sys.exit(2)
    if not os.path.exists(EXCEL_PATH):
        print(f"错误：找不到标准色板文件：{EXCEL_PATH}")
        sys.exit(2)
    os.makedirs(OUT_DIR, exist_ok=True)

    # 加载数据
    colorbars = _parse_json_colorbars(json_path)
    std_cards = _load_standard_cards_from_excel(EXCEL_PATH, GROUP_SIZE)

    # 输出文件名（指定前缀）
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    excel_base = os.path.splitext(os.path.basename(EXCEL_PATH))[0]
    out_path = os.path.join(OUT_DIR, f"{excel_base}_{ts}.txt")

    lines: List[str] = []
    lines.append(f"Record File: {json_path}")
    lines.append(f"Standard Excel: {EXCEL_PATH}")
    lines.append(f"Standard Group Size: {GROUP_SIZE}, Total Cards: {len(std_cards)}")
    lines.append("")

    if not colorbars:
        lines.append("未在 JSON 中解析到任何检测色板。")
    else:
        for cb in colorbars:
            cb_id = cb["colorbar_id"]
            det_rgbs = cb["detected_rgbs"]
            best_idx, best_score, details = _find_best_card_for_colorbar(det_rgbs, std_cards)

            lines.append(f"[ColorBar] id={cb_id}")
            lines.append(f"  Best Standard Card Index: {best_idx + 1}  |  Detected Count: {len(det_rgbs)}  |  Total ΔE={best_score:.3f}")
            if details:
                lines.append("  Matches:")
                for i, d in enumerate(details, start=1):
                    lines.append(
                        f"    #{i:02d}: Detected {_fmt_rgb(d['det_rgb'])} | {_fmt_cmyk(d['det_cmyk'])} | {_fmt_lab(d['det_lab'])}"
                    )
                    lines.append(
                        f"          Standard {_fmt_rgb(d['std_rgb'])} | {_fmt_cmyk(d['std_cmyk'])} | {_fmt_lab(d['std_lab'])}  |  ΔE={d['de']:.3f}"
                    )
            lines.append("")

    # 写出文件并同步到控制台
    content = "\n".join(lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"已生成报告：{out_path}")
    print(content)


if __name__ == "__main__":
    main()