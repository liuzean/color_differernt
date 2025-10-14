# -*- coding: utf-8 -*-
"""
批量校验记录文件(JSON/TXT)中的色彩值与ΔE：
- 输入：一个记录文件路径（.json 或 .txt）
- 对每个色块：
  * 读取 Detected RGB 与 Standard CMYK
  * CMYK(0-100)->/100->(1,1,4)->cmyk_to_rgb，构造 1x1 RGB 图像
  * 使用 calculate_color_difference 计算 ΔE
  * 使用 rgb_to_lab 计算双方 LAB
  * 与记录文件中的值比较（允许误差±1），统计一致/不一致数量
运行(Windows PowerShell):
  python tests/validate_color_records.py "Result Output/analysis_XXXX.json"
  python tests/validate_color_records.py "Result Output/analysis_XXXX.txt"
"""

import os
import re
import sys
import json
import math
import numpy as np

# 将项目根目录加入 sys.path，确保可导入 core.*
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.color.utils import cmyk_to_rgb, rgb_to_lab, calculate_color_difference


def _rgb_tuple_to_img(rgb_255):
    r, g, b = [int(x) for x in rgb_255]
    return np.array([[[r, g, b]]], dtype=np.uint8)  # (1,1,3)


def _cmyk_percent_to_rgb_img(cmyk_percent):
    cmyk_norm = np.array([float(c) / 100.0 for c in cmyk_percent], dtype=float).reshape(1, 1, 4)
    rgb_arr = cmyk_to_rgb(cmyk_norm)  # (1,1,3) 期望 uint8/float同尺度转为uint8使用
    rgb = np.array(rgb_arr[0, 0], dtype=np.float32)
    rgb_uint8 = np.clip(np.round(rgb), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    return rgb_uint8


def _rgb_img_to_lab_tuple(rgb_img_1x1):
    lab_arr = rgb_to_lab(rgb_img_1x1)  # -> (1,1,3)
    lab = np.array(lab_arr[0, 0], dtype=float).reshape(3,)
    return tuple(float(v) for v in lab)


def _within_tol(a, b, tol=1.0):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def _seq_within_tol(seq_a, seq_b, tol=1.0):
    try:
        a = [float(x) for x in seq_a][:3]
        b = [float(x) for x in seq_b][:3]
        if len(a) != len(b):
            return False
        return all(_within_tol(x, y, tol) for x, y in zip(a, b))
    except Exception:
        return False


def _parse_json_records(path):
    """返回 items 列表：每项包含 det_rgb, det_lab(rec), std_cmyk, std_rgb(rec), std_lab(rec), delta_e(rec)"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items_out = []
    cbs = data.get("colorbar_results") or []
    # 兼容字段名：pure_color_analyses / color_analysis
    for cb in cbs:
        entries = cb.get("pure_color_analyses") or cb.get("color_analysis") or []
        for it in entries:
            det_rgb = it.get("pure_color_rgb") or (it.get("detected") or {}).get("rgb")
            det_lab_rec = it.get("detected_lab") or (it.get("detected") or {}).get("lab")
            gt = (it.get("ground_truth_match") or {})
            std = (gt.get("closest_color") or it.get("standard") or {})
            std_cmyk = std.get("cmyk")
            std_rgb_rec = std.get("rgb")
            std_lab_rec = std.get("lab")
            de_rec = gt.get("delta_e") or it.get("delta_e") or it.get("de")

            if det_rgb is None or std_cmyk is None:
                continue

            items_out.append({
                "det_rgb": det_rgb,
                "det_lab_rec": det_lab_rec,
                "std_cmyk": std_cmyk,
                "std_rgb_rec": std_rgb_rec,
                "std_lab_rec": std_lab_rec,
                "delta_e_rec": de_rec,
            })
    return items_out


_NUM_TUP_RE = re.compile(r"\(([^)]+)\)")
def _parse_tuple(line):
    m = _NUM_TUP_RE.search(line)
    if not m:
        return None
    parts = [p.strip() for p in m.group(1).split(",")]
    vals = []
    for p in parts:
        if p == "":
            continue
        try:
            vals.append(int(p))
            continue
        except Exception:
            pass
        try:
            vals.append(float(p))
        except Exception:
            pass
    return vals if vals else None


def _parse_txt_records(path):
    """解析 TXT 文件为 items 列表：字段同 _parse_json_records 返回的结构"""
    items_out = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln == "Detected":
            det_rgb = det_cmyk = det_lab = None
            std_rgb = std_cmyk = std_lab = None
            # 期望三行 Detected 明细
            j = i + 1
            while j < len(lines) and lines[j].startswith(("RGB:", "CMYK:", "LAB:", "RGB", "CMYK", "LAB", "  ")):
                t = lines[j].lstrip()
                if t.startswith("RGB:"):
                    det_rgb = _parse_tuple(t)
                elif t.startswith("CMYK:"):
                    det_cmyk = _parse_tuple(t)  # 未用，来源记录
                elif t.startswith("LAB:"):
                    det_lab = _parse_tuple(t)
                else:
                    break
                j += 1
            # 下一行应为 Standard
            if j < len(lines) and lines[j] == "Standard":
                j += 1
                while j < len(lines) and lines[j].startswith(("RGB:", "CMYK:", "LAB:", "  ")):
                    t = lines[j].lstrip()
                    if t.startswith("RGB:"):
                        std_rgb = _parse_tuple(t)
                    elif t.startswith("CMYK:"):
                        std_cmyk = _parse_tuple(t)
                    elif t.startswith("LAB:"):
                        std_lab = _parse_tuple(t)
                    else:
                        break
                    j += 1
            # ΔE 行
            de_rec = None
            if j < len(lines) and lines[j].startswith("ΔE:"):
                try:
                    de_rec = float(lines[j].split("ΔE:")[1].strip().split()[0])
                except Exception:
                    de_rec = None
                j += 1

            if det_rgb is not None and std_cmyk is not None:
                items_out.append({
                    "det_rgb": det_rgb,
                    "det_lab_rec": det_lab,
                    "std_cmyk": std_cmyk,
                    "std_rgb_rec": std_rgb,
                    "std_lab_rec": std_lab,
                    "delta_e_rec": de_rec,
                })

            i = j
            continue
        i += 1
    return items_out


def _compute_from_record(item):
    det_img = _rgb_tuple_to_img(item["det_rgb"])
    std_img = _cmyk_percent_to_rgb_img(item["std_cmyk"])
    # ΔE
    avg_de, _ = calculate_color_difference(det_img, std_img)
    # LAB
    det_lab = _rgb_img_to_lab_tuple(det_img)
    std_lab = _rgb_img_to_lab_tuple(std_img)
    # 也返回 std_rgb(计算得到)
    std_rgb = tuple(int(x) for x in std_img.reshape(3,))
    return float(avg_de), det_lab, std_lab, std_rgb


def main():
    if len(sys.argv) < 2:
        print("用法: python tests/validate_color_records.py <记录文件路径(.json 或 .txt)>")
        sys.exit(2)

    rec_path = sys.argv[1]
    if not os.path.isfile(rec_path):
        print(f"错误：文件不存在: {rec_path}")
        sys.exit(2)

    ext = os.path.splitext(rec_path)[1].lower()
    if ext == ".json":
        records = _parse_json_records(rec_path)
        source = "JSON"
    else:
        records = _parse_txt_records(rec_path)
        source = "TXT"

    total = len(records)
    if total == 0:
        print(f"{source} 中未解析到任何色块记录。")
        sys.exit(0)

    # 统计
    match_det_lab = 0
    match_std_lab = 0
    match_std_rgb = 0
    match_de = 0

    for idx, it in enumerate(records, 1):
        de_calc, det_lab_calc, std_lab_calc, std_rgb_calc = _compute_from_record(it)

        # 比较
        if it.get("det_lab_rec") is not None and _seq_within_tol(det_lab_calc, it["det_lab_rec"], tol=1.0):
            match_det_lab += 1
        if it.get("std_lab_rec") is not None and _seq_within_tol(std_lab_calc, it["std_lab_rec"], tol=1.0):
            match_std_lab += 1
        if it.get("std_rgb_rec") is not None and _seq_within_tol(std_rgb_calc, it["std_rgb_rec"], tol=1.0):
            match_std_rgb += 1
        if it.get("delta_e_rec") is not None and _within_tol(de_calc, it["delta_e_rec"], tol=1.0):
            match_de += 1

    print(f"记录来源: {source}")
    print(f"总色块数: {total}")
    print(f"检测LAB一致(±1): {match_det_lab}/{total}")
    print(f"标准LAB一致(±1): {match_std_lab}/{total}")
    print(f"标准RGB一致(±1): {match_std_rgb}/{total}")
    print(f"ΔE一致(±1): {match_de}/{total}")


if __name__ == "__main__":
    main()