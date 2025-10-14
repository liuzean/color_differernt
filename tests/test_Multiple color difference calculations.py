# -*- coding: utf-8 -*-
"""
批量校验记录文件中的色彩值与ΔE：
- 默认读取：Result Output/analysis_2025-10-14_15-26-12.json
- 也支持通过命令行参数传入 .json 或 .txt 文件路径
- 每个色块：
  * 读取 Detected RGB 与 Standard CMYK
  * CMYK(0-100)->/100->(1,1,4)->cmyk_to_rgb，构造 1x1 RGB 图像
  * 使用 calculate_color_difference 计算 ΔE
  * 使用 rgb_to_lab 计算双方 LAB
  * 与记录文件中的值比较（允许误差±1），统计一致/不一致数量
运行(Windows PowerShell):
  python tests/validate_color_records.py
  或
  python tests/validate_color_records.py "Result Output/analysis_2025-10-14_15-26-12.json"
"""

import os
import re
import sys
import json
import numpy as np

# 将项目根目录加入 sys.path，确保可导入 core.*
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.color.utils import cmyk_to_rgb, rgb_to_lab, calculate_color_difference

# 若想查看每条不一致详情，设为 True
PRINT_MISMATCHES = True


def _rgb_tuple_to_img(rgb_255):
    r, g, b = [int(x) for x in rgb_255[:3]]
    return np.array([[[r, g, b]]], dtype=np.uint8)  # (1,1,3)


def _cmyk_percent_to_rgb_img(cmyk_percent):
    cmyk_norm = np.array([float(c) / 100.0 for c in cmyk_percent[:4]], dtype=float).reshape(1, 1, 4)
    rgb_arr = cmyk_to_rgb(cmyk_norm)  # (1,1,3)
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


def _get_attr_or_key(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _pick_standard(item: dict):
    # 常见位置直接取
    for k in ("standard", "closest_color", "closest_ground_truth"):
        v = item.get(k)
        if v:
            return v
    # 也可能嵌套在 ground_truth_match 中
    gtm = item.get("ground_truth_match") or {}
    for k in ("closest_color", "closest_ground_truth", "standard"):
        v = gtm.get(k)
        if v:
            return v
    return None


def _parse_json_records(path):
    """返回 items 列表：每项包含 det_rgb, det_lab(rec), std_cmyk, std_rgb(rec), std_lab(rec), delta_e(rec)"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items_out = []
    cbs = data.get("colorbar_results") or []
    # 兼容字段名：pure_color_analyses / color_analysis / items / results
    for cb in cbs:
        entries = (
            cb.get("pure_color_analyses")
            or cb.get("color_analysis")
            or cb.get("items")
            or cb.get("results")
            or []
        )
        for it in entries:
            # Detected
            det_rgb = it.get("pure_color_rgb") or (it.get("detected") or {}).get("rgb")
            det_lab_rec = it.get("detected_lab") or (it.get("detected") or {}).get("lab")
            # Standard
            std = _pick_standard(it) or {}
            std_cmyk = _get_attr_or_key(std, "cmyk")
            std_rgb_rec = _get_attr_or_key(std, "rgb")
            std_lab_rec = _get_attr_or_key(std, "lab")
            # ΔE
            de_rec = (it.get("ground_truth_match") or {}).get("delta_e")
            if de_rec is None:
                de_rec = it.get("delta_e") or it.get("de")

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
            vals.append(int(p)); continue
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
        if ln.lower().startswith("colorbar #") or ln == "":
            i += 1
            continue

        # 兼容“Detected ... / Standard ...”块式输出
        if ln.startswith("Detected"):
            det_rgb = det_lab = None
            std_rgb = std_cmyk = std_lab = None
            j = i + 1
            while j < len(lines) and lines[j].startswith("Detected"):
                j += 1  # 跳过标题

            # 读取 Detected 明细
            while j < len(lines) and (lines[j].startswith("Detected") or lines[j].startswith("Standard") is False):
                t = lines[j].lstrip()
                if t.startswith("Detected RGB:"):
                    det_rgb = _parse_tuple(t)
                elif t.startswith("Detected CMYK"):
                    pass  # 记录中未用
                elif t.startswith("Detected LAB:"):
                    det_lab = _parse_tuple(t)
                else:
                    break
                j += 1

            # 读取 Standard 明细
            if j < len(lines) and lines[j].startswith("Standard"):
                j += 1
                while j < len(lines):
                    t = lines[j].lstrip()
                    if t.startswith("Standard RGB:"):
                        std_rgb = _parse_tuple(t)
                    elif t.startswith("Standard CMYK"):
                        std_cmyk = _parse_tuple(t)
                    elif t.startswith("Standard LAB:"):
                        std_lab = _parse_tuple(t)
                    else:
                        break
                    j += 1

            # 读取 ΔE 行
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
    # 标准 RGB(计算得到)
    std_rgb = tuple(int(x) for x in std_img.reshape(3,))
    return float(avg_de), det_lab, std_lab, std_rgb


def main():
    # 默认文件
    default_path = os.path.join(ROOT, "Result Output", "analysis_2025-10-14_15-26-12.json")
    # 若提供了参数则使用参数
    rec_path = sys.argv[1] if len(sys.argv) > 1 else default_path

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

    match_det_lab = 0
    match_std_lab = 0
    match_std_rgb = 0
    match_de = 0

    for idx, it in enumerate(records, 1):
        de_calc, det_lab_calc, std_lab_calc, std_rgb_calc = _compute_from_record(it)

        det_ok = it.get("det_lab_rec") is not None and _seq_within_tol(det_lab_calc, it["det_lab_rec"], tol=1.0)
        std_lab_ok = it.get("std_lab_rec") is not None and _seq_within_tol(std_lab_calc, it["std_lab_rec"], tol=1.0)
        std_rgb_ok = it.get("std_rgb_rec") is not None and _seq_within_tol(std_rgb_calc, it["std_rgb_rec"], tol=1.0)
        de_ok = it.get("delta_e_rec") is not None and _within_tol(de_calc, it["delta_e_rec"], tol=1.0)

        match_det_lab += int(bool(det_ok))
        match_std_lab += int(bool(std_lab_ok))
        match_std_rgb += int(bool(std_rgb_ok))
        match_de += int(bool(de_ok))

        if PRINT_MISMATCHES and not (det_ok and std_lab_ok and std_rgb_ok and de_ok):
            print(f"[不一致] 条目#{idx}:")
            if not det_ok and it.get("det_lab_rec") is not None:
                print(f"  Detected LAB 记录: {tuple(round(float(x),2) for x in it['det_lab_rec'][:3])}  计算: {tuple(round(x,2) for x in det_lab_calc)}")
            if not std_lab_ok and it.get("std_lab_rec") is not None:
                print(f"  Standard LAB 记录: {tuple(round(float(x),2) for x in it['std_lab_rec'][:3])}  计算: {tuple(round(x,2) for x in std_lab_calc)}")
            if not std_rgb_ok and it.get("std_rgb_rec") is not None:
                print(f"  Standard RGB 记录: {tuple(int(x) for x in it['std_rgb_rec'][:3])}  计算: {std_rgb_calc}")
            if not de_ok and it.get("delta_e_rec") is not None:
                print(f"  ΔE 记录: {round(float(it['delta_e_rec']),2)}  计算: {round(de_calc,2)}")

    print(f"记录来源: {source}")
    print(f"文件: {rec_path}")
    print(f"总色块数: {total}")
    print(f"检测LAB一致(±1): {match_det_lab}/{total}")
    print(f"标准LAB一致(±1): {match_std_lab}/{total}")
    print(f"标准RGB一致(±1): {match_std_rgb}/{total}")
    print(f"ΔE一致(±1): {match_de}/{total}")


if __name__ == "__main__":
    main()