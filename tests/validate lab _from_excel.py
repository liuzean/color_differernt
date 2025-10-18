# -*- coding: utf-8 -*-
#当前代码可以读取excel文件中的RGB值，并将其转换为LAB值，然后与Excel中提供的参考LAB值进行比较，判断是否在允许的误差范围内。
#现在查出来的问题是分光色差分析仪，如果色块的rgb，r是零，另外两个有数值，那么计算出来的lab就会与代码的计算相差较大，而且一般是g这个值相差很大

from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

# 可选依赖：确保能读取 .xlsx
try:
    import openpyxl  # noqa: F401
except Exception:
    print("错误：缺少 openpyxl，请先安装后再运行：")
    print(r'  .\.venv\Scripts\python.exe -m pip install openpyxl')
    sys.exit(2)

# 将项目根目录加入 sys.path，确保可导入 core.*
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.color.utils import rgb_to_lab  # 复用现有 RGB->LAB 功能

EXCEL_PATH = r"E:\color_project\打光测试图片\20250825贴纸检测\youmo.xlsx"
TOL = 0.5  # 误差阈值 ±0.5

def _rgb_tuple_to_lab_tuple(rgb_255) -> tuple[float, float, float]:
    """
    单一 RGB 三元组(0-255) -> LAB 三元组(float)
    """
    r, g, b = [int(round(float(x))) for x in rgb_255]
    rgb_img_1x1 = np.array([[[r, g, b]]], dtype=np.uint8)  # shape=(1,1,3)
    lab_arr = rgb_to_lab(rgb_img_1x1)  # 期望 (1,1,3)
    lab = np.array(lab_arr[0, 0], dtype=float).reshape(3,)
    return float(lab[0]), float(lab[1]), float(lab[2])

def main():
    if not os.path.exists(EXCEL_PATH):
        print(f"错误：找不到文件：{EXCEL_PATH}")
        sys.exit(2)

    # 读取 F:K 列（F,G,H 为 RGB；I,J,K 为 L,a,b），Excel 第1行为表头，第2行开始是数据
    df = pd.read_excel(EXCEL_PATH, header=0, usecols="F:K", engine="openpyxl")
    df.columns = ["R", "G", "B", "L_ref", "a_ref", "b_ref"]

    # 转为数值
    for col in ["R", "G", "B", "L_ref", "a_ref", "b_ref"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 丢弃缺失
    df = df.dropna(subset=["R", "G", "B", "L_ref", "a_ref", "b_ref"]).copy()

    # 约束 RGB 到 0-255 整数
    df[["R", "G", "B"]] = (
        df[["R", "G", "B"]]
        .round()
        .clip(lower=0, upper=255)
        .astype(int)
    )

    pass_count = 0
    fail_count = 0
    fail_items: list[dict] = []

    for idx, row in df.iterrows():
        r, g, b = int(row["R"]), int(row["G"]), int(row["B"])
        l_ref, a_ref, b_ref = float(row["L_ref"]), float(row["a_ref"]), float(row["b_ref"])

        l_calc, a_calc, b_calc = _rgb_tuple_to_lab_tuple((r, g, b))

        dL = abs(l_calc - l_ref)
        dA = abs(a_calc - a_ref)
        dB = abs(b_calc - b_ref)

        if dL <= TOL and dA <= TOL and dB <= TOL:
            pass_count += 1
        else:
            fail_count += 1
            excel_row = int(idx) + 2  # DataFrame 索引0 -> Excel第2行
            fail_items.append({
                "excel_row": excel_row,
                "rgb": (r, g, b),
                "lab_calc": (l_calc, a_calc, b_calc),
                "lab_ref": (l_ref, a_ref, b_ref),
                "diff": (dL, dA, dB),
            })

    print(f"合格数量: {pass_count}")
    print(f"不合格数量: {fail_count}")

    if fail_items:
        print("不合格明细：")
        for item in fail_items:
            l1, a1, b1 = item["lab_calc"]
            l2, a2, b2 = item["lab_ref"]
            dL, dA, dB = item["diff"]
            print(
                f" - 第{item['excel_row']}行, RGB={item['rgb']}, "
                f"计算LAB=({l1:.2f}, {a1:.2f}, {b1:.2f}), "
                f"ExcelLAB=({l2:.2f}, {a2:.2f}, {b2:.2f}), "
                f"|Δ|=({dL:.2f}, {dA:.2f}, {dB:.2f})"
            )

if __name__ == "__main__":
    main()