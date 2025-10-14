# core/utils/result_saver.py

"""
负责将颜色分析结果保存到文件的模块。
"""

import json
import os
from datetime import datetime
import numpy as np

OUTPUT_DIR = "Result Output"

def _convert_numpy_types(obj):
    """
    递归地将数据结构中的NumPy类型转换为Python原生类型，以便JSON序列化。
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy_types(list(obj)))
    return obj

def save_analysis_to_files(analysis_data: dict):
    """
    将完整的分析结果同时保存为 .json 和 .txt 文件。
    """
    if not analysis_data or "colorbar_results" not in analysis_data:
        print("警告：分析数据为空或格式不正确，跳过保存。")
        return

    valid_colorbar_results = [
        res for res in analysis_data.get("colorbar_results", [])
        if res.get("block_count", 0) <= 7
    ]

    if not valid_colorbar_results:
        print("警告：没有找到有效的色板（色块数 <= 7），跳过保存。")
        return

    filtered_analysis_data = analysis_data.copy()
    filtered_analysis_data["colorbar_results"] = valid_colorbar_results

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_filename = f"analysis_{timestamp}"
    json_path = os.path.join(OUTPUT_DIR, f"{base_filename}.json")
    txt_path = os.path.join(OUTPUT_DIR, f"{base_filename}.txt")

    try:
        clean_data = _prepare_data_for_json(filtered_analysis_data)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, indent=4, ensure_ascii=False)
        print(f"结果已成功保存到: {json_path}")
    except Exception as e:
        print(f"错误：保存JSON文件失败: {e}")

    try:
        txt_content = _format_data_for_txt(filtered_analysis_data)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(txt_content)
        print(f"结果已成功保存到: {txt_path}")
    except Exception as e:
        print(f"错误：保存TXT文件失败: {e}")

def _prepare_data_for_json(data: dict) -> dict:
    """将分析数据清洗为适合JSON存储的格式。"""
    import copy
    clean_data = copy.deepcopy(data)
    
    if "annotated_image" in clean_data:
        del clean_data["annotated_image"]
        
    for colorbar in clean_data.get("colorbar_results", []):
        if "original_segment_pil" in colorbar: del colorbar["original_segment_pil"]
        if "segmented_colorbar_pil" in colorbar: del colorbar["segmented_colorbar_pil"]
        if "color_blocks" in colorbar: del colorbar["color_blocks"]
            
        analyses = colorbar.get("pure_color_analyses", [])
        if not isinstance(analyses, list): continue

        for analysis in analyses:
            if "ground_truth_match" in analysis:
                match = analysis["ground_truth_match"]
                gt_obj_key = "closest_ground_truth" if "closest_ground_truth" in analysis["ground_truth_match"] else "closest_color"

                if gt_obj_key in match and hasattr(match[gt_obj_key], '__dict__'):
                    match[gt_obj_key] = match[gt_obj_key].__dict__

    return _convert_numpy_types(clean_data)


def _format_data_for_txt(data: dict) -> str:
    """将分析数据格式化为人类可读的TXT文件内容。"""
    parts = []
    
    for colorbar in data.get("colorbar_results", []):
        colorbar_id = colorbar.get("colorbar_id", "N/A")
        parts.append(f"Colorbar #{colorbar_id}")
        parts.append("=" * 30)

        analyses = colorbar.get("pure_color_analyses", [])
        if not analyses:
            parts.append("  No color blocks found.\n")
            continue

        for analysis in analyses:
            gt_match = analysis.get("ground_truth_match", {})
            
            detected_rgb = analysis.get("pure_color_rgb", "N/A")
            detected_cmyk = analysis.get("pure_color_cmyk", ("N/A",)*4)
            detected_lab = analysis.get("detected_lab")
            
            parts.append("Detected")
            parts.append(f"  RGB: {detected_rgb}")
            parts.append(f"  CMYK: {detected_cmyk}")
            if detected_lab is not None and len(detected_lab) >= 3:
                # 最终修正：使用f-string格式化确保一位小数
                lab_str = f"({detected_lab[0]:.1f}, {detected_lab[1]:.1f}, {detected_lab[2]:.1f})"
                parts.append(f"  LAB: {lab_str}")
            else:
                parts.append("  LAB: N/A")

            closest_color = gt_match.get("closest_color")
            if not closest_color and "closest_ground_truth" in gt_match:
                 closest_color = gt_match["closest_ground_truth"]

            parts.append("Standard")
            if closest_color:
                standard_rgb = getattr(closest_color, 'rgb', closest_color.get('rgb', 'N/A') if isinstance(closest_color, dict) else 'N/A')
                standard_cmyk = getattr(closest_color, 'cmyk', closest_color.get('cmyk', ('N/A',)*4) if isinstance(closest_color, dict) else ('N/A',)*4)
                standard_lab = getattr(closest_color, 'lab', closest_color.get('lab') if isinstance(closest_color, dict) else None)

                parts.append(f"  RGB: {standard_rgb}")
                parts.append(f"  CMYK: {standard_cmyk}")
                if standard_lab is not None and len(standard_lab) >= 3:
                    # 最终修正：使用f-string格式化确保一位小数
                    lab_str = f"({standard_lab[0]:.1f}, {standard_lab[1]:.1f}, {standard_lab[2]:.1f})"
                    parts.append(f"  LAB: {lab_str}")
                else:
                    parts.append("  LAB: N/A")
            else:
                 parts.append("  RGB: N/A\n  CMYK: N/A\n  LAB: N/A")
            
            delta_e = gt_match.get('delta_e', float('inf'))
            level = gt_match.get('accuracy_level', '')
            symbol = ""
            if level == "Excellent": symbol = "✅"
            elif level in ["Very Good", "Good"]: symbol = "⚠️"
            elif level in ["Fair", "Poor", "Very Poor"]: symbol = "❌"
                
            parts.append(f"ΔE: {delta_e:.2f} {symbol}\n")

    return "\n".join(parts)