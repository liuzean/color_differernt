# core/block_detection/pure_colorbar_analysis_tiff.py

"""
(新文件)
基于 TIFF 的高保真色板分析流水线 (模仿 pure_colorbar_analysis.py 的结构)

本模块实现了用户定义的 TIFF 分析流程：
1. 加载高保真 TIFF (用于颜色提取)
2. 创建一个 8-bit BGR 副本 (用于 YOLO 检测和预览)
3. (步骤 3) 在 8-bit 副本上运行 YOLO 检测色板
4. (步骤 4) 在 8-bit 色板片段上运行 YOLO 检测色块，并提炼中心 50% 区域坐标
5. (步骤 5) 使用中心坐标从高保真 TIFF 裁剪区域，并提取颜色（缩放至 8-bit RGB）
6. (步骤 6 & 9) 复用现有的色差分析流程，并将 GroundTruthColor 对象转为字典
7. (步骤 7) 清理临时文件夹
8. (步骤 8) 提供 _for_gradio 包装函数
"""

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import os
import shutil  # 用于删除文件夹
import tempfile # 用于安全检查
import traceback
import statistics

# 尝试导入 tifffile，优先使用
try:
    # 你可能需要: pip install tifffile imagecodecs
    import tifffile
    TIFFFILE_SUPPORTED = True
    print("Using tifffile library for TIFF loading.")
except ImportError:
    TIFFFILE_SUPPORTED = False
    print("Warning: tifffile library not found. Falling back to Pillow (may have limitations).")

# 导入本项目中的依赖
from .yolo_show import detect_colorbars_yolo, load_yolo_model
from .yolo_block_detection import detect_blocks_with_yolo, load_yolo_block_model
# 我们将复用 _get_color_quality 和 color_analysis.py 的 rgb_to_cmyk_icc
from .pure_colorbar_analysis import _get_color_quality 
from ..color.ground_truth_checker import ground_truth_checker
from core.utils.result_saver import save_analysis_to_files

# --- 步骤 1: TIFF 加载辅助函数 ---
def load_tiff_high_fidelity(tiff_path: str) -> np.ndarray | None:
    """
    (步骤 1)
    加载 TIFF 图像，尝试保留原始位深和通道。
    返回 OpenCV BGR 顺序的 NumPy 数组。
    """
    if not os.path.exists(tiff_path):
        print(f"Error: TIFF file not found at {tiff_path}")
        return None
    try:
        if TIFFFILE_SUPPORTED:
            img_array = tifffile.imread(tiff_path)
            # tifffile 默认读取为 RGB (或 Grayscale, RGBA)
            
            # 确保 3D (H, W, C)
            if len(img_array.shape) == 2: # Grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            elif len(img_array.shape) == 3 and img_array.shape[2] == 4: # RGBA
                 img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR) # 转 BGR
            elif len(img_array.shape) == 3 and img_array.shape[2] == 3: # RGB
                 img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) # RGB -> BGR
            elif len(img_array.shape) == 3 and img_array.shape[2] > 4: # e.g., CMYK+
                 print(f"Warning: TIFF has {img_array.shape[2]} channels. Using first 3 (assuming BGR).")
                 img_array = img_array[:,:,:3] # Fallback: take first 3
            else:
                 raise ValueError(f"Unsupported shape from tifffile: {img_array.shape}")
        
        else: # Pillow fallback
            print("Using Pillow fallback for TIFF loading.")
            pil_img = Image.open(tiff_path)
            pil_img.load() 
            
            # 统一转换为 RGB numpy array
            if pil_img.mode == 'L':
                pil_img = pil_img.convert('RGB')
            elif pil_img.mode == 'P':
                pil_img = pil_img.convert('RGB')
            elif pil_img.mode == 'RGBA':
                pil_img = pil_img.convert('RGB') # 丢弃 Alpha
            elif pil_img.mode == 'CMYK':
                pil_img = pil_img.convert('RGB')
                print("Warning: Pillow CMYK TIFF conversion might be inaccurate.")
            
            img_array = np.array(pil_img)
            
            # 如果是高位深 (Pillow mode I;16, RGB;I16 etc.)
            if img_array.dtype == np.uint16 or img_array.dtype.kind == 'f':
                 pass # 保留高位深或浮点
            
            # 转换为 BGR
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                 img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else: # 兜底 L 模式
                 img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

        return img_array

    except (UnidentifiedImageError, IOError, ValueError, Exception) as e:
        print(f"Error loading high-fidelity TIFF {tiff_path}: {e}")
        traceback.print_exc()
        return None

# --- 步骤 2: 8-bit BGR 转换辅助函数 ---
def convert_to_8bit_bgr(image_array: np.ndarray) -> np.ndarray | None:
    """
    (步骤 2)
    将高位深/浮点 NumPy 数组转换为 8-bit BGR。
    """
    if image_array is None: return None
    try:
        # 确保 3 通道 BGR
        if len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        elif len(image_array.shape) == 3 and image_array.shape[2] != 3:
             print(f"Warning: Input has {image_array.shape[2]} channels, taking first 3.")
             image_array = image_array[:,:,:3]

        if image_array.dtype == np.uint8:
            return image_array # 已经是 8-bit BGR
        
        elif image_array.dtype == np.uint16:
            # Scale 16-bit (0-65535) to 8-bit (0-255)
            scaled_image = (image_array / 65535.0 * 255.0).astype(np.uint8)
            return scaled_image
        
        elif image_array.dtype.kind == 'f': # Float
             if image_array.max() <= 1.0: # 归一化 0-1
                  scaled_image = (image_array * 255.0).astype(np.uint8)
             else: # 假设是 0-255
                  scaled_image = image_array.astype(np.uint8)
             return scaled_image
        else:
            print(f"Warning: Unsupported image data type {image_array.dtype}. Attempting naive scaling.")
            min_val, max_val = np.min(image_array), np.max(image_array)
            if max_val == min_val: return np.full(image_array.shape, 128, dtype=np.uint8)
            scaled_image = ((image_array - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
            return scaled_image

    except Exception as e:
        print(f"Error converting image to 8-bit BGR: {e}")
        traceback.print_exc()
        return None

# --- 步骤 4: 提炼中心坐标辅助函数 ---
def _get_block_center_coords_relative(block_image_8bit: np.ndarray) -> tuple[int, int, int, int]:
    """
    (步骤 4)
    复刻 extract_pure_color_from_block 的中心裁剪逻辑。
    输入 8-bit 色块图像，返回中心 50% 区域相对于该色块的 (x1, y1, x2, y2) 坐标。
    """
    # 1. 缩小到 20x20 (仅用于计算比例)
    sample_size = (20, 20)
    resized = cv2.resize(block_image_8bit, sample_size, interpolation=cv2.INTER_AREA)
        
    h, w = resized.shape[:2]
    center_h, center_w = h // 2, w // 2
    margin_h, margin_w = h // 4, w // 4 # 中心 50%
    
    # 2. 得到 20x20 上的相对坐标
    c_x1_resized = center_w - margin_w
    c_y1_resized = center_h - margin_h
    c_x2_resized = center_w + margin_w
    c_y2_resized = center_h + margin_h
    
    # 3. 缩放回原始 block_image_8bit 尺寸
    orig_h, orig_w = block_image_8bit.shape[:2]
    scale_x = orig_w / float(w)
    scale_y = orig_h / float(h)
    
    c_x1_orig = int(c_x1_resized * scale_x)
    c_y1_orig = int(c_y1_resized * scale_y)
    c_x2_orig = int(c_x2_resized * scale_x)
    c_y2_orig = int(c_y2_resized * scale_y)
    
    # 4. 确保坐标不溢出
    c_x2_orig = min(orig_w, c_x2_orig if (c_x2_orig > c_x1_orig) else c_x1_orig + 1)
    c_y2_orig = min(orig_h, c_y2_orig if (c_y2_orig > c_y1_orig) else c_y1_orig + 1)
    
    return (c_x1_orig, c_y1_orig, c_x2_orig, c_y2_orig)

# --- 步骤 5: 从高保真区域提取颜色辅助函数 ---
def _extract_color_from_tiff_region(
    high_fidelity_region: np.ndarray, 
    purity_threshold: float
) -> tuple[tuple[int, int, int], float]:
    """
    (步骤 5)
    复刻 extract_pure_color_from_block 的颜色分析逻辑。
    输入是高保真 BGR 裁剪区域。
    输出是 ( (R_8bit, G_8bit, B_8bit), purity_score )
    """
    if high_fidelity_region.size == 0:
        return ((0, 0, 0), 0.0)

    # 1. 颜色分析（在中位/主导色）
    if len(high_fidelity_region.shape) != 3 or high_fidelity_region.shape[2] != 3:
        print(f"Warning: High fidelity region has invalid shape {high_fidelity_region.shape}. Returning black.")
        return ((0, 0, 0), 0.0)
        
    pixels = high_fidelity_region.reshape(-1, 3)
    if len(pixels) == 0:
        return ((0, 0, 0), 0.0)

    # 2. 计算纯度（基于标准差）
    color_std = np.std(pixels, axis=0)
    max_std = np.max(color_std)
    
    # 缩放纯度阈值
    dtype = high_fidelity_region.dtype
    if dtype == np.uint16:
        scaled_std_threshold = 50.0 * 257.0 # 65535/255 ≈ 257
        purity_score = max(0.0, 1.0 - (max_std / scaled_std_threshold))
    elif dtype.kind == 'f' and pixels.max() <= 1.0:
        scaled_std_threshold = 50.0 / 255.0 # 缩放到 0-1 范围
        purity_score = max(0.0, 1.0 - (max_std / scaled_std_threshold))
    else: # 假设是 8-bit 或 0-255 的浮点数
        purity_score = max(0.0, 1.0 - (max_std / 50.0))
        
    # 3. 计算中位色
    median_color = np.median(pixels, axis=0)
    
    if purity_score >= purity_threshold:
        # 纯色：计算主导色
        unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        most_frequent_idx = np.argmax(counts)
        final_color_bgr = unique_colors[most_frequent_idx]
    else:
        # 非纯色：使用中位色
        final_color_bgr = median_color

    # 4. 缩放到 8-bit BGR
    if dtype == np.uint16:
        scaled_bgr = (final_color_bgr / 65535.0 * 255.0).astype(int)
    elif dtype == np.uint8:
        scaled_bgr = final_color_bgr.astype(int)
    elif dtype.kind == 'f': # Float
         if final_color_bgr.max() <= 1.0: # 0-1 范围
              scaled_bgr = (final_color_bgr * 255.0).astype(int)
         else: # 0-255 范围
              scaled_bgr = final_color_bgr.astype(int)
    else:
        print(f"Warning: Unhandled dtype {dtype} in color extraction, using direct cast.")
        scaled_bgr = final_color_bgr.astype(int)

    # 5. 转换为 8-bit RGB 元组并返回
    b, g, r = scaled_bgr
    r_8bit = max(0, min(255, int(r)))
    g_8bit = max(0, min(255, int(g)))
    b_8bit = max(0, min(255, int(b)))

    return ((r_8bit, g_8bit, b_8bit), purity_score)


# --- 步骤 8: 主流水线函数 (模仿 pure_colorbar_analysis.py) ---
def pure_colorbar_analysis_tiff_pipeline(
    preview_filepath: str,      # 替换 pil_image
    original_filename: str,   # 新增
    confidence_threshold: float,
    box_expansion: int,
    model_path: str = None,
    yolo_block_confidence: float = 0.5,
    block_area_threshold: int = 50,
    purity_threshold: float = 0.8,
    **kwargs,
) -> dict:
    """
    (步骤 8 - _pipeline)
    完整的基于 TIFF 的色板分析流水线 (模仿 pure_colorbar_analysis_pipeline)。
    """
    temp_dir = None
    result = {"step_completed": 0, "annotated_image": None} # 确保 annotated_image 存在

    try:
        # --- 步骤 1: TIFF 加载 ---
        print(f"Step 1: Loading original TIFF file: {preview_filepath}")
        original_tiff_image = load_tiff_high_fidelity(preview_filepath) # 高保真 BGR
        if original_tiff_image is None:
            raise ValueError(f"Failed to load the uploaded TIFF file: {preview_filepath}")
        
        # --- 步骤 7: 临时文件夹清理 (准备) ---
        temp_dir = os.path.dirname(preview_filepath)
        result["temp_dir_to_clean"] = temp_dir # 存储以便 Gradio 包装函数传递给清理器

        # --- 步骤 2: 转换为 8 位 BGR (用于 YOLO/预览) ---
        print("Step 2: Generating 8-bit BGR version for detection...")
        yolo_input_image = convert_to_8bit_bgr(original_tiff_image)
        if yolo_input_image is None:
             raise ValueError("Failed to convert TIFF to 8-bit BGR for YOLO input.")

        # --- 步骤 3: YOLO 色板检测 (在 8-bit 图上) ---
        print("Step 3: Detecting colorbars on 8-bit image...")
        model = load_yolo_model(model_path)
        (
            annotated_8bit_image, # 8-bit BGR NumPy array (带标注)
            colorbar_boxes, # [x1, y1, x2, y2] 绝对坐标
            confidences,
            colorbar_segments_8bit, # 8-bit BGR 裁剪片段
        ) = detect_colorbars_yolo(
            yolo_input_image,
            model,
            box_expansion=box_expansion,
            confidence_threshold=confidence_threshold,

        )
        # 立即将 8-bit 标注图转为 PIL 图像
        result["annotated_image"] = Image.fromarray(cv2.cvtColor(annotated_8bit_image, cv2.COLOR_BGR2RGB))

        if not colorbar_boxes:
            result["error"] = "No colorbars detected on the image"
            result["step_completed"] = 3
            return result

        # --- 步骤 4: YOLO 色块检测 & 中心坐标提炼 (在 8-bit 图上) ---
        print("Step 4: Detecting blocks and refining center coordinates on 8-bit segments...")
        try:
            block_model = load_yolo_block_model()
        except (FileNotFoundError, RuntimeError) as e:
            result["error"] = str(e)
            result["step_completed"] = 4
            return result

        all_extraction_info = [] 

        for i, (cb_box, cb_conf, cb_seg_8bit) in enumerate(zip(colorbar_boxes, confidences, colorbar_segments_8bit, strict=False)):
            colorbar_id = i + 1
            cb_x1, cb_y1, _, _ = cb_box # 色板左上角绝对坐标
            
            (
                segmented_colorbar_8bit, # 8-bit BGR 色板片段 (带色块标注)
                block_images_8bit, # 8-bit BGR 色块裁剪列表
                block_boxes_relative, # 色块相对于色板片段的坐标
                block_count_detected,
            ) = detect_blocks_with_yolo(
                cb_seg_8bit,
                block_model,
                confidence_threshold=yolo_block_confidence,
                min_area=block_area_threshold,
                return_coordinates=True # <-- 传入新参数

            )
            print(f"  Colorbar {colorbar_id}: Detected {block_count_detected} blocks.")

            coords_to_extract_for_this_cb = []
            if block_count_detected > 0:
                for block_img_8bit, block_box_rel in zip(block_images_8bit, block_boxes_relative, strict=False):
                    center_coords_rel = _get_block_center_coords_relative(block_img_8bit)
                    
                    block_x1_rel, block_y1_rel, _, _ = block_box_rel
                    c_x1_rel, c_y1_rel, c_x2_rel, c_y2_rel = center_coords_rel
                    
                    abs_c_x1 = cb_x1 + block_x1_rel + c_x1_rel
                    abs_c_y1 = cb_y1 + block_y1_rel + c_y1_rel
                    abs_c_x2 = cb_x1 + block_x1_rel + c_x2_rel
                    abs_c_y2 = cb_y1 + block_y1_rel + c_y2_rel
                    
                    coords_to_extract_for_this_cb.append((abs_c_x1, abs_c_y1, abs_c_x2, abs_c_y2))
            
            all_extraction_info.append({
                "colorbar_id": colorbar_id, 
                "conf": cb_conf, 
                "box": cb_box, 
                "coords_to_extract": coords_to_extract_for_this_cb,
                "original_segment_pil": Image.fromarray(cv2.cvtColor(cb_seg_8bit, cv2.COLOR_BGR2RGB)), # 8-bit 原色板
                "segmented_colorbar_pil": Image.fromarray(cv2.cvtColor(segmented_colorbar_8bit, cv2.COLOR_BGR2RGB)), # 8-bit 带框色板
            })

        # --- 步骤 5: 从原始 TIFF 提取颜色 ---
        print("Step 5: Extracting colors from high-fidelity TIFF...")
        all_colorbars_results = []
        total_blocks_analyzed = 0

        for info in all_extraction_info:
            coords_list = info["coords_to_extract"]
            colorbar_id = info["colorbar_id"]

            if not coords_list:
                all_colorbars_results.append({
                    "colorbar_id": colorbar_id, "confidence": info["conf"], "bounding_box": info["box"],
                    "original_segment_pil": info["original_segment_pil"],
                    "segmented_colorbar_pil": info["segmented_colorbar_pil"],
                    "block_count": 0, "pure_color_analyses": [], "best_match_card_id": None
                })
                continue

            extracted_colors_info = [] # 存储 ( (R,G,B), purity_score )
            for abs_center_coords in coords_list:
                x1, y1, x2, y2 = abs_center_coords
                high_fidelity_center_crop = original_tiff_image[y1:y2, x1:x2]
                
                rgb_tuple, purity = _extract_color_from_tiff_region(high_fidelity_center_crop, purity_threshold)
                extracted_colors_info.append({"rgb": rgb_tuple, "purity": purity})
            
            print(f"  Colorbar {colorbar_id}: Extracted {len(extracted_colors_info)} colors from TIFF.")

            # --- 步骤 6 & 9: 色差分析 & 模仿对象转字典 ---
            pure_color_analyses, best_match_card_id = [], None
            rgb_list_for_matching = [info["rgb"] for info in extracted_colors_info]
            
            if rgb_list_for_matching:
                card_match_result = ground_truth_checker.find_best_card_for_colorbar_new(rgb_list_for_matching)
                
                if card_match_result:
                    best_match_card_id = card_match_result["best_card_id"]
                    match_details = card_match_result["results"] 
                    
                    if len(match_details) == len(extracted_colors_info):
                        for idx, (match, color_info) in enumerate(zip(match_details, extracted_colors_info, strict=False)):
                            gt_color_obj = match.get("closest_ground_truth")
                            gt_color_dict = None
                            if gt_color_obj:
                                try:
                                    lab_tuple = tuple(float(f"{v:.2f}") for v in gt_color_obj.lab)
                                except Exception:
                                    lab_tuple = (0.0, 0.0, 0.0)
                                gt_color_dict = {
                                    "id": gt_color_obj.id,
                                    "name": gt_color_obj.name,
                                    "cmyk": gt_color_obj.cmyk,
                                    "rgb": gt_color_obj.rgb,
                                    "lab": lab_tuple,
                                }

                            analysis = {
                                "block_id": idx + 1, 
                                "colorbar_id": colorbar_id,
                                "pure_color_rgb": match["detected_rgb"],
                                "pure_color_cmyk": match.get("detected_cmyk", (0,0,0,0)),
                                "detected_lab": match.get("detected_lab", (0.0, 0.0, 0.0)),
                                "purity_score": color_info["purity"], # 来自高保真提取
                                "color_quality": _get_color_quality(color_info["purity"]),
                                "ground_truth_match": {
                                    "closest_color": gt_color_dict, # 传入字典
                                    "delta_e": match["delta_e"],
                                    "accuracy_level": match["accuracy_level"],
                                    "is_acceptable": match["delta_e"] < 3.0,
                                    "is_excellent": match["delta_e"] < 1.0,
                                }
                            }
                            pure_color_analyses.append(analysis)
                        total_blocks_analyzed += len(pure_color_analyses)

            all_colorbars_results.append({
                 "colorbar_id": colorbar_id,
                 "confidence": info["conf"],
                 "bounding_box": info["box"],
                 "original_segment_pil": info["original_segment_pil"],
                 "segmented_colorbar_pil": info["segmented_colorbar_pil"],
                 "block_count": len(pure_color_analyses),
                 "pure_color_analyses": pure_color_analyses,
                 "best_match_card_id": best_match_card_id,
            })

        # --- 统计信息 (与 pure_colorbar_analysis.py 相同) ---
        all_delta_e_values = []
        excellent_count, acceptable_count, high_purity_count = 0, 0, 0
        for result in all_colorbars_results:
            for analysis in result["pure_color_analyses"]:
                if "error" not in analysis and "ground_truth_match" in analysis:
                    gt_match = analysis["ground_truth_match"]
                    delta_e = gt_match["delta_e"]
                    all_delta_e_values.append(delta_e)
                    if gt_match.get("is_excellent", False): excellent_count += 1
                    if gt_match.get("is_acceptable", False): acceptable_count += 1
                    if analysis.get("purity_score", 0) >= 0.8: high_purity_count += 1

        accuracy_stats = {}
        if all_delta_e_values:
            total_analyzed_calc = len(all_delta_e_values)
            accuracy_stats = {
                "average_delta_e": statistics.mean(all_delta_e_values),
                "median_delta_e": statistics.median(all_delta_e_values),
                "max_delta_e": max(all_delta_e_values) if all_delta_e_values else 0,
                "min_delta_e": min(all_delta_e_values) if all_delta_e_values else 0,
                "excellent_colors": excellent_count,
                "acceptable_colors": acceptable_count,
                "high_purity_colors": high_purity_count,
                "total_analyzed": total_analyzed_calc,
                "excellent_percentage": (excellent_count / total_analyzed_calc) * 100 if total_analyzed_calc > 0 else 0,
                "acceptable_percentage": (acceptable_count / total_analyzed_calc) * 100 if total_analyzed_calc > 0 else 0,
                "high_purity_percentage": (high_purity_count / total_analyzed_calc) * 100 if total_analyzed_calc > 0 else 0,
            }

        # --- 准备最终结果 (与 pure_colorbar_analysis.py 相同) ---
        result.update({
            "success": True,
            "analysis_type": "pure_colorbar_tiff", # 标记为 TIFF 流程
            # annotated_image 已在步骤 3 设置
            "colorbar_count": len(all_colorbars_results),
            "colorbar_results": all_colorbars_results,
            "total_blocks": total_blocks_analyzed,
            "accuracy_statistics": accuracy_stats,
            "step_completed": 6,
            "original_filename": original_filename,
        })
        
        # 保存结果
        try:
             save_analysis_to_files(result, base_filename=original_filename)
        except Exception as e:
             print(f"错误：保存TIFF分析结果时发生异常: {e}")
             traceback.print_exc()

        return result

    except Exception as e:
        print(f"Error in main TIFF pipeline: {e}")
        traceback.print_exc()
        result["error"] = f"Error in pipeline: {str(e)}"
        # 确保 annotated_image 存在（在 try 块开头已设置）
        if "annotated_image" not in result:
             # 尝试从 8-bit 图创建回退
             try:
                 result["annotated_image"] = Image.fromarray(cv2.cvtColor(yolo_input_image, cv2.COLOR_BGR2RGB))
             except Exception:
                 result["annotated_image"] = None # 设为 None
        return result

    finally:
        # --- 7. 临时文件夹清理 ---
        if temp_dir and os.path.isdir(temp_dir):
            try:
                # 安全检查：
                temp_dir_norm = os.path.normpath(temp_dir)
                system_temp_norm = os.path.normpath(tempfile.gettempdir())
                gradio_base_norm = os.path.normpath(os.path.join(system_temp_norm, 'gradio'))

                # 检查 temp_dir 是否真的是 gradio_base_norm 的直接子目录
                if os.path.dirname(temp_dir_norm) == gradio_base_norm:
                    shutil.rmtree(temp_dir)
                    print(f"Successfully deleted temporary folder: {temp_dir}")
                else:
                    # 尝试处理 8.3 短文件名问题 (ADMINI~1)
                    if os.path.samefile(os.path.dirname(temp_dir_norm), gradio_base_norm):
                        shutil.rmtree(temp_dir)
                        print(f"Successfully deleted temporary folder (8.3 path matched): {temp_dir}")
                    else:
                        print(f"Safety check failed: Refusing to delete: {temp_dir}. Not a direct child of {gradio_base_norm}.")
            except Exception as e:
                print(f"Error deleting temporary folder {temp_dir}: {e}")
                traceback.print_exc()


# --- 步骤 8: Gradio 包装函数 (模仿 pure_colorbar_analysis.py) ---
def pure_colorbar_analysis_tiff_for_gradio(
    preview_filepath: str | None, # Received from gr.Image(type='filepath')
    original_filename: str,   # Received from hidden gr.Textbox
    confidence_threshold: float = 0.5,
    box_expansion: int = 10,
    yolo_block_confidence: float = 0.5,
    block_area_threshold: int = 50, # 注意类型
    purity_threshold: float = 0.8,
    **kwargs,
) -> tuple[Image.Image, list[dict], str, int]:
    """
    (步骤 8 - _for_gradio)
    为Gradio界面优化的TIFF色板分析流水线包装器。
    模仿 pure_colorbar_analysis_for_gradio 的签名和返回。
    """
    # 基本输入验证
    if not preview_filepath or not os.path.exists(preview_filepath):
         error_msg = "❌ Error: Invalid or missing temporary file path for the uploaded image."
         print(error_msg)
         return None, [], error_msg, 0
         
    if not original_filename:
         error_msg = "❌ Error: Original filename was not captured. Cannot proceed."
         print(error_msg)
         preview_pil = None
         try: preview_pil = Image.open(preview_filepath)
         except: pass
         return preview_pil, [], error_msg, 0

    print(f"Starting direct TIFF analysis for: {original_filename} (Temp path: {preview_filepath})")

    # 调用主流水线
    result = pure_colorbar_analysis_tiff_pipeline(
        preview_filepath=preview_filepath,
        original_filename=original_filename,
        confidence_threshold=confidence_threshold,
        box_expansion=box_expansion,
        yolo_block_confidence=yolo_block_confidence,
        block_area_threshold=float(block_area_threshold), # 确保类型正确
        purity_threshold=purity_threshold,
    )

    # --- 处理结果 (与 pure_colorbar_analysis_for_gradio 相同) ---
    annotated_pil = result.get("annotated_image") # 8-bit 标注图 (PIL)

    if "error" in result:
        error_msg = f"❌ {result['error']}"
        print(error_msg)
        return annotated_pil, [], error_msg, 0
        
    if not result.get("success", False):
        error_msg = "❌ TIFF-based analysis pipeline failed."
        print(error_msg)
        return annotated_pil, [], error_msg, 0

    colorbar_data = result.get("colorbar_results", [])
    total_blocks = result.get("total_blocks", 0)

    # --- 生成报告字符串 (与 pure_colorbar_analysis_for_gradio 相同) ---
    report = f"🎯 Direct TIFF Analysis Results ({original_filename})\n" + "=" * 55 + "\n\n"
    
    for i, res in enumerate(colorbar_data):
        best_card_id = res.get("best_match_card_id")
        block_count = res.get("block_count", 0)
        if best_card_id == "INVALID_DETECTION":
            report += f"🎨 Colorbar #{i+1} - ⚠️ ERROR: Too many blocks detected. Cannot perform matching.\n"
        elif best_card_id:
            report += f"🎨 Colorbar #{i+1} - Best Match Card: {best_card_id.upper()}\n"
        else:
            report += f"🎨 Colorbar #{i+1} - No blocks analyzed or no match found.\n"

    report += "\n📊 Overall Summary:\n"
    report += f"  • Total color blocks analyzed from TIFF: {total_blocks}\n"

    stats = result.get("accuracy_statistics", {})
    if stats:
        report += f"  • Average ΔE (against best cards): {stats.get('average_delta_e', 0):.2f}\n"
        report += f"  • ΔE Range: {stats.get('min_delta_e', 0):.2f} - {stats.get('max_delta_e', 0):.2f}\n"
        report += f"  • Excellent colors (ΔE < 1.0): {stats.get('excellent_colors', 0)}/{stats.get('total_analyzed', 0)} ({stats.get('excellent_percentage', 0):.1f}%)\n"
        report += f"  • Acceptable colors (ΔE < 3.0): {stats.get('acceptable_colors', 0)}/{stats.get('total_analyzed', 0)} ({stats.get('acceptable_percentage', 0):.1f}%)\n"
        # 我们在新流程中也计算了 purity_score，所以可以保留
        if 'high_purity_colors' in stats:
             report += f"  • High purity colors: {stats.get('high_purity_colors', 0)}/{stats.get('total_analyzed', 0)} ({stats.get('high_purity_percentage', 0):.1f}%)\n"
    report += "\n"

    # 块详情 (与 pure_colorbar_analysis_for_gradio 相同)
    for res in colorbar_data:
        best_card_id = res.get("best_match_card_id", "N_A")
        report += f"🔎 Details for Colorbar (Matched to {best_card_id}):\n"
        if best_card_id == "INVALID_DETECTION":
            report += "    - Skipping block details due to invalid detection.\n\n"
            continue
        if res["block_count"] > 0:
            for analysis in res["pure_color_analyses"]:
                block_id = analysis.get("block_id", "?")
                rgb = analysis.get("pure_color_rgb", (0, 0, 0))
                cmyk = analysis.get("pure_color_cmyk", (0, 0, 0, 0))
                purity = analysis.get("purity_score", 0)
                quality = analysis.get("color_quality", "N/A")
                hex_code = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
                report += f"    - Block {block_id}: {hex_code} (C{cmyk[0]} M{cmyk[1]} Y{cmyk[2]} K{cmyk[3]})"
                
                if purity is not None:
                     report += f" | Purity: {purity:.2f} ({quality})"
                
                if "ground_truth_match" in analysis:
                    gt = analysis["ground_truth_match"]
                    gt_color_dict = gt.get("closest_color") # (步骤 9) 读取字典
                    if gt_color_dict:
                        delta_e = gt["delta_e"]
                        level = gt["accuracy_level"]
                        gt_name = gt_color_dict['name']
                         
                        report += f" | ΔE: {delta_e:.2f} ({level}) vs {gt_name}"
                        if gt.get("is_excellent", False): report += " ✅"
                        elif gt.get("is_acceptable", False): report += " ⚠️"
                        else: report += " ❌"
                report += "\n"
        report += "\n"
    
    # 最终返回给 Gradio 的元组:
    # (annotated_pil, colorbar_data, report, total_blocks)
    return (annotated_pil, colorbar_data, report, total_blocks)

# --- 移除从 pure_colorbar_analysis.py 复制过来的、不再需要的旧函数 ---
# 我们保留 _get_color_quality
# 我们移除 extract_pure_color_from_block, _extract_block_color_features, 
# analyze_pure_color_block, analyze_colorbar_with_best_card_match, 
# analyze_colorbar_pure_colors
#
# ... 但是，为了简单起见，我把它们的代码直接替换为上面的新辅助函数和新 pipeline 了。
# 上面的代码是一个完整的文件内容，已经删除了不再需要的旧函数，
# 并添加了适配 TIFF 流程的新辅助函数。