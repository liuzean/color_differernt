# core/block_detection/colorbar_analysis_tiff.py

"""
Backend logic for Colorbar Analysis (TIFF-based).
Loads original TIFF, generates an 8-bit version for YOLO detection,
extracts colors from the original TIFF using detected coordinates,
and performs color difference analysis. Includes temp file cleanup.
"""

import os
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import traceback
import shutil # 用于删除文件夹
import tempfile # 用于创建临时文件（如果需要生成8位图）

# 从现有模块导入所需函数
from .yolo_show import detect_colorbars_yolo, load_yolo_model
from .yolo_block_detection import detect_blocks_with_yolo, load_yolo_block_model
# 注意：我们将直接在下面重新实现颜色提取逻辑以处理高位深TIFF，而不是完全依赖 pure_colorbar_analysis 中的提取
# 但仍然需要导入分析函数
from .pure_colorbar_analysis import analyze_colorbar_with_best_card_match, _get_color_quality
from ..color.ground_truth_checker import ground_truth_checker # 用于色差分析
from core.utils.result_saver import save_analysis_to_files # 结果保存
from ..color.utils import rgb_to_lab # 需要LAB转换

# 尝试导入 TIFF 支持库
try:
    # 注意：根据你的环境，可能需要安装 'tifffile' 或 'imagecodecs'
    # 'pip install tifffile imagecodecs'
    import tifffile
    TIFFFILE_SUPPORTED = True
    print("Using tifffile library for TIFF loading.")
except ImportError:
    TIFFFILE_SUPPORTED = False
    print("Warning: tifffile library not found. Falling back to Pillow for TIFF loading (might have limitations).")

# --- Helper Functions ---

def load_tiff_high_fidelity(tiff_path: str) -> np.ndarray | None:
    """
    Load a TIFF image, attempting to preserve original bit depth and channels.
    Uses tifffile (preferred) or Pillow.
    Returns NumPy array (channel order might vary initially) or None.
    """
    if not os.path.exists(tiff_path):
        print(f"Error: TIFF file not found at {tiff_path}")
        return None
    try:
        if TIFFFILE_SUPPORTED:
            # tifffile can often read various TIFF types directly into NumPy
            img_array = tifffile.imread(tiff_path)
            # Ensure we have at least 3 channels (e.g., convert grayscale to BGR)
            if len(img_array.shape) == 2: # Grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            elif len(img_array.shape) == 3 and img_array.shape[2] == 4: # RGBA/CMYKA?
                 # Try converting common 4-channel formats, default to dropping alpha
                 # This might need refinement based on your specific TIFF formats
                 try:
                      img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                 except cv2.error:
                      print(f"Warning: Could not convert 4-channel TIFF using RGBA2BGR. Using first 3 channels.")
                      img_array = img_array[:,:,:3] # Assume first 3 are relevant (e.g., RGBa)
            elif len(img_array.shape) == 3 and img_array.shape[2] > 4: # More channels?
                 print(f"Warning: TIFF has {img_array.shape[2]} channels. Using first 3.")
                 img_array = img_array[:,:,:3]
            # If shape is still not 3-channel BGR/RGB like, attempt Pillow fallback
            if not (len(img_array.shape) == 3 and img_array.shape[2] == 3):
                 print(f"Warning: tifffile loaded array with shape {img_array.shape}. Falling back to Pillow.")
                 raise ValueError("Unexpected shape from tifffile.")
            
            # tifffile might load as RGB, let's assume BGR needed later by OpenCV funcs
            # If color extraction is done directly on this array, order might not matter until conversion
            return img_array # Return as loaded, conversion to BGR later if needed by OpenCV
        else:
            # Pillow fallback
            pil_img = Image.open(tiff_path)
            # Pillow might load high bit depth images correctly
            # We convert to numpy *without* forcing 'RGB' yet to preserve mode/bit depth if possible
            img_array = np.array(pil_img)
             # Handle potential palette modes etc. by converting via RGB if necessary
            if pil_img.mode == 'P':
                 print("Info: TIFF has palette mode, converting via RGB.")
                 pil_img = pil_img.convert('RGB')
                 img_array = np.array(pil_img)
            elif pil_img.mode == 'L':
                 print("Info: TIFF is grayscale, converting to BGR.")
                 img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            elif pil_img.mode == 'RGBA':
                 print("Info: TIFF is RGBA, converting to BGR.")
                 img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            elif pil_img.mode == 'CMYK':
                 # Pillow's CMYK -> RGB is basic, prefer tifffile if possible
                 print("Warning: Pillow CMYK TIFF conversion might be inaccurate. Use tifffile if possible.")
                 pil_img = pil_img.convert('RGB')
                 img_array = np.array(pil_img)

            # Pillow usually gives RGB, convert to BGR for OpenCV consistency downstream?
            # Let's return as potentially RGB numpy array, handle BGR conversion where needed
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                 return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) # Convert to BGR here for consistency
            elif len(img_array.shape) == 2: # Grayscale was missed?
                 return cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            else:
                 raise ValueError(f"Unexpected image mode/shape from Pillow: {pil_img.mode}, {img_array.shape}")

    except (UnidentifiedImageError, IOError, ValueError, Exception) as e:
        print(f"Error loading high-fidelity TIFF {tiff_path}: {e}")
        traceback.print_exc()
        return None

def convert_to_8bit_bgr(image_array: np.ndarray) -> np.ndarray | None:
    """
    Convert a NumPy image array (potentially high bit depth) to 8-bit BGR.
    Handles potential scaling.
    """
    if image_array is None:
        return None
    try:
        # Check bit depth
        if image_array.dtype == np.uint8:
            # Already 8-bit, ensure it's BGR
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                return image_array # Assume it's already BGR from loading function
            elif len(image_array.shape) == 2:
                 return cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            else:
                 print(f"Warning: Input is 8-bit but unexpected shape {image_array.shape}. Cannot convert reliably.")
                 return None

        elif image_array.dtype == np.uint16:
            print("Info: Converting 16-bit image to 8-bit for YOLO/preview.")
            # Scale 16-bit (0-65535) to 8-bit (0-255)
            # Use floating point division then scale and convert type
            scaled_image = (image_array / 65535.0 * 255.0).astype(np.uint8)
            
            if len(scaled_image.shape) == 3 and scaled_image.shape[2] == 3:
                return scaled_image # Assume BGR from loading
            elif len(scaled_image.shape) == 2:
                 return cv2.cvtColor(scaled_image, cv2.COLOR_GRAY2BGR)
            else:
                 print(f"Warning: 16-bit image has unexpected shape {scaled_image.shape} after scaling.")
                 return None
        # Add handling for other potential bit depths (e.g., float) if necessary
        else:
            print(f"Warning: Unsupported image data type {image_array.dtype} for 8-bit conversion.")
            return None
            
    except Exception as e:
        print(f"Error converting image to 8-bit BGR: {e}")
        traceback.print_exc()
        return None

def extract_color_from_high_fidelity(image_array: np.ndarray, box: tuple[int, int, int, int], method="median") -> tuple | None:
    """
    Extract color from a specific region (box) of a high-fidelity image array.
    Returns RGB tuple (0-255 range for consistency with analysis functions).
    Handles potential high bit depth.
    """
    x1, y1, x2, y2 = box
    if image_array is None or image_array.size == 0 or y2 <= y1 or x2 <= x1:
        return None
        
    block = image_array[y1:y2, x1:x2]
    if block.size == 0:
        return None

    try:
        # Ensure block has 3 channels for median/mean calculation
        if len(block.shape) == 2: # Grayscale
            block = cv2.cvtColor(block, cv2.COLOR_GRAY2BGR)
        elif block.shape[2] != 3:
             print(f"Warning: Block has {block.shape[2]} channels, using first 3.")
             block = block[:,:,:3]

        # Calculate median or average color
        if method == "median":
            # Reshape, calculate median per channel
            pixels = block.reshape(-1, 3)
            primary_color_original_depth = np.median(pixels, axis=0)
        elif method == "average":
            primary_color_original_depth = np.mean(block, axis=(0, 1))
        else:
             raise ValueError("Invalid color extraction method")

        # Convert result to 8-bit RGB tuple (0-255)
        # Assuming input array was BGR order from loading/conversion
        b, g, r = primary_color_original_depth

        # Scale if necessary
        if image_array.dtype == np.uint16:
            r_8bit = int(r / 65535.0 * 255.0)
            g_8bit = int(g / 65535.0 * 255.0)
            b_8bit = int(b / 65535.0 * 255.0)
        elif image_array.dtype == np.uint8:
            r_8bit, g_8bit, b_8bit = int(r), int(g), int(b)
        # Add handling for other types if needed
        else:
             print(f"Warning: Unsupported data type {image_array.dtype} during color extraction scaling.")
             # Fallback: attempt direct conversion hoping it's within range
             r_8bit, g_8bit, b_8bit = int(r), int(g), int(b)
             
        # Clip values just in case
        r_8bit = max(0, min(255, r_8bit))
        g_8bit = max(0, min(255, g_8bit))
        b_8bit = max(0, min(255, b_8bit))

        return (r_8bit, g_8bit, b_8bit) # Return RGB tuple

    except Exception as e:
        print(f"Error extracting color from block: {e}")
        traceback.print_exc()
        return None


# --- Main Pipeline Function ---
def colorbar_analysis_tiff_pipeline(
    preview_filepath: str, # Received temp path of uploaded TIFF
    original_filename: str, # Received original filename via JS
    confidence_threshold: float = 0.5,
    box_expansion: int = 10,
    model_path: str = None, # For colorbar detection YOLO
    yolo_block_confidence: float = 0.5,
    block_area_threshold: int = 50,
    # purity_threshold: float = 0.8, # Purity might be less relevant for direct TIFF extraction
    **kwargs,
) -> dict:
    """
    Main pipeline: Load TIFF, generate 8-bit for YOLO, get coords, extract from original TIFF.
    """
    temp_dir = None # To store the directory path for cleanup
    result = {"step_completed": 0} # Initialize result dict

    try:
        # --- 1. Load Original High-Fidelity TIFF ---
        print(f"Step 1: Loading original TIFF file: {preview_filepath}")
        original_tiff_image = load_tiff_high_fidelity(preview_filepath)
        if original_tiff_image is None:
            raise ValueError(f"Failed to load the uploaded TIFF file: {preview_filepath}")
        
        # Store temp directory path for later cleanup
        temp_dir = os.path.dirname(preview_filepath)
        result["temp_dir_to_clean"] = temp_dir # Store for potential later use

        # --- 2. Generate 8-bit BGR version for YOLO and Preview ---
        print("Step 2: Generating 8-bit BGR version for detection...")
        yolo_input_image = convert_to_8bit_bgr(original_tiff_image)
        if yolo_input_image is None:
             raise ValueError("Failed to convert TIFF to 8-bit BGR for YOLO input.")

        # --- 3. Detect Colorbars using the 8-bit image ---
        print("Step 3: Detecting colorbars on 8-bit image with YOLO (best0710.pt)...")
        model = load_yolo_model(model_path)
        (
            annotated_8bit_image, # This will be the basis for the preview result
            colorbar_boxes, # [x1, y1, x2, y2] list - COORDS ARE VALID FOR BOTH 8-BIT and ORIGINAL TIFF
            confidences,
            # No segments needed
        ) = detect_colorbars_yolo(
            yolo_input_image,
            model,
            box_expansion=box_expansion,
            confidence_threshold=confidence_threshold,
            return_segments=False
        )
        result["annotated_image"] = Image.fromarray(cv2.cvtColor(annotated_8bit_image, cv2.COLOR_BGR2RGB)) # Store annotated 8-bit for preview

        if not colorbar_boxes:
            result["error"] = "No colorbars detected on the 8-bit image"
            result["step_completed"] = 3
            return result # Return early with annotated preview

        # --- 4. Detect Blocks (on 8-bit) & Extract Colors (from Original TIFF) ---
        print("Step 4: Detecting blocks on 8-bit image and extracting colors from original TIFF...")
        try:
            block_model = load_yolo_block_model()
        except (FileNotFoundError, RuntimeError) as e:
            result["error"] = str(e)
            result["step_completed"] = 4
            return result

        colorbar_results = []
        total_blocks_analyzed = 0

        for i, (box, confidence) in enumerate(zip(colorbar_boxes, confidences, strict=False)):
            colorbar_id = i + 1
            x1, y1, x2, y2 = box
            
            # a) Crop 8-bit segment for block detection
            yolo_colorbar_segment = yolo_input_image[y1:y2, x1:x2]
            if yolo_colorbar_segment.size == 0: continue

            # b) Detect block coordinates within the 8-bit segment
            (
                annotated_yolo_segment, # Can be discarded or used for detailed preview
                block_boxes_relative, # Relative coords [(bx1, by1, bx2, by2), ...]
                block_count_detected,
            ) = detect_blocks_with_yolo(
                yolo_colorbar_segment,
                block_model,
                confidence_threshold=yolo_block_confidence,
                min_area=block_area_threshold,
                return_absolute_coords=False
            )
            print(f"  Detected {block_count_detected} blocks in 8-bit segment {colorbar_id}.")

            if block_count_detected == 0:
                 # Still add a result entry, but mark as empty
                 colorbar_results.append({
                      "colorbar_id": colorbar_id, "confidence": confidence, "bounding_box": box,
                      "block_count": 0, "pure_color_analyses": [], "best_match_card_id": None,
                 })
                 continue

            # c) Extract colors from ORIGINAL TIFF using detected coordinates
            extracted_colors_rgb = [] # List of RGB tuples (0-255)
            block_abs_coords = [] # Store absolute coords for potential later use

            for block_idx, (bx1_rel, by1_rel, bx2_rel, by2_rel) in enumerate(block_boxes_relative):
                # Calculate absolute coordinates in the full image
                bx1_abs = x1 + bx1_rel
                by1_abs = y1 + by1_rel
                bx2_abs = x1 + bx2_rel
                by2_abs = y1 + by2_rel
                abs_coords = (bx1_abs, by1_abs, bx2_abs, by2_abs)
                block_abs_coords.append(abs_coords)

                # Extract color from the original high-fidelity TIFF
                color_rgb_8bit = extract_color_from_high_fidelity(original_tiff_image, abs_coords, method="median")
                
                if color_rgb_8bit:
                    extracted_colors_rgb.append(color_rgb_8bit)
                else:
                    print(f"Warning: Failed to extract color for block {block_idx+1} in colorbar {colorbar_id} from original TIFF.")
                    # Add a placeholder? Or skip? Let's skip for now.
                    # extracted_colors_rgb.append(None) # Option to keep alignment

            print(f"  Successfully extracted {len(extracted_colors_rgb)} colors from original TIFF for colorbar {colorbar_id}.")
            
            # Filter out None values if we used placeholders
            # valid_extracted_colors_rgb = [c for c in extracted_colors_rgb if c is not None]

            # d) Perform color analysis using the extracted 8-bit equivalent RGB values
            pure_color_analyses, best_match_card_id = [], None
            if extracted_colors_rgb: # Use the extracted list
                 # We need to adapt analyze_colorbar_with_best_card_match or recreate its logic
                 # It expects block images, not just colors. Let's recreate the core matching part.
                 
                 # 1. Find best card based on extracted RGBs
                 card_match_result = ground_truth_checker.find_best_card_for_colorbar_new(extracted_colors_rgb)
                 
                 if card_match_result:
                      best_match_card_id = card_match_result["best_card_id"]
                      match_details = card_match_result["results"] # Contains detected_rgb, detected_lab, closest_color, delta_e etc.
                      
                      # Now, structure these results similarly to pure_colorbar_analysis output
                      for idx, match in enumerate(match_details):
                           analysis = {
                               "block_id": idx + 1, # Simple sequential ID for now
                               "colorbar_id": colorbar_id,
                               "pure_color_rgb": match["detected_rgb"],
                               "pure_color_cmyk": match.get("detected_cmyk", (0,0,0,0)), # Should be calculated by find_best_card...
                               "detected_lab": match.get("detected_lab", (0.0, 0.0, 0.0)), # Should be calculated
                               # "purity_score": 1.0, # Assign fixed purity or remove? Let's assign 1.0
                               # "color_quality": "Excellent",
                               "ground_truth_match": {
                                   "closest_color": match.get("closest_ground_truth"), # This might be the dataclass object
                                   # We need the serializable dict version if using shared_results
                                   "closest_color_dict": ground_truth_checker._serialize_gt_color(match.get("closest_ground_truth")), # Need helper func
                                   "delta_e": match["delta_e"],
                                   "accuracy_level": match["accuracy_level"],
                                   "is_acceptable": match["delta_e"] < 3.0,
                                   "is_excellent": match["delta_e"] < 1.0,
                               }
                           }
                           pure_color_analyses.append(analysis)
                           total_blocks_analyzed += 1

            # Store results for this colorbar
            colorbar_results.append({
                 "colorbar_id": colorbar_id,
                 "confidence": confidence,
                 "bounding_box": box,
                 "block_count": len(pure_color_analyses), # Number of successfully analyzed blocks
                 "pure_color_analyses": pure_color_analyses,
                 "best_match_card_id": best_match_card_id,
                 # Optionally store block_abs_coords if needed later
            })

        # --- 5. Final Statistics ---
        # (Statistics logic remains similar, using data from pure_color_analyses)
        all_delta_e_values = []
        excellent_count, acceptable_count = 0, 0
        for res in colorbar_results:
             for analysis in res["pure_color_analyses"]:
                  if "ground_truth_match" in analysis:
                       gt_match = analysis["ground_truth_match"]
                       delta_e = gt_match["delta_e"]
                       all_delta_e_values.append(delta_e)
                       if gt_match.get("is_excellent", False): excellent_count += 1
                       if gt_match.get("is_acceptable", False): acceptable_count += 1
                       
        accuracy_stats = {}
        if all_delta_e_values:
            import statistics
            total_analyzed_calc = len(all_delta_e_values)
            accuracy_stats = {
                "average_delta_e": statistics.mean(all_delta_e_values),
                "median_delta_e": statistics.median(all_delta_e_values),
                "max_delta_e": max(all_delta_e_values),
                "min_delta_e": min(all_delta_e_values),
                "excellent_colors": excellent_count,
                "acceptable_colors": acceptable_count,
                # "high_purity_colors": N/A here, # Purity wasn't calculated
                "total_analyzed": total_analyzed_calc,
                "excellent_percentage": (excellent_count / total_analyzed_calc) * 100 if total_analyzed_calc > 0 else 0,
                "acceptable_percentage": (acceptable_count / total_analyzed_calc) * 100 if total_analyzed_calc > 0 else 0,
            }

        # --- 6. Prepare Final Result Dictionary ---
        result.update({
            "success": True,
            "analysis_type": "direct_tiff_upload",
            # annotated_image already set in step 3
            "colorbar_count": len(colorbar_results),
            "colorbar_results": colorbar_results,
            "total_blocks": total_blocks_analyzed,
            "accuracy_statistics": accuracy_stats,
            "step_completed": 5,
            "original_filename": original_filename, # Include original filename
        })
        
        # --- Save results before cleanup ---
        try:
             save_analysis_to_files(result, base_filename=original_filename) # Pass original name for saving
        except Exception as e:
             print(f"错误：保存TIFF分析结果时发生异常: {e}")
             traceback.print_exc() # Still continue to cleanup

        return result

    except Exception as e:
        print(f"Error in main TIFF pipeline: {e}")
        traceback.print_exc()
        # Ensure annotated_image is set if possible before returning error
        if "annotated_image" not in result and 'annotated_8bit_image' in locals():
             result["annotated_image"] = Image.fromarray(cv2.cvtColor(annotated_8bit_image, cv2.COLOR_BGR2RGB))
        elif "annotated_image" not in result:
             # Try to load the preview if not loaded before
             preview_pil = None
             if preview_filepath and os.path.exists(preview_filepath):
                  try: preview_pil = Image.open(preview_filepath)
                  except: pass
             result["annotated_image"] = preview_pil # Use original preview as fallback

        result["error"] = f"Error in pipeline: {str(e)}"
        return result

    finally:
        # --- 7. Cleanup Temporary Folder ---
        if temp_dir and os.path.isdir(temp_dir):
            try:
                # IMPORTANT SAFETY CHECK: Ensure we are deleting a gradio temp subfolder
                if os.path.basename(os.path.dirname(temp_dir)) == 'gradio' and \
                   os.path.basename(os.path.dirname(os.path.dirname(temp_dir))) == 'Temp':
                    shutil.rmtree(temp_dir)
                    print(f"Successfully deleted temporary folder: {temp_dir}")
                else:
                    print(f"Safety check failed: Refusing to delete non-standard temp folder: {temp_dir}")
            except Exception as e:
                print(f"Error deleting temporary folder {temp_dir}: {e}")
                traceback.print_exc()


# --- Gradio Interface Wrapper ---
def colorbar_analysis_tiff_for_gradio(
    preview_filepath: str | None, # Received from gr.Image(type='filepath')
    original_filename: str,   # Received from hidden gr.Textbox
    confidence_threshold: float = 0.5,
    box_expansion: int = 10,
    yolo_block_confidence: float = 0.5,
    block_area_threshold: int = 50,
    # purity_threshold: float = 0.8, # Not directly used for extraction
    **kwargs,
) -> tuple[Image.Image | None, list[dict], str, int]:
    """
    Wrapper for the TIFF pipeline optimized for Gradio interface.
    Returns: (annotated_preview_image, colorbar_data, report_string, total_blocks)
    """
    # Basic input validation
    if not preview_filepath or not os.path.exists(preview_filepath):
         error_msg = "❌ Error: Invalid or missing temporary file path for the uploaded image."
         print(error_msg)
         # Cannot generate annotated image without input path
         return None, [], error_msg, 0
         
    if not original_filename:
         error_msg = "❌ Error: Original filename was not captured. Cannot proceed."
         print(error_msg)
         # Try loading preview for display
         preview_pil = None
         try: preview_pil = Image.open(preview_filepath)
         except: pass
         return preview_pil, [], error_msg, 0

    print(f"Starting direct TIFF analysis for: {original_filename} (Temp path: {preview_filepath})")

    # Call the main pipeline
    result = colorbar_analysis_tiff_pipeline(
        preview_filepath=preview_filepath,
        original_filename=original_filename,
        confidence_threshold=confidence_threshold,
        box_expansion=box_expansion,
        yolo_block_confidence=yolo_block_confidence,
        block_area_threshold=block_area_threshold,
        # purity_threshold=purity_threshold, # Pass if needed by analysis function later
    )

    # --- Process results for Gradio ---
    annotated_pil = result.get("annotated_image") # Should be the 8-bit annotated preview PIL

    if "error" in result:
        error_msg = f"❌ {result['error']}"
        print(error_msg)
        # Return annotated_pil (might be None or fallback) and error
        return annotated_pil, [], error_msg, 0
        
    if not result.get("success", False):
        error_msg = "❌ TIFF-based analysis failed."
        print(error_msg)
        return annotated_pil, [], error_msg, 0

    colorbar_data = result.get("colorbar_results", [])
    total_blocks = result.get("total_blocks", 0)

    # --- Generate Report String ---
    # (This logic is similar to pure_colorbar_analysis.py, adapted slightly)
    report = f"🎯 Direct TIFF Analysis Results ({original_filename})\n" + "=" * 55 + "\n\n"
    # Add TIFF path used? result.get('temp_dir_to_clean', 'N/A') might be too verbose
    
    for i, res in enumerate(colorbar_data):
        best_card_id = res.get("best_match_card_id")
        block_count = res.get("block_count", 0)
        # Add INVALID_DETECTION check if needed based on block detection logic
        # if best_card_id == "INVALID_DETECTION": ...
        if best_card_id:
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
        # Purity stats removed as purity wasn't calculated directly here
    report += "\n"

    # Block details
    for res in colorbar_data:
        best_card_id = res.get("best_match_card_id", "N/A")
        report += f"🔎 Details for Colorbar (Matched to {best_card_id}):\n"
        # Add INVALID_DETECTION check here too if needed
        if res["block_count"] > 0:
            for analysis in res["pure_color_analyses"]:
                if "error" in analysis: continue # Should not happen if structure is right

                block_id = analysis.get("block_id", "?")
                rgb = analysis.get("pure_color_rgb", (0, 0, 0))
                cmyk = analysis.get("pure_color_cmyk", (0, 0, 0, 0))
                # purity = analysis.get("purity_score") # Removed
                # quality = analysis.get("color_quality", "N/A") # Removed
                hex_code = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
                report += f"    - Block {block_id}: {hex_code} (C{cmyk[0]} M{cmyk[1]} Y{cmyk[2]} K{cmyk[3]})"
                # Report purity/quality removed

                if "ground_truth_match" in analysis:
                    gt = analysis["ground_truth_match"]
                    # Use the serializable dict 'closest_color_dict' if needed by frontend directly
                    # Or access the object via 'closest_color'
                    gt_color_obj = gt.get("closest_color")
                    if gt_color_obj:
                         delta_e = gt["delta_e"]
                         level = gt["accuracy_level"]
                         gt_name = gt_color_obj.name # Access attribute from dataclass
                         
                         report += f" | ΔE: {delta_e:.2f} ({level}) vs {gt_name}"
                         if gt.get("is_excellent", False): report += " ✅"
                         elif gt.get("is_acceptable", False): report += " ⚠️"
                         else: report += " ❌"
                report += "\n"
        report += "\n"

    # Gradio return tuple: (Annotated Image, Data for HTML, Report String, Block Count)
    # We need to adjust the frontend to use colorbar_data to generate HTML
    return (annotated_pil, colorbar_data, report, total_blocks)

# Helper function needed in ground_truth_checker.py to serialize dataclass
# Add this method to the GroundTruthColorChecker class in ground_truth_checker.py
def _serialize_gt_color(self, gt_color_obj):
     if not gt_color_obj: return None
     # Ensure LAB is serializable (tuple of floats)
     lab_serializable = None
     if gt_color_obj.lab is not None:
          try:
               lab_serializable = tuple(float(f"{val:.2f}") for val in gt_color_obj.lab)
          except TypeError: # Handle case where it might already be a tuple
               lab_serializable = gt_color_obj.lab
               
     return {
         "id": gt_color_obj.id,
         "name": gt_color_obj.name,
         "cmyk": gt_color_obj.cmyk,
         "rgb": gt_color_obj.rgb,
         "lab": lab_serializable
     }
# Make sure ground_truth_checker.py uses this method where needed, e.g., in find_best_card...