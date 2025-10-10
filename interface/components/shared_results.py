# interface/components/shared_results.py

"""
Shared UI component for displaying colorbar analysis results in a structured HTML format.
"""

from PIL import Image
import base64
from io import BytesIO


def image_to_base_64(pil_image: Image.Image) -> str:
    """
    [最终修正] Convert a PIL Image to a Base64 string for embedding in HTML, with robust error handling.
    """
    if not isinstance(pil_image, Image.Image):
        return ""
    try:
        buffered = BytesIO()
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        pil_image.save(buffered, format="JPEG", quality=90)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return ""


def update_shared_results_display(colorbar_data: list[dict]) -> str:
    """
    Generate an HTML string to display the results of a colorbar analysis.
    This is the final version with all requested UI changes.
    """
    if not colorbar_data:
        return "<div class='no-results'>No colorbars detected or analysis failed.</div>"

    html_parts = ["<div class='results-container'>"]

    for i, result in enumerate(colorbar_data):
        colorbar_id = result.get("colorbar_id", "N/A")
        best_match_card_id = result.get("best_match_card_id")
        block_count = result.get("block_count", 0)

        modal_id = f"modal_{i}"
        close_anchor_id = f"close_anchor_{i}"

        display_image_b64 = image_to_base_64(
            result.get("segmented_colorbar_pil")
        ) or image_to_base_64(result.get("original_segment_pil"))

        html_parts.append(f"<a id='{close_anchor_id}'></a>")

        # --- Card Header ---
        html_parts.append(
            f"<div class='colorbar-result-card'><div class='card-header'><h3>🎨 Colorbar #{colorbar_id}</h3>"
        )
        if best_match_card_id == "INVALID_DETECTION":
            html_parts.append(
                f"<span class='best-match-invalid'>ERROR: Too many blocks detected ({block_count} > 7)</span>"
            )
        elif best_match_card_id:
            html_parts.append(
                f"<span class='best-match'>Best Match Card: <strong>{best_match_card_id.upper()}</strong></span>"
            )
        else:
            html_parts.append("<span class='best-match-none'>No Match Found</span>")
        html_parts.append("</div>")

        # --- Main Content (Image on Top, Blocks Below) ---
        html_parts.append("<div class='card-content-top-down'>")

        # Top Part: Image with Fullscreen button
        html_parts.append("<div class='image-panel-top'>")
        if display_image_b64:
            html_parts.append(
                f"""
            <div class='modal' id='{modal_id}'>
                <a href='#{close_anchor_id}' class='modal-bg'></a>
                <div class='modal-content'>
                    <a href='#{close_anchor_id}' class='modal-close'>&times;</a>
                    <img src='{display_image_b64}'/>
                </div>
            </div>
            """
            )
            html_parts.append(
                f"<div class='image-wrapper'><img src='{display_image_b64}' alt='Colorbar Segment' /><a href='#{modal_id}' class='zoom-btn'>🔍</a></div>"
            )
        else:
            html_parts.append("<p class='error-text'>Image not available</p>")
        html_parts.append("</div>")

        # Bottom Part: Blocks Grid
        html_parts.append("<div class='blocks-panel-bottom'>")

        block_analyses = result.get("pure_color_analyses") or result.get(
            "block_analyses", []
        )

        if best_match_card_id == "INVALID_DETECTION":
            html_parts.append(
                "<p class='error-text'>Matching skipped due to too many detected blocks.</p>"
            )
        elif block_analyses:
            for analysis in block_analyses:
                if "error" in analysis:
                    continue

                # --- 1. 准备所有需要展示的数据 ---
                # --- 检测色 ---
                 # [Compatibility INFO] New display format. Old format was "C100 M0..." and "L*x a*y...".
                
                # --- 检测色 ---
                rgb = analysis.get("pure_color_rgb") or analysis.get("primary_color_rgb", (0, 0, 0))
                rgb_hex = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
                
                detected_cmyk = analysis.get("pure_color_cmyk") or analysis.get("primary_color_cmyk", (0,0,0,0))
                detected_lab_raw = analysis.get("detected_lab") or analysis.get("primary_color_lab", (0,0,0))
                
                detected_rgb_str = f"RGB: ({rgb[0]}, {rgb[1]}, {rgb[2]})"
                detected_cmyk_str = f"CMYK: ({detected_cmyk[0]}, {detected_cmyk[1]}, {detected_cmyk[2]}, {detected_cmyk[3]})"
                detected_lab_str = f"LAB: ({detected_lab_raw[0]:.1f}, {detected_lab_raw[1]:.1f}, {detected_lab_raw[2]:.1f})"

                # --- 标准色 ---
                gt_match = analysis.get("ground_truth_match") or analysis.get("ground_truth_comparison", {})
                delta_e = gt_match.get("delta_e", float("inf"))
                
                closest_color_info = gt_match.get("closest_color", {})
                if closest_color_info:
                    gt_rgb = closest_color_info.get("rgb", (0, 0, 0))
                    gt_cmyk = closest_color_info.get("cmyk", (0, 0, 0, 0))
                    gt_lab = closest_color_info.get("lab", (0, 0, 0))
                    
                    standard_rgb_str = f"RGB: ({gt_rgb[0]}, {gt_rgb[1]}, {gt_rgb[2]})"
                    gt_cmyk_str = f"CMYK: ({gt_cmyk[0]}, {gt_cmyk[1]}, {gt_cmyk[2]}, {gt_cmyk[3]})"
                    standard_lab_str = f"LAB: ({gt_lab[0]:.1f}, {gt_lab[1]:.1f}, {gt_lab[2]:.1f})"
                else:
                    standard_rgb_str = "RGB: N/A"
                    gt_cmyk_str = "CMYK: N/A"
                    standard_lab_str = "LAB: N/A"

                
                status_symbol = ""
                if "is_excellent" in gt_match:
                    if gt_match["is_excellent"]:
                        status_symbol = "✅"
                    elif gt_match["is_acceptable"]:
                        status_symbol = "⚠️"
                    else:
                        status_symbol = "❌"

                # --- 2. 生成新的、更详细的HTML卡片 ---
                html_parts.append(
                    f"""
                <div class='block-card-new'>
                    <div class='block-color-swatch-new' style='background-color: {rgb_hex};'></div>
                    <div class='block-details-new'>
                         <div class='block-color-group'>
                            <div class='block-title'>Detected</div>
                            <div class='block-value'>{detected_rgb_str}</div>
                            <div class='block-value'>{detected_cmyk_str}</div>
                            <div class='block-value'>{detected_lab_str}</div>
                         </div>
                         <div class='block-color-group'>
                            <div class='block-title'>Standard</div>
                            <div class='block-value'>{standard_rgb_str}</div>
                            <div class='block-value'>{gt_cmyk_str}</div>
                            <div class='block-value'>{standard_lab_str}</div>
                         </div>
                         <div class='block-delta-e-new'>ΔE: {delta_e:.2f} {status_symbol}</div>
                    </div>
                </div>
                """
                )
        html_parts.append("</div>")
        html_parts.append("</div>")
        html_parts.append("</div>")

    html_parts.append("</div>")

    # --- CSS Styling ---
    html_parts.append(
        """
    <style>
        .results-container { font-family: sans-serif; }
        .no-results, .error-text { text-align: center; color: #888; padding: 20px; }
        .colorbar-result-card { border: 1px solid #ddd; border-radius: 8px; margin-bottom: 20px; background: #f9f9f9; padding: 15px; }
        .card-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #eee; padding-bottom: 10px; margin-bottom: 15px; }
        .card-header h3 { margin: 0; color: #333; font-size: 1.1em; }
        .best-match { background-color: #e7f3ff; color: #005a9e; padding: 5px 10px; border-radius: 12px; font-size: 0.9em; }
        .best-match-invalid { background-color: #ffe7e7; color: #9e0000; padding: 5px 10px; border-radius: 12px; font-size: 0.9em; font-weight: bold; }
        
        .card-content-top-down { display: flex; flex-direction: column; gap: 15px; }
        .image-panel-top {
            width: 100%; border: 1px solid #ddd; border-radius: 4px; padding: 5px;
            background: #fff; display: flex; justify-content: center; align-items: center;
            max-height: 100px; overflow: hidden;
        }
        .image-wrapper { position: relative; max-width: 100%; max-height: 100%; }
        .image-panel-top img { width: auto; height: auto; max-width: 100%; max-height: 90px; display: block; }
        .zoom-btn {
            position: absolute; top: 5px; right: 5px; background: rgba(0,0,0,0.5); color: white;
            border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center;
            text-decoration: none; font-size: 14px; transition: background 0.2s; z-index: 10;
        }
        .zoom-btn:hover { background: rgba(0,0,0,0.8); }

        .blocks-panel-bottom {
            display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 10px; align-content: start;
        }
        
        .block-card-new {
            border: 1px solid #ccc; border-radius: 6px; background: #fff; padding: 8px;
            display: flex; align-items: center; gap: 10px;
        }
        .block-color-swatch-new { width: 50px; height: 50px; border-radius: 4px; border: 1px solid #888; flex-shrink: 0; }
        
        .block-details-new { text-align: left; flex-grow: 1; }

        /* --- 从这里开始是新增/修改的规则 --- */
        .block-color-group { margin-bottom: 5px; }
        .block-title { font-size: 0.75em; color: #888; font-weight: bold; }
        .block-value { font-size: 0.8em; color: #333; line-height: 1.3; }
        
        .block-delta-e-new { font-size: 0.9em; font-weight: bold; color: #333; margin-top: 4px; }
        
        /* 隐藏旧的、不再需要的CMYK单行显示规则 */
        .block-detected-cmyk-new, .block-gt-cmyk-new { display: none; } 
        /* --- 修改结束 --- */

        /* [修正] Fullscreen Modal Styles */
        .modal {
            visibility: hidden; position: fixed; top: 0; left: 0;
            width: 100%; height: 100%; background: rgba(0,0,0,0.8);
            z-index: 9998; opacity: 0; transition: opacity 0.3s, visibility 0.3s;
            display: flex; justify-content: center; align-items: center;
        }
        .modal:target { visibility: visible; opacity: 1; }
        .modal-bg { position: absolute; width: 100%; height: 100%; top: 0; left: 0; cursor: pointer; }
        .modal-content {
            position: relative;
            max-width: 90vw; max-height: 90vh;
            padding: 10px; background: white; border-radius: 8px;
        }
        .modal-content img {
            display: block;
            max-width: 100%;
            max-height: calc(90vh - 20px); /* 90vh minus padding */
            object-fit: contain; /* [修正] Ensures full image is visible */
        }
        .modal-close {
            position: absolute; top: -15px; right: -15px;
            text-decoration: none; background: #333; color: #fff;
            border-radius: 50%; width: 30px; height: 30px;
            display: flex; align-items: center; justify-content: center;
            font-size: 20px; line-height: 1; border: 2px solid white;
            z-index: 100;
        }
    </style>
    """
    )

    return "".join(html_parts)