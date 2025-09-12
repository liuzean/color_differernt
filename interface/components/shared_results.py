# interface/components/shared_results.py

"""
Shared UI component for displaying colorbar analysis results in a structured HTML format.
"""

from PIL import Image
import base64
from io import BytesIO


def image_to_base64(pil_image: Image.Image) -> str:
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
        # [新逻辑] 为关闭跳转创建唯一的锚点ID
        close_anchor_id = f"close_anchor_{i}"

        display_image_b64 = image_to_base64(result.get("segmented_colorbar_pil")) or image_to_base64(result.get("original_segment_pil"))

        # [新逻辑] 在卡片前添加关闭锚点
        html_parts.append(f"<a id='{close_anchor_id}'></a>")
        
        # --- Card Header ---
        html_parts.append(f"<div class='colorbar-result-card'><div class='card-header'><h3>🎨 Colorbar #{colorbar_id}</h3>")
        if best_match_card_id == "INVALID_DETECTION":
            html_parts.append(f"<span class='best-match-invalid'>ERROR: Too many blocks detected ({block_count} > 7)</span>")
        elif best_match_card_id:
            html_parts.append(f"<span class='best-match'>Best Match Card: <strong>{best_match_card_id.upper()}</strong></span>")
        else:
            html_parts.append("<span class='best-match-none'>No Match Found</span>")
        html_parts.append("</div>")

        # --- Main Content (Image on Top, Blocks Below) ---
        html_parts.append("<div class='card-content-top-down'>")
        
        # Top Part: Image with Fullscreen button
        html_parts.append("<div class='image-panel-top'>")
        if display_image_b64:
            # Fullscreen Modal Structure
            html_parts.append(f"""
            <div class='modal' id='{modal_id}'>
                <a href='#{close_anchor_id}' class='modal-bg'></a>
                <div class='modal-content'>
                    <a href='#{close_anchor_id}' class='modal-close'>&times;</a>
                    <img src='{display_image_b64}'/>
                </div>
            </div>
            """)
            # Image container with zoom button
            html_parts.append(f"<div class='image-wrapper'><img src='{display_image_b64}' alt='Colorbar Segment' /><a href='#{modal_id}' class='zoom-btn'>🔍</a></div>")
        else:
            html_parts.append("<p class='error-text'>Image not available</p>")
        html_parts.append("</div>")

        # Bottom Part: Blocks Grid
        html_parts.append("<div class='blocks-panel-bottom'>")
        
        block_analyses = result.get("pure_color_analyses") or result.get("block_analyses", [])

        if best_match_card_id == "INVALID_DETECTION":
             html_parts.append("<p class='error-text'>Matching skipped due to too many detected blocks.</p>")
        elif block_analyses:
            for analysis in block_analyses:
                if "error" in analysis: continue

                rgb = analysis.get("pure_color_rgb") or analysis.get("primary_color_rgb", (0,0,0))
                rgb_hex = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
                detected_cmyk = analysis.get("pure_color_cmyk") or analysis.get("primary_color_cmyk", ('N/A','N/A','N/A','N/A'))
                detected_cmyk_str = f"C{detected_cmyk[0]} M{detected_cmyk[1]} Y{detected_cmyk[2]} K{detected_cmyk[3]}"
                
                gt_match = analysis.get("ground_truth_match") or analysis.get("ground_truth_comparison", {})
                delta_e = gt_match.get("delta_e", float('inf'))
                
                status_symbol = ""
                if "is_excellent" in gt_match:
                    if gt_match["is_excellent"]: status_symbol = "✅"
                    elif gt_match["is_acceptable"]: status_symbol = "⚠️"
                    else: status_symbol = "❌"

                closest_color_info = gt_match.get("closest_color", {})
                gt_cmyk = closest_color_info.get('cmyk', ('N/A','N/A','N/A','N/A'))
                gt_cmyk_str = f"C{gt_cmyk[0]} M{gt_cmyk[1]} Y{gt_cmyk[2]} K{gt_cmyk[3]}"

                html_parts.append(f"""
                <div class='block-card-new'>
                    <div class='block-color-swatch-new' style='background-color: {rgb_hex};'></div>
                    <div class='block-details-new'>
                         <div class='block-detected-cmyk-new'>Detected: {detected_cmyk_str}</div>
                         <div class='block-gt-cmyk-new'>Standard: {gt_cmyk_str}</div>
                         <div class='block-delta-e-new'>ΔE: {delta_e:.2f} {status_symbol}</div>
                    </div>
                </div>
                """)
        html_parts.append("</div>")
        html_parts.append("</div>")
        html_parts.append("</div>")

    html_parts.append("</div>")

    # --- CSS Styling ---
    html_parts.append("""
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
            display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px; align-content: start;
        }
        
        .block-card-new {
            border: 1px solid #ccc; border-radius: 6px; background: #fff; padding: 8px;
            display: flex; align-items: center; gap: 10px;
        }
        .block-color-swatch-new { width: 50px; height: 50px; border-radius: 4px; border: 1px solid #888; flex-shrink: 0; }
        .block-details-new { text-align: left; flex-grow: 1; }
        .block-detected-cmyk-new { font-size: 0.8em; color: #333; line-height: 1.2; }
        .block-gt-cmyk-new { font-size: 0.8em; color: #777; line-height: 1.2; }
        .block-delta-e-new { font-size: 0.9em; font-weight: bold; color: #333; margin-top: 4px; }

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
    """)

    return "".join(html_parts)