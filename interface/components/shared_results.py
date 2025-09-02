"""
Shared Result Components for Colorbar Analysis

This module provides shared result display components that can be used by both
colorbar analysis and ground-truth demo interfaces. Results are designed to be
concise with one-line groups and clear visual indicators.
"""

import base64
import io
from typing import Dict, List, Optional

import gradio as gr
from PIL import Image


def create_concise_colorbar_display(colorbar_data: List[Dict]) -> str:
    """
    Create concise HTML display for colorbar analysis results.

    Args:
        colorbar_data: List of colorbar analysis results

    Returns:
        HTML string with concise colorbar display
    """
    if not colorbar_data:
        return "<div class='no-results'>No colorbar data available</div>"

    html = """
    <style>
    .colorbar-container { margin-bottom: 15px; border: 1px solid #ddd; border-radius: 6px; padding: 10px; background: #fafafa; }
    .colorbar-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; padding: 8px; background: #f0f0f0; border-radius: 4px; }
    .colorbar-title { font-weight: bold; color: #333; font-size: 14px; }
    .colorbar-confidence { background: #4CAF50; color: white; padding: 2px 6px; border-radius: 10px; font-size: 11px; }
    .colorbar-images { display: flex; gap: 10px; margin-bottom: 10px; }
    .colorbar-image { flex: 1; text-align: center; }
    .colorbar-image img { max-width: 100%; height: 60px; border: 1px solid #ccc; border-radius: 3px; }
    .colorbar-image-label { font-size: 10px; color: #666; margin-top: 2px; }
    .color-blocks-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 8px; }
    .color-block { display: flex; align-items: center; gap: 8px; padding: 6px; background: white; border: 1px solid #ddd; border-radius: 4px; font-size: 11px; }
    .color-block.excellent { border-left: 4px solid #4CAF50; }
    .color-block.acceptable { border-left: 4px solid #FF9800; }
    .color-block.poor { border-left: 4px solid #f44336; }
    .color-preview { width: 30px; height: 30px; border-radius: 4px; border: 1px solid #333; flex-shrink: 0; }
    .color-info { flex: 1; }
    .color-block-id { font-weight: bold; color: #333; }
    .cmyk-values { color: #666; font-size: 10px; }
    .delta-e-info { display: flex; align-items: center; gap: 4px; }
    .delta-e-value { font-weight: bold; }
    .result-indicator { font-size: 12px; }
    .summary-stats { background: #e8f4f8; border: 1px solid #2196F3; border-radius: 4px; padding: 8px; margin-bottom: 10px; font-size: 11px; }
    .summary-stats-title { font-weight: bold; color: #1976D2; margin-bottom: 4px; }
    </style>
    """

    # Calculate overall statistics
    total_blocks = 0
    excellent_count = 0
    acceptable_count = 0
    high_purity_count = 0
    all_delta_e = []

    for colorbar in colorbar_data:
        for analysis in colorbar.get("pure_color_analyses", []):
            if "error" not in analysis:
                total_blocks += 1
                gt_match = analysis.get("ground_truth_match", {})
                if gt_match.get("is_excellent"):
                    excellent_count += 1
                if gt_match.get("is_acceptable"):
                    acceptable_count += 1
                if analysis.get("purity_score", 0) >= 0.8:
                    high_purity_count += 1
                if "delta_e" in gt_match:
                    all_delta_e.append(gt_match["delta_e"])

    # Add summary statistics
    if all_delta_e:
        avg_delta_e = sum(all_delta_e) / len(all_delta_e)
        html += f"""
        <div class="summary-stats">
            <div class="summary-stats-title">📊 Analysis Summary</div>
            <div>Total blocks: {total_blocks} | Avg ΔE: {avg_delta_e:.2f} | Excellent: {excellent_count} | Acceptable: {acceptable_count} | High purity: {high_purity_count}</div>
        </div>
        """

    # Process each colorbar
    for colorbar in colorbar_data:
        colorbar_id = colorbar.get("colorbar_id", "?")
        confidence = colorbar.get("confidence", 0)
        original_colorbar = colorbar.get("original_colorbar")
        segmented_colorbar = colorbar.get("segmented_colorbar")
        pure_color_analyses = colorbar.get("pure_color_analyses", [])

        html += f"""
        <div class="colorbar-container">
            <div class="colorbar-header">
                <div class="colorbar-title">🎯 Colorbar {colorbar_id}</div>
                <div class="colorbar-confidence">{confidence:.2f}</div>
            </div>
        """

        # Show detected | ground-truth colorbar header
        if original_colorbar or segmented_colorbar:
            html += '<div class="colorbar-images">'

            if original_colorbar:
                buffer = io.BytesIO()
                original_colorbar.save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode()
                html += f"""
                <div class="colorbar-image">
                    <img src="data:image/png;base64,{img_str}" alt="Detected colorbar">
                    <div class="colorbar-image-label">Detected Colorbar</div>
                </div>
                """

            if segmented_colorbar:
                buffer = io.BytesIO()
                segmented_colorbar.save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode()
                html += f"""
                <div class="colorbar-image">
                    <img src="data:image/png;base64,{img_str}" alt="Ground-truth colorbar">
                    <div class="colorbar-image-label">Ground-Truth Colorbar</div>
                </div>
                """

            html += "</div>"

        # Color blocks in concise grid
        if pure_color_analyses:
            html += '<div class="color-blocks-grid">'

            for analysis in pure_color_analyses:
                if "error" in analysis:
                    continue

                block_id = analysis.get("block_id", "?")
                pure_rgb = analysis.get("pure_color_rgb", (0, 0, 0))
                pure_cmyk = analysis.get("pure_color_cmyk", (0, 0, 0, 0))
                gt_match = analysis.get("ground_truth_match", {})

                # Determine block styling
                block_class = "color-block"
                result_icon = "❌"
                if gt_match.get("is_excellent"):
                    block_class += " excellent"
                    result_icon = "✅"
                elif gt_match.get("is_acceptable"):
                    block_class += " acceptable"
                    result_icon = "⚠️"
                else:
                    block_class += " poor"
                    result_icon = "❌"

                # Color preview
                color_style = f"background-color: rgb({pure_rgb[0]}, {pure_rgb[1]}, {pure_rgb[2]});"

                # Delta E info
                delta_e = gt_match.get("delta_e", 0)
                delta_e_display = f"ΔE: {delta_e:.2f}" if delta_e > 0 else "ΔE: N/A"

                html += f"""
                <div class="{block_class}">
                    <div class="color-preview" style="{color_style}"></div>
                    <div class="color-info">
                        <div class="color-block-id">{colorbar_id}.{block_id}</div>
                        <div class="cmyk-values">C:{pure_cmyk[0]} M:{pure_cmyk[1]} Y:{pure_cmyk[2]} K:{pure_cmyk[3]}</div>
                    </div>
                    <div class="delta-e-info">
                        <div class="delta-e-value">{delta_e_display}</div>
                        <div class="result-indicator">{result_icon}</div>
                    </div>
                </div>
                """

            html += "</div>"
        else:
            html += '<div style="text-align: center; color: #666; font-style: italic; padding: 10px;">No pure color blocks detected</div>'

        html += "</div>"

    return html


def create_shared_colorbar_results_component():
    """
    Create a shared colorbar results component that can be reused across interfaces.

    Returns:
        Gradio HTML component for displaying colorbar results
    """
    return gr.HTML(
        label="🎨 Colorbar Analysis Results",
        value="<div style='text-align: center; padding: 20px; color: #666; background: #f9f9f9; border-radius: 6px;'>Upload an image and analyze to see detailed results</div>",
    )


def update_shared_results_display(colorbar_data: List[Dict]) -> str:
    """
    Update the shared results display with new colorbar data.

    Args:
        colorbar_data: List of colorbar analysis results

    Returns:
        HTML string for the updated display
    """
    if not colorbar_data:
        return "<div style='text-align: center; padding: 20px; color: #666; background: #f9f9f9; border-radius: 6px;'>No colorbar data available</div>"

    return create_concise_colorbar_display(colorbar_data)
