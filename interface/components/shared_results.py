"""
颜色条分析共享结果组件

此模块提供可在颜色条分析和真值演示界面之间共享的结果显示组件。
结果设计为简洁的单行组显示，带有清晰的视觉指示器。
"""

import base64
import io

import gradio as gr


def create_concise_colorbar_display(colorbar_data: list[dict]) -> str:
    """
    为颜色条分析结果创建简洁的HTML显示

    Args:
        colorbar_data: 颜色条分析结果列表

    Returns:
        包含简洁颜色条显示的HTML字符串
    """
    # 检查是否有颜色条数据
    if not colorbar_data:
        return "<div class='no-results'>无可用的颜色条数据</div>"

    # CSS样式定义 - 设计简洁紧凑的显示风格
    html = """
    <style>
    /* 颜色条容器样式 */
    .colorbar-container { margin-bottom: 15px; border: 1px solid #ddd; border-radius: 6px; padding: 10px; background: #fafafa; }

    /* 颜色条标题栏样式 */
    .colorbar-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; padding: 8px; background: #f0f0f0; border-radius: 4px; }
    .colorbar-title { font-weight: bold; color: #333; font-size: 14px; }
    .colorbar-confidence { background: #4CAF50; color: white; padding: 2px 6px; border-radius: 10px; font-size: 11px; }

    /* 颜色条图像显示样式 */
    .colorbar-images { display: flex; gap: 10px; margin-bottom: 10px; }
    .colorbar-image { flex: 1; text-align: center; }
    .colorbar-image img { max-width: 100%; height: 60px; border: 1px solid #ccc; border-radius: 3px; }
    .colorbar-image-label { font-size: 10px; color: #666; margin-top: 2px; }

    /* 色块网格布局样式 */
    .color-blocks-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 8px; }
    .color-block { display: flex; align-items: center; gap: 8px; padding: 6px; background: white; border: 1px solid #ddd; border-radius: 4px; font-size: 11px; }

    /* 根据准确性级别的色块样式 */
    .color-block.excellent { border-left: 4px solid #4CAF50; }  /* 优秀：绿色 */
    .color-block.acceptable { border-left: 4px solid #FF9800; } /* 可接受：橙色 */
    .color-block.poor { border-left: 4px solid #f44336; }       /* 较差：红色 */

    /* 颜色预览和信息样式 */
    .color-preview { width: 30px; height: 30px; border-radius: 4px; border: 1px solid #333; flex-shrink: 0; }
    .color-info { flex: 1; }
    .color-block-id { font-weight: bold; color: #333; }
    .cmyk-values { color: #666; font-size: 10px; }

    /* Delta E 信息显示样式 */
    .delta-e-info { display: flex; align-items: center; gap: 4px; }
    .delta-e-value { font-weight: bold; }
    .result-indicator { font-size: 12px; }

    /* 统计摘要样式 */
    .summary-stats { background: #e8f4f8; border: 1px solid #2196F3; border-radius: 4px; padding: 8px; margin-bottom: 10px; font-size: 11px; }
    .summary-stats-title { font-weight: bold; color: #1976D2; margin-bottom: 4px; }
    </style>
    """

    # 计算整体统计数据
    total_blocks = 0          # 总色块数
    excellent_count = 0       # 优秀级别色块数
    acceptable_count = 0      # 可接受级别色块数
    high_purity_count = 0     # 高纯度色块数
    all_delta_e = []         # 所有Delta E值

    # 遍历所有颜色条数据统计信息
    for colorbar in colorbar_data:
        # 遍历每个颜色条的纯色分析结果
        for analysis in colorbar.get("pure_color_analyses", []):
            # 跳过有错误的分析结果
            if "error" not in analysis:
                total_blocks += 1

                # 获取真值匹配结果
                gt_match = analysis.get("ground_truth_match", {})

                # 统计优秀和可接受的色块
                if gt_match.get("is_excellent"):
                    excellent_count += 1
                if gt_match.get("is_acceptable"):
                    acceptable_count += 1

                # 统计高纯度色块
                if analysis.get("purity_score", 0) >= 0.8:
                    high_purity_count += 1

                # 收集所有Delta E值
                if "delta_e" in gt_match:
                    all_delta_e.append(gt_match["delta_e"])

    # 添加统计摘要（如果有Delta E数据）
    if all_delta_e:
        avg_delta_e = sum(all_delta_e) / len(all_delta_e)  # 计算平均Delta E
        html += f"""
        <div class="summary-stats">
            <div class="summary-stats-title">📊 分析摘要</div>
            <div>总色块数: {total_blocks} | 平均ΔE: {avg_delta_e:.2f} | 优秀: {excellent_count} | 可接受: {acceptable_count} | 高纯度: {high_purity_count}</div>
        </div>
        """

    # 处理每个颜色条的显示
    for colorbar in colorbar_data:
        # 提取颜色条基本信息
        colorbar_id = colorbar.get("colorbar_id", "?")              # 颜色条ID
        confidence = colorbar.get("confidence", 0)                   # 检测置信度
        original_colorbar = colorbar.get("original_colorbar")        # 原始检测到的颜色条
        segmented_colorbar = colorbar.get("segmented_colorbar")      # 分割后的颜色条
        pure_color_analyses = colorbar.get("pure_color_analyses", []) # 纯色分析结果

        # 创建颜色条容器和标题
        html += f"""
        <div class="colorbar-container">
            <div class="colorbar-header">
                <div class="colorbar-title">🎯 颜色条 {colorbar_id}</div>
                <div class="colorbar-confidence">{confidence:.2f}</div>
            </div>
        """

        # 显示颜色条图像（原始 | 真值对比）
        if original_colorbar or segmented_colorbar:
            html += '<div class="colorbar-images">'

            # 显示原始检测到的颜色条
            if original_colorbar:
                # 将PIL图像转换为base64字符串用于HTML显示
                buffer = io.BytesIO()
                original_colorbar.save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode()
                html += f"""
                <div class="colorbar-image">
                    <img src="data:image/png;base64,{img_str}" alt="检测到的颜色条">
                    <div class="colorbar-image-label">检测到的颜色条</div>
                </div>
                """

            # 显示分割后的颜色条（如YOLO检测结果）
            if segmented_colorbar:
                buffer = io.BytesIO()
                segmented_colorbar.save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode()
                html += f"""
                <div class="colorbar-image">
                    <img src="data:image/png;base64,{img_str}" alt="真值颜色条">
                    <div class="colorbar-image-label">真值颜色条</div>
                </div>
                """

            html += "</div>"

        # 以简洁网格显示色块分析结果
        if pure_color_analyses:
            html += '<div class="color-blocks-grid">'

            # 遍历每个色块的分析结果
            for analysis in pure_color_analyses:
                # 跳过有错误的分析结果
                if "error" in analysis:
                    continue

                # 提取色块信息
                block_id = analysis.get("block_id", "?")                    # 色块ID
                pure_rgb = analysis.get("pure_color_rgb", (0, 0, 0))        # RGB值
                pure_cmyk = analysis.get("pure_color_cmyk", (0, 0, 0, 0))   # CMYK值
                gt_match = analysis.get("ground_truth_match", {})           # 真值匹配结果

                # 根据分析质量确定色块样式
                block_class = "color-block"
                result_icon = "❌"  # 默认为较差

                if gt_match.get("is_excellent"):
                    block_class += " excellent"  # 优秀：绿色边框
                    result_icon = "✅"
                elif gt_match.get("is_acceptable"):
                    block_class += " acceptable"  # 可接受：橙色边框
                    result_icon = "⚠️"
                else:
                    block_class += " poor"  # 较差：红色边框
                    result_icon = "❌"

                # 设置颜色预览的CSS样式
                color_style = f"background-color: rgb({pure_rgb[0]}, {pure_rgb[1]}, {pure_rgb[2]});"

                # 获取Delta E信息
                delta_e = gt_match.get("delta_e", 0)
                delta_e_display = f"ΔE: {delta_e:.2f}" if delta_e > 0 else "ΔE: 无数据"

                # 创建色块显示元素
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
            # 如果没有检测到纯色块，显示提示信息
            html += '<div style="text-align: center; color: #666; font-style: italic; padding: 10px;">未检测到纯色块</div>'

        html += "</div>"  # 结束颜色条容器

    return html


def create_shared_colorbar_results_component():
    """
    创建可在多个界面中重用的共享颜色条结果组件

    Returns:
        用于显示颜色条结果的Gradio HTML组件
    """
    return gr.HTML(
        label="🎨 颜色条分析结果",
        value="<div style='text-align: center; padding: 20px; color: #666; background: #f9f9f9; border-radius: 6px;'>上传图像并分析以查看详细结果</div>",
    )


def update_shared_results_display(colorbar_data: list[dict]) -> str:
    """
    使用新的颜色条数据更新共享结果显示

    Args:
        colorbar_data: 颜色条分析结果列表

    Returns:
        更新显示的HTML字符串
    """
    # 检查是否有数据
    if not colorbar_data:
        return "<div style='text-align: center; padding: 20px; color: #666; background: #f9f9f9; border-radius: 6px;'>无可用的颜色条数据</div>"

    # 调用主要的显示创建函数
    return create_concise_colorbar_display(colorbar_data)