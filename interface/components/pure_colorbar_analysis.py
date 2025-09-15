"""
纯色颜色条分析界面组件

该组件为重新设计的基于纯色的颜色条分析系统提供界面，
具有增强的真值比较和清晰的CMYK/delta E报告功能。
支持双YOLO模型的色块检测。
"""

import gradio as gr
from PIL import Image

from core.block_detection.pure_colorbar_analysis import (
    pure_colorbar_analysis_for_gradio,
)
from core.color.ground_truth_checker import ground_truth_checker


def process_pure_colorbar_analysis(
    input_image: Image.Image,
    # YOLO颜色条参数
    confidence_threshold: float = 0.6,
    box_expansion: int = 10,
    # YOLO色块检测参数
    block_confidence_threshold: float = 0.5,
    min_block_area: int = 50,
    # 纯色分析参数
    purity_threshold: float = 0.8,
) -> tuple[Image.Image, str, str]:
    """
    处理基于双YOLO模型的纯色颜色条分析并返回格式化结果。

    流程：
    1. YOLO检测颜色条区域 (best0710.pt)
    2. YOLO检测色块 (best.pt)
    3. 纯色分析

    返回:
        (标注图像, 状态消息, 结果HTML) 的元组
    """
    if input_image is None:
        return None, "未提供图像", ""

    try:
        # 运行双YOLO纯色颜色条分析
        (
            annotated_image,  # 标注后的图像
            colorbar_data,  # 颜色条数据
            analysis_report,  # 分析报告
            total_blocks,  # 总色块数
        ) = pure_colorbar_analysis_for_gradio(
            input_image,
            confidence_threshold=confidence_threshold,
            box_expansion=box_expansion,
            block_confidence_threshold=block_confidence_threshold,
            min_block_area=min_block_area,
            purity_threshold=purity_threshold,
        )

        if not colorbar_data:
            # 即使没有颜色条，也可能返回一个带注释的图像
            if annotated_image:
                return annotated_image, "未检测到颜色条", ""
            return None, "未检测到颜色条", ""

        # 创建专注于纯色的增强HTML显示
        results_html = create_pure_colorbar_display(colorbar_data)

        status = (
            f"✅ 双YOLO分析完成: {len(colorbar_data)} 个颜色条, {total_blocks} 个色块"
        )

        return annotated_image, status, results_html

    except Exception as e:
        error_msg = f"❌ 双YOLO分析过程中发生错误: {str(e)}"
        return input_image, error_msg, ""


def create_pure_colorbar_display(colorbar_data: list[dict]) -> str:
    """创建专注于纯色分析结果的增强HTML显示。"""
    if not colorbar_data:
        return "<div class='no-results'>无可用的纯色颜色条数据</div>"

    # CSS样式定义
    html = """
    <style>
    .pure-colorbar-container { margin-bottom: 20px; border: 2px solid #2196F3; border-radius: 8px; padding: 15px; background: #f8f9fa; }
    .pure-colorbar-header { display: flex; align-items: center; margin-bottom: 12px; font-weight: bold; color: #1976D2; font-size: 16px; }
    .confidence-badge { background: #4CAF50; color: white; padding: 3px 8px; border-radius: 12px; font-size: 12px; margin-left: 10px; }
    .colorbar-segments { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px; }
    .segment-box { text-align: center; border: 1px solid #ddd; border-radius: 6px; padding: 8px; background: white; }
    .segment-label { font-size: 12px; color: #666; margin-bottom: 6px; font-weight: bold; }
    .segment-image { border: 1px solid #ccc; border-radius: 4px; max-width: 100%; height: auto; }
    .pure-color-blocks-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 8px; margin-top: 10px; }
    .pure-color-block { border: 2px solid #e0e0e0; border-radius: 6px; padding: 10px; background: white; text-align: center; font-size: 11px; transition: border-color 0.3s; }
    .pure-color-block.excellent { border-color: #4CAF50; background: #f1f8e9; }
    .pure-color-block.acceptable { border-color: #FF9800; background: #fff3e0; }
    .pure-color-block.poor { border-color: #f44336; background: #ffebee; }
    .color-preview { width: 40px; height: 40px; border-radius: 8px; margin: 0 auto 8px auto; border: 2px solid #333; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }
    .color-info { margin-bottom: 8px; }
    .pure-rgb-info { color: #1976D2; font-weight: bold; font-size: 12px; margin-bottom: 4px; }
    .cmyk-values { color: #333; font-size: 11px; margin-bottom: 6px; font-weight: bold; background: #f5f5f5; padding: 4px; border-radius: 3px; }
    .purity-info { margin-bottom: 6px; padding: 4px; border-radius: 3px; font-size: 10px; }
    .purity-info.high { background: #e8f5e8; color: #2e7d32; }
    .purity-info.medium { background: #fff3e0; color: #f57c00; }
    .purity-info.low { background: #ffebee; color: #c62828; }
    .delta-e-info { margin-top: 6px; padding: 5px; border-radius: 4px; font-size: 10px; font-weight: bold; }
    .delta-e-info.excellent { background: #c8e6c9; color: #1b5e20; }
    .delta-e-info.good { background: #ffe0b2; color: #e65100; }
    .delta-e-info.poor { background: #ffcdd2; color: #b71c1c; }
    .delta-e-value { font-size: 12px; font-weight: bold; }
    .accuracy-level { font-size: 9px; opacity: 0.9; }
    .ground-truth-match { font-size: 9px; opacity: 0.8; margin-top: 3px; }
    .status-indicator { font-size: 16px; margin-left: 5px; }
    .summary-stats { background: #e3f2fd; border: 1px solid #2196F3; border-radius: 6px; padding: 10px; margin-bottom: 15px; font-size: 12px; }
    .summary-stats h4 { margin: 0 0 8px 0; color: #1976D2; }
    .detection-method { background: #fff3e0; border: 1px solid #FF9800; border-radius: 4px; padding: 8px; margin-bottom: 10px; font-size: 11px; }
    </style>
    """

    # 计算总体统计数据
    total_blocks = 0  # 总色块数
    excellent_count = 0  # 优秀级别色块数
    acceptable_count = 0  # 可接受级别色块数
    high_purity_count = 0  # 高纯度色块数
    all_delta_e = []  # 所有Delta E值

    # 遍历所有颜色条数据统计信息
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

    # 添加检测方法说明
    html += """
    <div class="detection-method">
        🔧 <strong>检测方法:</strong> 双YOLO检测模式
        <br>📍 <strong>颜色条模型:</strong> best0710.pt 
        <br>📍 <strong>色块模型:</strong> best.pt (新模型)
    </div>
    """

    # 添加统计摘要
    html += f"""
    <div class="summary-stats">
        <h4>📊 纯色分析摘要</h4>
        <div>分析的纯色块总数: <strong>{total_blocks}</strong></div>
    """

    # 如果有Delta E数据，添加详细统计
    if all_delta_e:
        avg_delta_e = sum(all_delta_e) / len(all_delta_e)
        html += f"""
        <div>平均 ΔE: <strong>{avg_delta_e:.2f}</strong></div>
        <div>优秀颜色 (ΔE &lt; 1.0): <strong>{excellent_count}/{total_blocks}</strong> ({(excellent_count/total_blocks*100):.1f}%)</div>
        <div>可接受颜色 (ΔE &lt; 3.0): <strong>{acceptable_count}/{total_blocks}</strong> ({(acceptable_count/total_blocks*100):.1f}%)</div>
        <div>高纯度颜色 (&gt; 0.8): <strong>{high_purity_count}/{total_blocks}</strong> ({(high_purity_count/total_blocks*100):.1f}%)</div>
        """

    html += "</div>"

    # 遍历每个颜色条，生成详细显示
    for colorbar in colorbar_data:
        colorbar_id = colorbar.get("colorbar_id", "?")  # 颜色条ID
        confidence = colorbar.get("confidence", 0)  # 检测置信度
        block_count = colorbar.get("block_count", 0)  # 色块数量
        original_colorbar = colorbar.get("original_segment_pil")  # 原始颜色条图像
        segmented_colorbar = colorbar.get(
            "segmented_colorbar_pil"
        )  # 分割后的颜色条图像
        pure_color_analyses = colorbar.get("pure_color_analyses", [])  # 纯色分析结果

        html += f"""
        <div class="pure-colorbar-container">
            <div class="pure-colorbar-header">
                🎯 纯色颜色条 {colorbar_id} (双YOLO检测)
                <span class="confidence-badge">{confidence:.2f}</span>
            </div>
        """

        # 显示原始和分割后的颜色条图像
        if original_colorbar or segmented_colorbar:
            html += """
            <div class="colorbar-segments">
            """
            if original_colorbar:
                # 将PIL图像转换为base64用于显示
                import base64
                import io

                buffer = io.BytesIO()
                original_colorbar.save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode()
                html += f"""
                <div class="segment-box">
                    <div class="segment-label">原始颜色条</div>
                    <img src="data:image/png;base64,{img_str}" class="segment-image" alt="原始颜色条">
                </div>
                """

            if segmented_colorbar:
                # 将PIL图像转换为base64用于显示
                buffer = io.BytesIO()
                segmented_colorbar.save(buffer, format="PNG")
                img_str = base64.b64encode(buffer.getvalue()).decode()
                html += f"""
                <div class="segment-box">
                    <div class="segment-label">YOLO检测的色块</div>
                    <img src="data:image/png;base64,{img_str}" class="segment-image" alt="YOLO色块检测结果">
                </div>
                """

            html += "</div>"

        # 纯色块分析结果
        if pure_color_analyses:
            html += """
            <div class="pure-color-blocks-grid">
            """

            for analysis in pure_color_analyses:
                if "error" in analysis:  # 跳过有错误的分析结果
                    continue

                block_id = analysis.get("block_id", "?")  # 色块ID
                pure_rgb = analysis.get("pure_color_rgb", (0, 0, 0))  # RGB值
                pure_cmyk = analysis.get("pure_color_cmyk", (0, 0, 0, 0))  # CMYK值
                purity_score = analysis.get("purity_score", 0.0)  # 纯度分数
                color_quality = analysis.get("color_quality", "Unknown")  # 颜色质量
                gt_match = analysis.get("ground_truth_match", {})  # 真值匹配结果

                # 根据性能确定色块样式
                block_class = "pure-color-block"
                if gt_match.get("is_excellent"):
                    block_class += " excellent"  # 优秀
                elif gt_match.get("is_acceptable"):
                    block_class += " acceptable"  # 可接受
                else:
                    block_class += " poor"  # 较差

                # 确定纯度样式
                purity_class = "purity-info"
                if purity_score >= 0.8:
                    purity_class += " high"  # 高纯度
                elif purity_score >= 0.6:
                    purity_class += " medium"  # 中等纯度
                else:
                    purity_class += " low"  # 低纯度

                # 颜色预览样式
                color_style = f"background-color: rgb({pure_rgb[0]}, {pure_rgb[1]}, {pure_rgb[2]});"

                html += f"""
                <div class="{block_class}">
                    <div class="color-preview" style="{color_style}"></div>
                    <div class="color-info">
                        <div class="pure-rgb-info">色块 {colorbar_id}.{block_id} (YOLO)</div>
                        <div class="pure-rgb-info">RGB({pure_rgb[0]}, {pure_rgb[1]}, {pure_rgb[2]})</div>
                        <div class="cmyk-values">
                            C={pure_cmyk[0]}% M={pure_cmyk[1]}%<br>
                            Y={pure_cmyk[2]}% K={pure_cmyk[3]}%
                        </div>
                        <div class="{purity_class}">
                            纯度: {purity_score:.2f} ({color_quality})
                        </div>
                """

                # 真值比较结果
                if gt_match.get("closest_color"):
                    delta_e = gt_match.get("delta_e", 0)  # Delta E值
                    accuracy_level = gt_match.get(
                        "accuracy_level", "Unknown"
                    )  # 准确度级别
                    gt_color = gt_match["closest_color"]  # 最接近的真值颜色

                    # Delta E样式设置
                    delta_e_class = "delta-e-info"
                    status_icon = ""
                    if gt_match.get("is_excellent"):
                        delta_e_class += " excellent"
                        status_icon = "✅"  # 优秀
                    elif gt_match.get("is_acceptable"):
                        delta_e_class += " good"
                        status_icon = "⚠️"  # 良好
                    else:
                        delta_e_class += " poor"
                        status_icon = "❌"  # 较差

                    html += f"""
                        <div class="{delta_e_class}">
                            <div class="delta-e-value">ΔE: {delta_e:.2f} <span class="status-indicator">{status_icon}</span></div>
                            <div class="accuracy-level">{accuracy_level}</div>
                            <div class="ground-truth-match">对比 {gt_color['name']}</div>
                        </div>
                    """

                html += """
                    </div>
                </div>
                """

            html += "</div>"
        else:
            html += f"<div style='text-align: center; color: #666; font-style: italic;'>颜色条 {colorbar_id} 中未检测到纯色块</div>"

        html += "</div>"

    return html


def create_pure_colorbar_analysis_interface():
    """创建基于双YOLO模型的纯色颜色条分析Gradio界面"""

    with gr.Row():
        with gr.Column():
            gr.Markdown("## 🎯 基于双YOLO的颜色条分析")
            gr.Markdown(
                "上传包含颜色条的图像进行**双YOLO模型分析**：\n"
                "- **第一步**: YOLO检测颜色条区域 (best0710.pt)\n"
                "- **第二步**: YOLO检测色块 (best.pt)\n"
                "- **第三步**: 精确的**CMYK匹配**和**delta E计算**"
            )

            # 图像上传组件
            input_image = gr.Image(label="📷 上传颜色条图像", type="pil", scale=2)

            # 参数设置折叠面板
            with gr.Accordion("🔧 双YOLO检测设置", open=False):
                gr.Markdown("**YOLO颜色条检测参数 (best0710.pt)**")
                with gr.Row():
                    confidence_threshold = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.6,
                        step=0.1,
                        label="颜色条置信度",
                        info="YOLO颜色条检测置信度阈值",
                    )
                    box_expansion = gr.Slider(
                        minimum=0,
                        maximum=50,
                        value=10,
                        step=5,
                        label="框扩展(像素)",
                        info="扩展检测到的颜色条框",
                    )

                gr.Markdown("**YOLO色块检测参数 (best.pt)**")
                with gr.Row():
                    block_confidence_threshold = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label="色块置信度",
                        info="YOLO色块检测置信度阈值",
                    )
                    min_block_area = gr.Slider(
                        minimum=10,
                        maximum=500,
                        value=50,
                        step=10,
                        label="最小色块面积",
                        info="接受的最小色块面积(像素)",
                    )

                gr.Markdown("**纯色分析参数**")
                with gr.Row():
                    purity_threshold = gr.Slider(
                        minimum=0.5,
                        maximum=1.0,
                        value=0.8,
                        step=0.05,
                        label="纯度阈值",
                        info="最小颜色纯度分数",
                    )

            # 操作按钮
            with gr.Row():
                analyze_btn = gr.Button("🚀 双YOLO分析", variant="primary", scale=2)
                clear_btn = gr.Button("🗑️ 清除", scale=1)

        with gr.Column():
            # 结果图像显示
            result_image = gr.Image(label="📊 分析结果", type="pil", scale=2)

            # 状态文本显示
            status_text = gr.Textbox(
                label="状态", value="上传 → 双YOLO分析", interactive=False, scale=1
            )

    # 结果显示区域
    with gr.Row():
        with gr.Column():
            results_display = gr.HTML(
                label="🎨 双YOLO分析结果",
                value="<div style='text-align: center; color: #666; padding: 20px;'>上传图像并点击'双YOLO分析'查看包含CMYK值和delta E比较的详细结果。<br><br><strong>检测流程:</strong><br>1. YOLO检测颜色条 → 2. YOLO检测色块 → 3. 纯色分析</div>",
            )

    # 真值参考区域
    with gr.Row():
        with gr.Column():
            gr.Markdown("## 📋 真值颜色参考")
            with gr.Row():
                show_reference_btn = gr.Button("📊 显示参考图表")
                show_yaml_btn = gr.Button("📝 显示YAML配置")

            # 参考图表显示
            reference_chart = gr.Image(label="真值参考图表", visible=False)

            # YAML配置显示
            yaml_config = gr.Code(label="真值YAML配置", language="yaml", visible=False)

    # 事件处理函数定义
    def run_analysis(img, conf, box_exp, block_conf, min_area, purity_thresh):
        """运行双YOLO分析的事件处理函数"""
        if img is None:
            return None, "❌ 请上传图像", ""

        return process_pure_colorbar_analysis(
            img,
            confidence_threshold=conf,
            box_expansion=box_exp,
            block_confidence_threshold=block_conf,
            min_block_area=min_area,
            purity_threshold=purity_thresh,
        )

    def clear_all():
        """清除所有输入和输出"""
        return None, None, "上传 → 双YOLO分析", ""

    def show_reference_chart():
        """显示参考图表"""
        try:
            reference_image = ground_truth_checker.generate_reference_chart()
            return gr.Image(value=reference_image, visible=True)
        except Exception as e:
            print(f"生成参考图表时出错: {e}")
            return gr.Image(visible=False)

    def show_yaml_config():
        """显示YAML配置"""
        try:
            yaml_content = ground_truth_checker.get_palette_yaml()
            return gr.Code(value=yaml_content, visible=True)
        except Exception as e:
            print(f"生成YAML配置时出错: {e}")
            return gr.Code(value="# 生成YAML配置时出错", visible=True)

    # 连接事件处理器
    analyze_btn.click(
        fn=run_analysis,
        inputs=[
            input_image,
            confidence_threshold,
            box_expansion,
            block_confidence_threshold,
            min_block_area,
            purity_threshold,
        ],
        outputs=[result_image, status_text, results_display],
    )

    clear_btn.click(
        fn=clear_all, outputs=[input_image, result_image, status_text, results_display]
    )

    show_reference_btn.click(
        fn=show_reference_chart,
        outputs=[reference_chart],
    )

    show_yaml_btn.click(
        fn=show_yaml_config,
        outputs=[yaml_config],
    )

    # 图像变化时更新状态
    input_image.change(
        fn=lambda img: "准备 → 双YOLO分析" if img else "上传 → 双YOLO分析",
        inputs=[input_image],
        outputs=[status_text],
    )

    # 返回主要组件的引用
    return {
        "input_image": input_image,
        "result_image": result_image,
        "status_text": status_text,
        "results_display": results_display,
    }
