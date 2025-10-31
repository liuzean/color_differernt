# interface/components/colorbar_analysis_tiff.py

"""
Gradio UI component for Colorbar Analysis (TIFF-based).
Final Plan Attempt: Uses gr.Image(type='filepath') for fast upload trigger
and relies on JS injection in gui.py to capture the original filename.
"""

import gradio as gr
from PIL import Image
import os
import traceback # 用于打印错误
import shutil
import tempfile

# 导入共享的结果显示函数
from .shared_results import update_shared_results_display
from core.block_detection.pure_colorbar_analysis_tiff import pure_colorbar_analysis_tiff_for_gradio

# --- 占位后台函数 ---
# 接收临时文件路径(filepath)和原始文件名(original_filename)


# --- 界面构建函数 (使用 JS 注入所需 ID) ---
def create_colorbar_analysis_tiff_ui():
    """Create the Gradio UI for TIFF-based analysis using JS filename capture."""

    with gr.Column():
        with gr.Row():
            # Input column
            with gr.Column(scale=1):
                # 【关键修改】使用 gr.Image，设置 type='filepath' 和 elem_id
                input_image = gr.Image(
                    label="📁 Upload Preview Image (TIFF)",
                    type="filepath", # 后端接收临时文件路径
                    height=250,
                    sources=["upload"],
                    elem_id="tiff_input_image" # 给 JS 一个明确的 ID 来查找 input 元素
                )

                # 【关键修改】隐藏的 Textbox 用于接收 JS 写入的文件名，设置 elem_id
                hidden_original_filename = gr.Textbox(
                    label="Original Filename (Hidden)",
                    value="", # 初始值为空
                    visible=False, # 界面上不可见
                    elem_id="hidden_orig_filename" # 给 JS 一个明确的 ID 来查找 textarea 元素
                )

                # 增加一个 gr.State 用于存储临时文件路径，方便清理
                temp_filepath_state = gr.State(value=None)

                # 当 input_image 的值（即 filepath）变化时，更新 state
                input_image.change(lambda x: x, inputs=input_image, outputs=temp_filepath_state)


                # 参数设置 (保持不变)
                with gr.Accordion("⚙️ Parameters", open=False):
                    gr.Markdown("Colorbar Detection (YOLO)")
                    with gr.Row():
                        confidence_threshold = gr.Slider(0.1, 1.0, 0.5, step=0.05, label="Confidence")
                        box_expansion = gr.Slider(0, 50, 10, step=1, label="Expansion")
                    gr.Markdown("Block Detection (YOLO)")
                    with gr.Row():
                        yolo_block_confidence = gr.Slider(0.1, 1.0, 0.5, step=0.05, label="Block Confidence")
                        block_area_threshold = gr.Slider(10, 200, 50, step=5, label="Min Area")
                    gr.Markdown("Color Analysis (from TIFF)")
                    with gr.Row():
                        purity_threshold = gr.Slider(0.5, 1.0, 0.8, step=0.05, label="Purity (Reference Only)")

                # Buttons (保持不变)
                with gr.Row():
                    analyze_btn = gr.Button("🚀 Analyze TIFF", variant="primary", scale=2)
                    clear_btn = gr.Button("🧹 Clear", scale=1)

                status_text = gr.Textbox(
                    label="Status", value="Upload Preview → Analyze TIFF", interactive=False, lines=1
                )

            # Results column (保持不变)
            with gr.Column(scale=2):
                # result_image 的标签调整回 Results
                result_image = gr.Image(label="🎯 Results", type="pil", height=250) 

        # Full-width results display (保持不变)
        results_display = gr.HTML(
            value="<div style='text-align: center; padding: 15px; color: #666; background: #f9f9f9; border-radius: 6px;'>📷 Upload preview and analyze TIFF to see results</div>"
        )

    # --- Event Handlers ---

    # Analyze button click handler
# Analyze button click handler (FINAL VERSION)
    def run_tiff_analysis_wrapper(
        preview_filepath: str | None, # 这是 gr.Image 传来的临时路径
        original_filename: str, # 这是隐藏 Textbox 传来的原始文件名
        conf_thresh: float,
        box_exp: float,
        yolo_block_conf: float,
        block_area: float,
        purity_thresh: float,
    ) -> tuple[Image.Image | None, str, str]:
        """
        Wrapper function that calls the real TIFF backend, 
        processes results, and generates HTML.
        """
        # --- 1. 输入验证 ---
        if not preview_filepath or not os.path.exists(preview_filepath):
             error_msg = "❌ Error: No preview image file path received. Please upload again."
             print(error_msg)
             return None, error_msg, f"<div style='color: red;'>{error_msg}</div>"
             
        if not original_filename:
             error_msg = "❌ Error: Original filename not captured. Check JS or upload again."
             print(error_msg)
             preview_pil = None
             try: 
                 preview_pil = Image.open(preview_filepath)
             except Exception: 
                 pass # 尝试加载预览图
             return preview_pil, error_msg, f"<div style='color: red;'>{error_msg} Check browser console (F12).</div>"

        try:
            # --- 2. 调用真正的后台函数 ---
            # (这是我们上一步在 pure_colorbar_analysis_tiff.py 中创建的)
            (
                annotated_pil,    # 标注过的 8-bit PIL 图像
                colorbar_data,    # 包含颜色分析结果的列表 (用于 HTML)
                report,           # 包含摘要的纯文本报告
                total_blocks
            ) = pure_colorbar_analysis_tiff_for_gradio(
                preview_filepath,
                original_filename,
                confidence_threshold=conf_thresh,
                box_expansion=box_exp,
                yolo_block_confidence=yolo_block_conf,
                block_area_threshold=block_area,
                purity_threshold=purity_thresh,
            )

            # --- 3. 处理后台返回的错误 ---
            if "Error" in report or "failed" in report:
                 # 如果后台返回了错误信息，将其显示出来
                 print(f"Backend returned error: {report}")
                 return annotated_pil, report, f"<div style='color: red; white-space: pre-wrap;'>{report}</div>"

            # --- 4. (关键) 调用 shared_results.py 生成 HTML ---
            # 我们复用现有的 HTML 生成函数
            results_html = update_shared_results_display(colorbar_data)
            
            # --- 5. 生成成功的状态消息 ---
            # 从报告字符串中提取摘要部分
            summary_report = "Analysis complete."
            if "📊 Overall Summary:" in report:
                 summary_report = "📊 Overall Summary:" + report.split("📊 Overall Summary:")[1].strip()

            status_message = f"✅ TIFF Analysis complete: {len(colorbar_data)} colorbar(s), {total_blocks} blocks found.\n{summary_report}"

            # 返回最终结果给 Gradio 界面
            # 对应 outputs=[result_image, status_text, results_display]
            return annotated_pil, status_message, results_html

        except Exception as e:
            # 捕获前端包装函数中可能发生的意外错误
            print(f"--- UNCAUGHT ERROR in run_tiff_analysis_wrapper ---")
            traceback.print_exc()
            print(f"---------------------------------------------------")
            error_msg = f"❌ Unhandled Error in UI layer: {str(e)}"
            return None, error_msg, f"<div style='color: red;'>{error_msg}</div>"

    # Clear button click handler
    # 【关键修改】增加清理临时文件的逻辑
    def clear_all_tiff_and_temps(current_temp_filepath):
        if current_temp_filepath and os.path.exists(current_temp_filepath):
            try:
                temp_dir_norm = os.path.normpath(os.path.dirname(current_temp_filepath))
                system_temp_norm = os.path.normpath(tempfile.gettempdir())
                gradio_base_norm = os.path.normpath(os.path.join(system_temp_norm, 'gradio'))

                # 安全检查
                if os.path.dirname(temp_dir_norm) == gradio_base_norm or \
                  (os.path.exists(gradio_base_norm) and os.path.samefile(os.path.dirname(temp_dir_norm), gradio_base_norm)): # 增加 exists 检查

                    shutil.rmtree(temp_dir_norm)
                    print(f"Clear button successfully deleted temporary folder: {temp_dir_norm}")
                else:
                    print(f"Clear button safety check failed: Refusing to delete: {temp_dir_norm}.")
            except Exception as e:
                print(f"Clear button error deleting temporary folder {temp_dir_norm}: {e}")
        else:
            print("Clear button: No temporary file path known, skipping deletion.")

        # 返回清除界面所需的值
        return (
            None, # 清空 Image 组件
            "",   # 清空隐藏 Textbox 的值
            None, # 清空 State
            None, # 清空结果图
            "Upload Preview → Analyze TIFF", # 重置状态文本
            "<div style='text-align: center; padding: 15px; color: #666; background: #f9f9f9; border-radius: 6px;'>📷 Upload preview and analyze TIFF</div>", # 重置结果 HTML
        )

    # Wire up events
    analyze_btn.click(
        fn=run_tiff_analysis_wrapper, # 调用包装函数
        inputs=[
            input_image,              # 提供 filepath
            hidden_original_filename, # 提供原始文件名
            confidence_threshold,
            box_expansion,
            yolo_block_confidence,
            block_area_threshold,
            purity_threshold,
        ],
        outputs=[result_image, status_text, results_display],
    )

    clear_btn.click(
        fn=clear_all_tiff_and_temps,
        inputs=[temp_filepath_state], # 传入当前临时文件路径
        outputs=[
            input_image,
            hidden_original_filename,
            temp_filepath_state, # 清空 State
            result_image,
            status_text,
            results_display
        ]
    )

    # 当图片上传时，更新状态提示 (保持不变)
    input_image.change(
        fn=lambda fp: "Ready → Analyze TIFF" if fp else "Upload Preview → Analyze TIFF",
        inputs=[input_image],
        outputs=[status_text],
    )

    # 返回 Gradio 组件列表 (包括新 state)
    return (
        input_image,
        hidden_original_filename,
        temp_filepath_state, # 返回 State
        result_image,
        status_text,
        results_display,
        confidence_threshold,
        box_expansion,
        yolo_block_confidence,
        block_area_threshold,
        purity_threshold,
        analyze_btn,
        clear_btn,
    )