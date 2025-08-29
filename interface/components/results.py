import gradio as gr


def create_results_ui():
    """Create simplified results UI without color checker integration"""

    with gr.Tabs():
        with gr.TabItem("Alignment Result"):
            with gr.Row():
                aligned_image = gr.Image(label="Aligned Image", height=400)
                diff_map = gr.Image(label="Difference Map", height=400)

        with gr.TabItem("Color Difference"):
            with gr.Row():
                heatmap = gr.Image(label="Color Difference Heatmap", height=350)
                heatmap_colorbar = gr.Image(label="Heatmap with Colorbar", height=350)

            with gr.Row():
                overlayed_heatmap = gr.Image(label="Overlaid Heatmap", height=350)
                highlighted = gr.Image(
                    label="High Color Difference Regions", height=350
                )

            with gr.Row():
                block_heatmap = gr.Image(
                    label="Block Color Difference Heatmap", height=350
                )
                overlay_blocks = gr.Image(label="Block Boundaries Overlay", height=350)

        with gr.TabItem("Summary"):
            composite = gr.Image(label="Comprehensive Analysis", height=500)

            with gr.Row():
                histogram = gr.Image(label="Color Difference Histogram", height=350)
                stats_chart = gr.Image(label="Statistical Summary", height=350)

            stats_display = gr.JSON(label="Statistical Data", show_label=True)

        with gr.TabItem("Method Comparison", visible=False) as comparison_tab:
            comparison_aligned = gr.Image(
                label="Alignment Results Comparison", height=400
            )
            comparison_heatmap = gr.Image(label="Heatmap Comparison", height=400)
            comparison_stats = gr.Image(label="Statistical Comparison", height=400)

        with gr.TabItem("ICC Conversion"):
            with gr.Row():
                icc_original = gr.Image(label="Original Image", height=350)
                icc_converted = gr.Image(label="ICC Converted Image", height=350)

            with gr.Row():
                icc_comparison = gr.Image(label="Before/After Comparison", height=300)
                icc_info = gr.JSON(label="ICC Conversion Info", show_label=True)

    return (
        aligned_image,
        diff_map,
        heatmap,
        heatmap_colorbar,
        overlayed_heatmap,
        highlighted,
        block_heatmap,
        overlay_blocks,
        composite,
        histogram,
        stats_chart,
        stats_display,
        comparison_tab,
        comparison_aligned,
        comparison_heatmap,
        comparison_stats,
        icc_original,
        icc_converted,
        icc_comparison,
        icc_info,
    )
