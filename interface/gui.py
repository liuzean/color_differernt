#!/usr/bin/env python

"""
Color Difference Analysis Gradio Interface - Simplified
"""

import functools

import gradio as gr
import matplotlib as plt


from .components.color_checker import create_color_checker_ui
from .components.colorbar_analysis import create_colorbar_analysis_ui
from .components.ground_truth_colorbar_demo import create_ground_truth_colorbar_demo_ui
from .components.preview import create_preview_ui, update_preview
from .components.results import create_results_ui
from .components.settings import create_settings_ui
from .config import load_config
from .handlers.callbacks import process_images_handler, save_config_handler
from .components.colorbar_analysis_tiff import create_colorbar_analysis_tiff_ui

plt.use("Agg")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

config = load_config()

js_code = """
async () => {
  // Function to add listener safely
  const addFileListener = () => {
      // Get file input element
      const imageInputContainer = document.getElementById('tiff_input_image');
      let fileInput = imageInputContainer ? imageInputContainer.querySelector('input[type="file"]') : null;

      // Get hidden textbox element
      const hiddenTextboxContainer = document.getElementById('hidden_orig_filename');
      let hiddenTextarea = hiddenTextboxContainer ? hiddenTextboxContainer.querySelector('textarea') : null;

      if (fileInput && hiddenTextarea) {
          console.log("JS: Found elements, adding event listener.");

          // Prevent adding multiple listeners if UI re-renders partially
          if (fileInput.dataset.listenerAttached === 'true') {
              console.log("JS: Listener already attached.");
              return;
          }
          fileInput.dataset.listenerAttached = 'true'; // Mark as attached

          fileInput.addEventListener('change', (event) => {
              const files = event.target.files;
              let filename = ''; // Default to empty
              if (files && files.length > 0) {
                  filename = files[0].name;
                  console.log('JS: File selected:', filename);
              } else {
                  console.log('JS: File input cleared or no file selected.');
              }

              // Update hidden textarea only if the value changes
              if (hiddenTextarea.value !== filename) {
                  console.log('JS: Updating hidden filename to:', filename);
                  hiddenTextarea.value = filename;
                  // Manually trigger input event for Gradio backend
                  const inputEvent = new Event('input', { bubbles: true });
                  hiddenTextarea.dispatchEvent(inputEvent);
                  console.log('JS: Dispatched input event for hidden textarea.');
              }
          });
          console.log("JS: Event listener added successfully.");
      } else {
          if (!fileInput) console.error("JS Error: File input for 'tiff_input_image' not found during listener setup.");
          if (!hiddenTextarea) console.error("JS Error: Hidden textarea for 'hidden_orig_filename' not found during listener setup.");
          // Retry after a short delay if elements aren't ready immediately
          console.log("JS: Elements not found, retrying listener setup...");
          setTimeout(addFileListener, 500); // Retry after 500ms
      }
  };

  // Initial attempt to add the listener
  addFileListener();

  // Fallback: Use MutationObserver to re-apply if Gradio re-renders components
  const observer = new MutationObserver((mutationsList, observer) => {
      for(const mutation of mutationsList) {
          if (mutation.type === 'childList' || mutation.type === 'attributes') {
              // Check if the target elements might have been re-rendered
              const imageInputContainer = document.getElementById('tiff_input_image');
              const fileInput = imageInputContainer ? imageInputContainer.querySelector('input[type="file"]') : null;
               if (fileInput && fileInput.dataset.listenerAttached !== 'true') {
                   console.log("JS Observer: Detected potential re-render, re-attaching listener.");
                   addFileListener(); // Re-run setup if listener seems missing
                   // Consider disconnecting observer if setup is stable: observer.disconnect();
                   break; // Assume setup is done once listener is re-attached
               }
          }
      }
  });

  // Observe the body for changes, might need refinement based on Gradio's structure
  observer.observe(document.body, { childList: true, subtree: true, attributes: true });
  console.log("JS: MutationObserver started.");

}
"""

def create_interface():
    """Create the main Gradio interface."""
    with gr.Blocks(title="Color Difference Analysis Tool",js=js_code) as demo:
        gr.Markdown("# Color Difference Analysis Tool")

        with gr.Tabs():
            # Intelligent Colorbar Analysis Tab
            with gr.TabItem("🎯 Colorbar Analysis"):
                create_colorbar_analysis_ui()
          
            with gr.TabItem("🎨 Colorbar Analysis_2 (TIFF)"):
                create_colorbar_analysis_tiff_ui()

            # Ground-Truth Colorbar Demo Tab
            with gr.TabItem("🎨 Ground-Truth Demo"):
                create_ground_truth_colorbar_demo_ui()

            # CMYK Color Checker Tab
            with gr.TabItem("CMYK Color Checker"):
                create_color_checker_ui()

            # Main Analysis Tab
            with gr.TabItem("Analysis"):
                with gr.Row():
                    with gr.Column(scale=2):
                        (
                            template_file,
                            target_file,
                            template_preview,
                            target_preview,
                        ) = create_preview_ui()

                    with gr.Column(scale=1):
                        process_btn = gr.Button("Start Analysis", variant="primary")
                        save_btn = gr.Button("Save Settings", variant="secondary")

                        result_text = gr.Textbox(label="Status", lines=2)
                        avg_delta_e = gr.Number(label="Average ΔE")
                        progress = gr.Textbox(label="Progress", interactive=False)

                # Results section - simplified
                (
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
                ) = create_results_ui()

                # Settings section - simplified
                color_space_preview, settings_components = create_settings_ui(config)
                # color_space_preview = settings_components[0]  # Extract color_space_preview separately

        # Set up callbacks for main analysis
        # color_space_preview = settings_components[0]  # First setting component

        template_file.change(
            fn=functools.partial(
                update_preview,
                srgb_profile_name=config.get("icc", {}).get(
                    "srgb_profile", "sRGB IEC61966-21.icc"
                ),
                cmyk_profile_name=config.get("icc", {}).get(
                    "cmyk_profile", "JapanColor2001Coated.icc"
                ),
            ),
            inputs=[template_file, color_space_preview],
            outputs=[template_preview],
        )

        target_file.change(
            fn=functools.partial(
                update_preview,
                srgb_profile_name=config.get("icc", {}).get(
                    "srgb_profile", "sRGB IEC61966-21.icc"
                ),
                cmyk_profile_name=config.get("icc", {}).get(
                    "cmyk_profile", "JapanColor2001Coated.icc"
                ),
            ),
            inputs=[target_file, color_space_preview],
            outputs=[target_preview],
        )

        color_space_preview.change(
            fn=functools.partial(
                update_preview,
                srgb_profile_name=config.get("icc", {}).get(
                    "srgb_profile", "sRGB IEC61966-21.icc"
                ),
                cmyk_profile_name=config.get("icc", {}).get(
                    "cmyk_profile", "JapanColor2001Coated.icc"
                ),
            ),
            inputs=[template_file, color_space_preview],
            outputs=[template_preview],
        )

        color_space_preview.change(
            fn=functools.partial(
                update_preview,
                srgb_profile_name=config.get("icc", {}).get(
                    "srgb_profile", "sRGB IEC61966-21.icc"
                ),
                cmyk_profile_name=config.get("icc", {}).get(
                    "cmyk_profile", "JapanColor2001Coated.icc"
                ),
            ),
            inputs=[target_file, color_space_preview],
            outputs=[target_preview],
        )

        # Process button callback
        process_btn.click(
            fn=process_images_handler,
            inputs=[
                template_file,
                target_file,
                *settings_components,  # Unpack the processing components tuple
            ],
            outputs=[
                result_text,
                avg_delta_e,
                progress,
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
                comparison_aligned,
                comparison_heatmap,
                comparison_stats,
                icc_original,
                icc_converted,
                icc_comparison,
                icc_info,
            ],
        )

        # Save config callback
        save_btn.click(
            fn=save_config_handler,
            inputs=settings_components,  # Use the processing components tuple
            outputs=[result_text],
        )

    return demo


def launch_interface():
    """Launch the Gradio interface."""
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True,
        show_api=False,
    )
