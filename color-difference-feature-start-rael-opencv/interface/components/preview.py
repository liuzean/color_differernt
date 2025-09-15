import cv2
import gradio as gr
import numpy as np
from PIL import Image

from core.color.icc_trans import cmyk_to_srgb_array, srgb_to_cmyk_array


def process_image_preview(image_file, color_space):
    """Process an image for preview based on selected color space."""
    if image_file is None:
        return None

    try:
        # Load image
        if isinstance(image_file, str):
            # If it's a file path
            image = Image.open(image_file)
        else:
            # If it's already a PIL Image or file object
            image = Image.open(image_file)

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # If sRGB (default), return as-is
        if color_space == "sRGB (default)":
            return image

        # Convert to numpy array for ICC transformation
        image_array = np.array(image)
        # Convert RGB to BGR for OpenCV
        bgr_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # Apply ICC transformation based on selected color space
        if "CMYK" in color_space or "Japan" in color_space:
            # Convert to CMYK and back to simulate the color space transformation
            cmyk_array, _ = srgb_to_cmyk_array(bgr_array)
            final_rgb_array, final_image = cmyk_to_srgb_array(cmyk_array)
            return final_image
        else:
            # For other color spaces, apply the transformation if available
            # This is a simplified version - in a full implementation,
            # you would use the specific ICC profile for the selected color space
            return image

    except Exception as e:
        print(f"Error processing image preview: {e}")
        return None


def update_template_preview(template_file, color_space):
    """Update template image preview based on color space selection."""
    return process_image_preview(template_file, color_space)


def update_target_preview(target_file, color_space):
    """Update target image preview based on color space selection."""
    return process_image_preview(target_file, color_space)


def update_preview(
    image_file, color_space, srgb_profile_name=None, cmyk_profile_name=None
):
    """Update a single image preview based on color space selection.

    This function is used by the GUI for individual preview updates.
    """
    return process_image_preview(image_file, color_space)


def update_both_previews(template_file, target_file, color_space):
    """Update both image previews when color space changes."""
    template_preview = process_image_preview(template_file, color_space)
    target_preview = process_image_preview(target_file, color_space)
    return template_preview, target_preview


def create_preview_ui():
    """Create the preview UI with ICC transformation functionality."""
    with gr.Row():
        with gr.Column(scale=1):
            template_file = gr.File(label="Template Image", file_types=["image"])
            target_file = gr.File(label="Target Image", file_types=["image"])

        with gr.Column(scale=1):
            template_preview = gr.Image(
                label="Template Preview", type="pil", height=200
            )
            target_preview = gr.Image(label="Target Preview", type="pil", height=200)

    return (
        template_file,
        target_file,
        template_preview,
        target_preview,
    )
