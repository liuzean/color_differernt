import os

import yaml


# Load configuration file
def load_config(config_path="config.yaml"):
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    else:
        # Default configuration
        return {
            "detector": {"type": "surf", "nfeatures": 2000},
            "matcher": {
                "crossCheck": False,
                "ratio_test": True,
                "ratio_threshold": 0.7,
            },
            "alignment": {
                "ransac_reproj_threshold": 5.0,
                "min_matches": 4,
                "refinement": {
                    "enabled": True,
                    "nfeatures": 3000,
                    "crossCheck": True,
                    "ratio_threshold": 0.6,
                    "ransac_reproj_threshold": 3.0,
                    "min_matches": 8,
                },
            },
            "preprocessing": {
                "resize_max_dimension": 1500,
                "enhance_contrast": True,
                "clahe_clip_limit": 2.0,
                "clahe_grid_size": [8, 8],
            },
            "color_difference": {
                "method": "ciede2000",
                "threshold": 3.0,
                "mask_background": True,
            },
            "visualization": {
                "output_dir": "alignment_results",
                "save_intermediate": True,
                "generate_report": True,
                "compare_all_methods": False,
            },
            "blockwise": {
                "enabled": True,
                "block_size_h": 32,
                "block_size_w": 32,
                "use_mask": True,
            },
            "icc": {
                "enabled": False,
                "srgb_profile": "sRGB IEC61966-21.icc",
                "cmyk_profile": "JapanColor2001Coated.icc",
                "apply_conversion": False,
                "conversion_direction": "srgb_to_cmyk",  # or "cmyk_to_srgb"
                "save_converted": True,
                "rendering_intent": "perceptual",
            },
            "ui": {"theme": "soft", "show_descriptions": True},
        }


# Save configuration to file
def save_config(config, config_path="config.yaml"):
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
