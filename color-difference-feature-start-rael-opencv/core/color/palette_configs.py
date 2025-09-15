"""
CMYK Color Palette Configurations
Simple YAML-based configuration for CMYK-only color checkers
"""

from typing import Any

import yaml

# Default CMYK palette configuration as YAML string
DEFAULT_CMYK_CONFIG = """
# CMYK Color Checker Configuration
# All colors are in CMYK percentages (0-100)

layout:
  swatch_size_mm: 5
  swatch_spacing_mm: 1
  group_spacing_mm: 3
  arrangement: single_column  # or 'grid'
  columns: 7  # for grid layout
  page_width_mm: 60
  page_height_mm: 550
  margin_mm: 20

gradients:
  - name: "Cyan Basic"
    start: [100, 0, 0, 0]  # [C, M, Y, K]
    end: [0, 0, 0, 0]
    steps: 7

  - name: "Magenta Basic"
    start: [0, 100, 0, 0]
    end: [0, 0, 0, 0]
    steps: 7

  - name: "Yellow Basic"
    start: [0, 0, 100, 0]
    end: [0, 0, 0, 0]
    steps: 7

  - name: "Black Basic"
    start: [0, 0, 0, 100]
    end: [0, 0, 0, 0]
    steps: 7

  - name: "Orange Basic"
    start: [0, 55, 100, 0]
    end: [0, 0, 0, 0]
    steps: 7

  - name: "Purple Basic"
    start: [80, 100, 0, 0]
    end: [0, 0, 0, 0]
    steps: 7

  - name: "Green Basic"
    start: [90, 0, 100, 0]
    end: [0, 0, 0, 0]
    steps: 7

  - name: "White Basic"
    start: [0, 0, 0, 0]
    end: [0, 0, 0, 0]
    steps: 7
"""

CMYK_7_COLOR_CONFIG = """
# CMYK 7-Color Configuration
# Extended version with 11 steps per gradient

layout:
  swatch_size_mm: 5
  swatch_spacing_mm: 1
  group_spacing_mm: 3
  arrangement: single_column
  columns: 7
  page_width_mm: 60
  page_height_mm: 550
  margin_mm: 20

gradients:
  - name: "Cyan Gradient"
    start: [100, 0, 0, 0]
    end: [0, 0, 0, 0]
    steps: 11

  - name: "Magenta Gradient"
    start: [0, 100, 0, 0]
    end: [0, 0, 0, 0]
    steps: 11

  - name: "Yellow Gradient"
    start: [0, 0, 100, 0]
    end: [0, 0, 0, 0]
    steps: 11

  - name: "Black Gradient"
    start: [0, 0, 0, 100]
    end: [0, 0, 0, 0]
    steps: 11

  - name: "Orange Gradient"
    start: [0, 55, 100, 0]
    end: [0, 0, 0, 0]
    steps: 11

  - name: "Green Gradient"
    start: [90, 0, 100, 0]
    end: [0, 0, 0, 0]
    steps: 11

  - name: "Purple Gradient"
    start: [80, 100, 0, 0]
    end: [0, 0, 0, 0]
    steps: 11
"""


class CMYKConfigManager:
    """Simple YAML-based CMYK configuration manager"""

    def __init__(self):
        self.presets = {
            "cmyk_basic": DEFAULT_CMYK_CONFIG,
            "cmyk_7_color": CMYK_7_COLOR_CONFIG,
        }
        self.current_config = None
        self.load_preset("cmyk_basic")

    def load_preset(self, preset_name: str) -> str:
        """Load a preset configuration and return YAML string"""
        if preset_name in self.presets:
            self.current_config = yaml.safe_load(self.presets[preset_name])
            return self.presets[preset_name]
        return self.presets["cmyk_basic"]

    def load_from_yaml(self, yaml_string: str) -> tuple[bool, str]:
        """Load configuration from YAML string. Returns (success, error_message)"""
        try:
            config = yaml.safe_load(yaml_string)
            validation_result = self.validate_config(config)
            if validation_result[0]:
                self.current_config = config
                return True, "Configuration loaded successfully"
            else:
                return False, validation_result[1]
        except yaml.YAMLError as e:
            return False, f"YAML parsing error: {str(e)}"

    def validate_config(self, config: dict[str, Any]) -> tuple[bool, str]:
        """Validate CMYK configuration. Returns (is_valid, error_message)"""
        try:
            # Check required structure
            if "gradients" not in config:
                return False, "Missing 'gradients' section"

            if "layout" not in config:
                return False, "Missing 'layout' section"

            # Validate each gradient
            for i, gradient in enumerate(config["gradients"]):
                if "name" not in gradient:
                    return False, f"Gradient {i + 1}: Missing 'name'"

                if "start" not in gradient or "end" not in gradient:
                    return (
                        False,
                        f"Gradient '{gradient.get('name', i + 1)}': Missing 'start' or 'end' colors",
                    )

                if "steps" not in gradient:
                    return (
                        False,
                        f"Gradient '{gradient.get('name', i + 1)}': Missing 'steps'",
                    )

                # Validate CMYK values
                for color_type in ["start", "end"]:
                    color = gradient[color_type]
                    if not isinstance(color, list) or len(color) != 4:
                        return (
                            False,
                            f"Gradient '{gradient['name']}': {color_type} color must be [C, M, Y, K] array",
                        )

                    for j, value in enumerate(color):
                        if not isinstance(value, int | float) or not (
                            0 <= value <= 100
                        ):
                            channel = ["C", "M", "Y", "K"][j]
                            return (
                                False,
                                f"Gradient '{gradient['name']}': {color_type} {channel} must be 0-100",
                            )

                # Validate steps
                steps = gradient["steps"]
                if not isinstance(steps, int) or not (2 <= steps <= 20):
                    return (
                        False,
                        f"Gradient '{gradient['name']}': steps must be integer 2-20",
                    )

            return True, "Configuration is valid"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def get_yaml_string(self) -> str:
        """Get current configuration as YAML string"""
        if self.current_config:
            return yaml.dump(
                self.current_config, default_flow_style=False, sort_keys=False
            )
        return self.presets["cmyk_basic"]

    def get_gradient_names(self) -> list[str]:
        """Get list of gradient names"""
        if not self.current_config:
            return []
        return [g["name"] for g in self.current_config.get("gradients", [])]

    def get_gradient(self, name: str) -> dict[str, Any]:
        """Get gradient configuration by name"""
        if not self.current_config:
            return {}

        for gradient in self.current_config.get("gradients", []):
            if gradient["name"] == name:
                return gradient
        return {}

    def update_gradient(
        self, name: str, start: list[float], end: list[float], steps: int
    ) -> bool:
        """Update a gradient configuration"""
        if not self.current_config:
            return False

        for gradient in self.current_config.get("gradients", []):
            if gradient["name"] == name:
                gradient["start"] = start
                gradient["end"] = end
                gradient["steps"] = steps
                return True
        return False

    def get_palette_data(self) -> list[dict[str, Any]]:
        """Convert current config to palette data for generator"""
        if not self.current_config:
            return []

        palette_data = []
        for gradient in self.current_config.get("gradients", []):
            palette_data.append(
                {
                    "name": gradient["name"],
                    "start_color": tuple(gradient["start"]),
                    "end_color": tuple(gradient["end"]),
                    "steps": gradient["steps"],
                }
            )

        return palette_data

    def get_layout_config(self) -> dict[str, Any]:
        """Get layout configuration"""
        if not self.current_config:
            return {}
        return self.current_config.get("layout", {})

    def get_preset_names(self) -> list[str]:
        """Get available preset names"""
        return list(self.presets.keys())
