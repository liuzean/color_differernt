#!/usr/bin/env python

"""
ICC Profile Handling Module

Provides functionality for working with ICC color profiles, including
profile creation, loading, and color space transformations.
"""

import os
import struct
from pathlib import Path
from typing import Any

import numpy as np

from .utils import ColorSpace, convert_color


class ICCProfile:
    """
    Represents an ICC color profile

    This class provides basic ICC profile functionality for color management.
    It supports reading profile metadata and basic transformations.
    """

    def __init__(self, profile_path: str | None = None):
        """
        Initialize ICC profile

        Args:
            profile_path: Path to ICC profile file (.icc or .icm)
        """
        self.profile_path = profile_path
        self.header = {}
        self.tags = {}
        self.profile_data = None

        if profile_path and os.path.exists(profile_path):
            self.load_profile(profile_path)

    def load_profile(self, profile_path: str) -> None:
        """
        Load ICC profile from file

        Args:
            profile_path: Path to ICC profile file
        """
        self.profile_path = profile_path

        with open(profile_path, "rb") as f:
            self.profile_data = f.read()

        # Parse ICC profile header
        self._parse_header()
        self._parse_tags()

    def _parse_header(self) -> None:
        """Parse ICC profile header information"""
        if len(self.profile_data) < 128:
            raise ValueError("Invalid ICC profile: header too short")

        # ICC profile header is 128 bytes
        header_data = self.profile_data[:128]

        # Parse key header fields
        self.header = {
            "profile_size": struct.unpack(">I", header_data[0:4])[0],
            "cmm_type": header_data[4:8].decode("ascii", errors="ignore"),
            "version": struct.unpack(">I", header_data[8:12])[0],
            "device_class": header_data[12:16].decode("ascii", errors="ignore"),
            "color_space": header_data[16:20].decode("ascii", errors="ignore"),
            "pcs": header_data[20:24].decode(
                "ascii", errors="ignore"
            ),  # Profile Connection Space
            "platform": header_data[40:44].decode("ascii", errors="ignore"),
            "profile_flags": struct.unpack(">I", header_data[44:48])[0],
            "rendering_intent": struct.unpack(">I", header_data[64:68])[0],
        }

    def _parse_tags(self) -> None:
        """Parse ICC profile tag table"""
        if len(self.profile_data) < 132:
            return

        # Tag count is at offset 128
        tag_count = struct.unpack(">I", self.profile_data[128:132])[0]

        # Each tag entry is 12 bytes: signature (4) + offset (4) + size (4)
        for i in range(tag_count):
            offset = 132 + i * 12
            if offset + 12 > len(self.profile_data):
                break

            tag_signature = self.profile_data[offset : offset + 4].decode(
                "ascii", errors="ignore"
            )
            tag_offset = struct.unpack(
                ">I", self.profile_data[offset + 4 : offset + 8]
            )[0]
            tag_size = struct.unpack(">I", self.profile_data[offset + 8 : offset + 12])[
                0
            ]

            # Store tag information
            self.tags[tag_signature] = {
                "offset": tag_offset,
                "size": tag_size,
                "data": None,
            }

            # Load tag data if within bounds
            if tag_offset + tag_size <= len(self.profile_data):
                self.tags[tag_signature]["data"] = self.profile_data[
                    tag_offset : tag_offset + tag_size
                ]

    def get_profile_description(self) -> str:
        """
        Get profile description text

        Returns:
            Profile description string
        """
        if "desc" in self.tags and self.tags["desc"]["data"]:
            # Parse text description tag
            desc_data = self.tags["desc"]["data"]
            if len(desc_data) >= 8:
                # Skip tag signature and reserved bytes
                text_length = (
                    struct.unpack(">I", desc_data[8:12])[0]
                    if len(desc_data) >= 12
                    else 0
                )
                if text_length > 0 and len(desc_data) >= 12 + text_length:
                    return (
                        desc_data[12 : 12 + text_length]
                        .decode("ascii", errors="ignore")
                        .rstrip("\x00")
                    )

        return f"ICC Profile: {os.path.basename(self.profile_path) if self.profile_path else 'Unknown'}"

    def get_color_space(self) -> str:
        """
        Get profile color space

        Returns:
            Color space identifier
        """
        return self.header.get("color_space", "Unknown").strip("\x00")

    def get_profile_class(self) -> str:
        """
        Get profile device class

        Returns:
            Profile device class
        """
        return self.header.get("device_class", "Unknown").strip("\x00")

    def is_cmyk_profile(self) -> bool:
        """Check if this is a CMYK profile"""
        return self.get_color_space().lower() == "cmyk"

    def is_rgb_profile(self) -> bool:
        """Check if this is an RGB profile"""
        return self.get_color_space().lower() == "rgb"

    def get_white_point(self) -> tuple[float, float, float] | None:
        """
        Get profile white point in XYZ coordinates

        Returns:
            White point XYZ coordinates or None if not available
        """
        if "wtpt" in self.tags and self.tags["wtpt"]["data"]:
            wtpt_data = self.tags["wtpt"]["data"]
            if len(wtpt_data) >= 20:
                # XYZ values are stored as s15Fixed16Number (4 bytes each)
                x = struct.unpack(">I", wtpt_data[8:12])[0] / 65536.0
                y = struct.unpack(">I", wtpt_data[12:16])[0] / 65536.0
                z = struct.unpack(">I", wtpt_data[16:20])[0] / 65536.0
                return (x, y, z)

        return None

    def get_profile_info(self) -> dict[str, Any]:
        """
        Get comprehensive profile information

        Returns:
            Dictionary with profile metadata
        """
        return {
            "description": self.get_profile_description(),
            "color_space": self.get_color_space(),
            "profile_class": self.get_profile_class(),
            "version": self.header.get("version", 0),
            "size": self.header.get("profile_size", 0),
            "white_point": self.get_white_point(),
            "rendering_intent": self.header.get("rendering_intent", 0),
            "tag_count": len(self.tags),
            "tags": list(self.tags.keys()),
        }


class ICCTransform:
    """
    ICC-based color transformation

    Provides color transformations using ICC profiles for accurate
    color management between different color spaces and devices.
    """

    def __init__(self, input_profile: ICCProfile, output_profile: ICCProfile):
        """
        Initialize ICC transform

        Args:
            input_profile: Source ICC profile
            output_profile: Destination ICC profile
        """
        self.input_profile = input_profile
        self.output_profile = output_profile

        # Validate profiles
        if not input_profile.profile_data or not output_profile.profile_data:
            raise ValueError("Both input and output profiles must be loaded")

    def transform_image(
        self, image: np.ndarray, rendering_intent: int = 0
    ) -> np.ndarray:
        """
        Transform image using ICC profiles

        Args:
            image: Input image array
            rendering_intent: Rendering intent (0=perceptual, 1=relative, 2=saturation, 3=absolute)

        Returns:
            Transformed image array
        """
        # This is a simplified implementation
        # In a full ICC implementation, this would use the Color Management Module (CMM)
        # to perform the actual transformation through Profile Connection Space (PCS)

        input_space = self._get_color_space_enum(self.input_profile.get_color_space())
        output_space = self._get_color_space_enum(self.output_profile.get_color_space())

        if input_space and output_space:
            # Use the general color conversion function
            return convert_color(image, input_space, output_space)
        else:
            # Fallback: return original image
            return image.copy()

    def _get_color_space_enum(self, color_space_str: str) -> ColorSpace | None:
        """Convert ICC color space string to ColorSpace enum"""
        space_map = {
            "RGB ": ColorSpace.RGB,
            "CMYK": ColorSpace.CMYK,
            "Lab ": ColorSpace.LAB,
        }
        return space_map.get(color_space_str.upper(), None)


def load_standard_profiles() -> dict[str, ICCProfile]:
    """
    Load standard ICC profiles that come with the system

    Returns:
        Dictionary of loaded standard profiles
    """
    profiles = {}

    # Look for standard ICC profiles in common locations
    profile_dirs = [
        Path("core/color/icc"),  # Project ICC directory
        Path("/System/Library/ColorSync/Profiles"),  # macOS
        Path("/usr/share/color/icc"),  # Linux
        Path("C:/Windows/System32/spool/drivers/color"),  # Windows
    ]

    standard_profiles = {
        "sRGB": ["sRGB Profile.icc", "sRGB IEC61966-2-1.icc", "sRGB IEC61966-21.icc"],
        "Adobe RGB": ["Adobe RGB (1998).icc", "AdobeRGB1998.icc"],
        "ProPhoto RGB": ["ProPhoto.icc", "ProPhotoRGB.icc"],
        "Japan Color 2001": ["JapanColor2001Coated.icc", "Japan Color 2001 Coated.icc"],
        "US Web Coated SWOP": ["USWebCoatedSWOP.icc", "US Web Coated (SWOP) v2.icc"],
    }

    for profile_name, filenames in standard_profiles.items():
        for profile_dir in profile_dirs:
            if not profile_dir.exists():
                continue

            for filename in filenames:
                profile_path = profile_dir / filename
                if profile_path.exists():
                    try:
                        profiles[profile_name] = ICCProfile(str(profile_path))
                        break
                    except Exception as e:
                        print(f"Warning: Could not load profile {profile_path}: {e}")

            if profile_name in profiles:
                break

    return profiles


def create_simple_cmyk_profile(
    name: str = "Simple CMYK", description: str = "Basic CMYK Profile"
) -> ICCProfile:
    """
    Create a simple CMYK ICC profile

    This creates a basic CMYK profile for testing purposes.
    In a real implementation, this would create a proper ICC profile
    with complete color transformation tables.

    Args:
        name: Profile name
        description: Profile description

    Returns:
        Simple CMYK ICC profile
    """
    # This is a minimal implementation
    # A full ICC profile would require proper LUT creation
    # and color transformation tables

    profile = ICCProfile()
    profile.header = {
        "profile_size": 1024,
        "cmm_type": "ADBE",
        "version": 0x02400000,
        "device_class": "prtr",  # Printer profile
        "color_space": "CMYK",
        "pcs": "Lab ",  # Lab as Profile Connection Space
        "platform": "APPL",
        "profile_flags": 0,
        "rendering_intent": 0,
    }

    profile.tags = {
        "desc": {
            "offset": 128,
            "size": 100,
            "data": description.encode("ascii")[:96].ljust(100, b"\x00"),
        }
    }

    return profile


def get_profile_gamut_volume(profile: ICCProfile) -> float:
    """
    Estimate the gamut volume of an ICC profile

    Args:
        profile: ICC profile to analyze

    Returns:
        Estimated gamut volume (relative units)
    """
    # This is a simplified implementation
    # A full implementation would analyze the profile's LUT tables
    # to calculate the actual gamut volume

    color_space = profile.get_color_space()

    # Rough estimates based on typical color spaces
    gamut_estimates = {
        "RGB ": 1.0,  # sRGB baseline
        "CMYK": 0.7,  # Typical CMYK gamut
        "Lab ": 1.5,  # Large Lab gamut
        "XYZ ": 1.2,  # XYZ gamut
    }

    return gamut_estimates.get(color_space, 0.5)


def compare_profiles(profile1: ICCProfile, profile2: ICCProfile) -> dict[str, Any]:
    """
    Compare two ICC profiles

    Args:
        profile1: First profile
        profile2: Second profile

    Returns:
        Comparison results
    """
    return {
        "color_space_match": profile1.get_color_space() == profile2.get_color_space(),
        "class_match": profile1.get_profile_class() == profile2.get_profile_class(),
        "profile1_info": profile1.get_profile_info(),
        "profile2_info": profile2.get_profile_info(),
        "gamut_ratio": get_profile_gamut_volume(profile2)
        / max(get_profile_gamut_volume(profile1), 0.1),
    }


def validate_icc_profile(profile: ICCProfile) -> dict[str, Any]:
    """
    Validate ICC profile integrity

    Args:
        profile: ICC profile to validate

    Returns:
        Validation results
    """
    validation = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "info": profile.get_profile_info(),
    }

    # Check basic profile structure
    if not profile.profile_data:
        validation["is_valid"] = False
        validation["errors"].append("No profile data loaded")
        return validation

    if len(profile.profile_data) < 128:
        validation["is_valid"] = False
        validation["errors"].append("Profile data too short (missing header)")
        return validation

    # Check header consistency
    expected_size = profile.header.get("profile_size", 0)
    actual_size = len(profile.profile_data)

    if expected_size != actual_size:
        validation["warnings"].append(
            f"Profile size mismatch: header says {expected_size}, actual {actual_size}"
        )

    # Check required tags
    required_tags = ["desc", "cprt", "wtpt"]  # Description, Copyright, White Point
    for tag in required_tags:
        if tag not in profile.tags:
            validation["warnings"].append(f"Missing recommended tag: {tag}")

    # Check color space validity
    color_space = profile.get_color_space()
    valid_spaces = ["RGB ", "CMYK", "Lab "]
    if color_space not in valid_spaces:
        validation["warnings"].append(f"Unusual color space: {color_space}")

    return validation
