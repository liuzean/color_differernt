"""
Test configuration and shared fixtures for pytest
"""

import os
import sys
from pathlib import Path

import pytest

# Add the project root directory to the Python path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))


# Fixtures for test resources
@pytest.fixture
def test_image_dir():
    """Returns the path to the test images directory."""
    return os.path.join(PROJECT_ROOT, "color-diff-images")


@pytest.fixture
def sample_image_path():
    """Returns the path to a sample test image."""
    image_dir = os.path.join(PROJECT_ROOT, "color-diff-images")
    sample_files = [
        f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    if sample_files:
        return os.path.join(image_dir, sample_files[0])
    pytest.skip("No sample images found for testing")


@pytest.fixture
def temp_output_dir(tmp_path):
    """Creates a temporary directory for test outputs."""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir()
    return output_dir
