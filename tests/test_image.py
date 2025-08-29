"""
Tests for image processing functionality
"""

import cv2
import numpy as np
import pytest

from core.image.alignment import ImageAlignment
from core.image.features.detector import DetectorFactory
from core.image.features.matcher import MatcherFactory
from core.image.masking import MaskingProcessor
from core.image.preprocessor import ImagePreprocessor


@pytest.fixture
def sample_images():
    """Create synthetic test images for alignment testing"""
    # Create a base image
    base = np.zeros((300, 400, 3), dtype=np.uint8)

    # Draw some features (squares, circles)
    cv2.rectangle(base, (50, 50), (100, 100), (255, 0, 0), -1)
    cv2.rectangle(base, (200, 150), (250, 200), (0, 255, 0), -1)
    cv2.circle(base, (300, 200), 30, (0, 0, 255), -1)
    cv2.circle(base, (100, 200), 20, (255, 255, 0), -1)

    # Create a slightly transformed version (shifted)
    transform_matrix = np.float32([[1, 0, 10], [0, 1, 5]])
    transformed = cv2.warpAffine(base, transform_matrix, (400, 300))

    # Create mask
    mask = np.zeros((300, 400), dtype=np.uint8)
    cv2.rectangle(mask, (45, 45), (105, 105), 255, -1)
    cv2.rectangle(mask, (195, 145), (255, 205), 255, -1)
    cv2.circle(mask, (300, 200), 32, 255, -1)
    cv2.circle(mask, (100, 200), 22, 255, -1)

    return {
        "base": base,
        "transformed": transformed,
        "mask": mask,
        "shift_x": 10,
        "shift_y": 5,
    }


def test_detector_factory():
    """Test creation of different feature detectors"""
    # Test SIFT detector creation
    sift_detector = DetectorFactory.create("sift")
    assert sift_detector is not None

    # Test SURF detector creation
    surf_detector = DetectorFactory.create("surf")
    assert surf_detector is not None

    # Test ORB detector creation
    orb_detector = DetectorFactory.create("orb")
    assert orb_detector is not None

    # Test with parameters
    orb_with_params = DetectorFactory.create("orb", nfeatures=1000)
    assert orb_with_params is not None


def test_matcher_factory():
    """Test creation of feature matchers"""
    # Create matcher for SIFT detector
    sift_matcher = MatcherFactory.create_for_detector("sift")
    assert sift_matcher is not None

    # Create matcher for SURF detector
    surf_matcher = MatcherFactory.create_for_detector("surf")
    assert surf_matcher is not None

    # Create matcher with custom parameters
    custom_matcher = MatcherFactory.create_for_detector("orb", crossCheck=True)
    assert custom_matcher is not None


def test_image_preprocessor():
    """Test image preprocessing functionality"""
    preprocessor = ImagePreprocessor()
    preprocessor.max_dimension = 200
    preprocessor.enhance_contrast = True

    # Create test image
    test_img = np.ones((400, 600, 3), dtype=np.uint8) * 127

    # Test resizing
    resized = preprocessor.resize_image(test_img)
    assert max(resized.shape[0], resized.shape[1]) == 200

    # Test contrast enhancement
    enhanced = preprocessor.enhance_image(test_img)
    assert enhanced.shape == test_img.shape


def test_image_alignment(sample_images):
    """Test image alignment functionality"""
    base = sample_images["base"]
    transformed = sample_images["transformed"]
    expected_shift_x = sample_images["shift_x"]
    expected_shift_y = sample_images["shift_y"]

    # Create detector and matcher
    detector = DetectorFactory.create("orb", nfeatures=1000)
    matcher = MatcherFactory.create_for_detector("orb")

    # Create aligner
    aligner = ImageAlignment(detector=detector, matcher=matcher)

    # Align images
    result = aligner.align_images(
        base,
        transformed,
        ratio_test=True,
        ratio_threshold=0.75,
        ransac_reproj_threshold=5.0,
        min_matches=4,
    )

    # Check if alignment succeeded
    assert result.success
    assert result.aligned_image is not None

    # Verify alignment accuracy
    if result.homography is not None:
        # The homography should approximately reverse the shift
        assert abs(result.homography[0, 2] + expected_shift_x) < 3
        assert abs(result.homography[1, 2] + expected_shift_y) < 3


def test_masking_processor(sample_images, temp_output_dir):
    """Test background masking functionality"""
    image = sample_images["base"]
    mask = sample_images["mask"]

    # Test removing background
    result = MaskingProcessor.remove_background(image, mask)
    assert result.shape == image.shape

    # Masked areas should be preserved
    non_masked_area = image * (1 - mask[:, :, np.newaxis] / 255)
    assert np.sum(result[mask == 0]) < np.sum(non_masked_area)
