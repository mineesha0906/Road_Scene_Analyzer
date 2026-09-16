import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)
import cv2
import numpy as np

from src.preprocessing import preprocess_image
from src.edge_detection import canny_edges
from src.lane_detection import detect_lanes
from src.segmentation import kmeans_segment
from src.feature_extraction import extract_hog


# ==================================================
# CREATE TEST IMAGE
# ==================================================

def create_test_image():

    # Create black image
    image = np.zeros(
        (360, 640, 3),
        dtype=np.uint8
    )

    # Create road-like region
    cv2.rectangle(
        image,

        (100, 200),

        (540, 359),

        (100, 100, 100),

        -1
    )

    # Left lane
    cv2.line(
        image,

        (250, 350),

        (310, 200),

        (255, 255, 255),

        5
    )

    # Right lane
    cv2.line(
        image,

        (390, 350),

        (330, 200),

        (255, 255, 255),

        5
    )

    return image


# ==================================================
# TEST PREPROCESSING
# ==================================================

def test_preprocessing():

    image = create_test_image()

    result, gray, blurred = (
        preprocess_image(
            image
        )
    )

    assert result.shape == (
        360,
        640,
        3
    )

    assert gray.shape == (
        360,
        640
    )


# ==================================================
# TEST CANNY
# ==================================================

def test_canny():

    image = create_test_image()

    edges = canny_edges(
        image
    )

    assert edges.shape == (
        360,
        640
    )


# ==================================================
# TEST HOUGH
# ==================================================

def test_lane_detection():

    image = create_test_image()

    result, edge_map, count = (
        detect_lanes(
            image
        )
    )

    assert result.shape == (
        360,
        640,
        3
    )

    assert edge_map.shape == (
        360,
        640
    )

    assert count >= 0


# ==================================================
# TEST K-MEANS
# ==================================================

def test_kmeans():

    image = create_test_image()

    result = kmeans_segment(
        image,
        k=3
    )

    assert result.shape == (
        360,
        640,
        3
    )


# ==================================================
# TEST HOG
# ==================================================

def test_hog():

    image = create_test_image()

    hog_image, feature_count = (
        extract_hog(
            image
        )
    )

    assert hog_image.shape == (
        64,
        128
    )

    assert feature_count > 0