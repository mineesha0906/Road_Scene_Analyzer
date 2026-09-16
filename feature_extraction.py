import cv2

from skimage.feature import hog


def extract_hog(image):
    """
    Extracts Histogram of Oriented Gradients (HOG)
    features from an image.

    Returns:
        HOG visualization
        Number of extracted features
    """

    # Check input
    if image is None or image.size == 0:
        raise ValueError(
            "Empty image provided."
        )

    # ----------------------------------------------
    # RESIZE
    # ----------------------------------------------

    resized = cv2.resize(
        image,
        (128, 64)
    )

    # ----------------------------------------------
    # GRAYSCALE
    # ----------------------------------------------

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    # ----------------------------------------------
    # HOG FEATURE EXTRACTION
    # ----------------------------------------------

    features, hog_image = hog(
        gray,

        orientations=9,

        pixels_per_cell=(
            8,
            8
        ),

        cells_per_block=(
            2,
            2
        ),

        block_norm="L2-Hys",

        visualize=True
    )

    # ----------------------------------------------
    # NORMALIZE HOG VISUALIZATION
    # ----------------------------------------------

    hog_image = hog_image.astype(
        "float32"
    )

    if hog_image.max() > hog_image.min():

        hog_image = cv2.normalize(
            hog_image,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

    hog_image = hog_image.astype(
        "uint8"
    )

    return (
        hog_image,
        len(features)
    )