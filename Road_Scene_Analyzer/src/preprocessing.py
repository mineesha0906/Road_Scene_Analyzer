import cv2


def preprocess_image(image):
    """
    Performs basic image preprocessing.

    Steps:
    1. Resize image
    2. Convert to grayscale
    3. Apply Gaussian filtering
    4. Apply CLAHE for contrast enhancement

    Parameters:
        image: Input image in BGR format

    Returns:
        enhanced: Enhanced image
        gray: Grayscale image
        blurred: Gaussian blurred image
    """

    # Check whether an image was actually provided
    if image is None or image.size == 0:
        raise ValueError("Empty image provided.")

    # ------------------------------------------------
    # STEP 1: RESIZE
    # ------------------------------------------------

    image = cv2.resize(
        image,
        (640, 360)
    )

    # ------------------------------------------------
    # STEP 2: GRAYSCALE CONVERSION
    # ------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # ------------------------------------------------
    # STEP 3: GAUSSIAN FILTERING
    # ------------------------------------------------

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # ------------------------------------------------
    # STEP 4: HISTOGRAM ENHANCEMENT
    # ------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced_gray = clahe.apply(
        blurred
    )

    # Convert enhanced grayscale image
    # back to BGR
    enhanced = cv2.cvtColor(
        enhanced_gray,
        cv2.COLOR_GRAY2BGR
    )

    return enhanced, gray, blurred