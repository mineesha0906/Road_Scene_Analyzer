import cv2


def canny_edges(
    image,
    low_threshold=50,
    high_threshold=150
):
    """
    Performs Canny Edge Detection.

    Parameters:
        image: Input BGR image
        low_threshold: Lower threshold
        high_threshold: Upper threshold

    Returns:
        Binary edge image
    """

    # Check input
    if image is None or image.size == 0:
        raise ValueError("Empty image provided.")

    # Convert image to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Apply Gaussian blur
    # to reduce noise
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Apply Canny Edge Detection
    edges = cv2.Canny(
        blurred,
        low_threshold,
        high_threshold
    )

    return edges