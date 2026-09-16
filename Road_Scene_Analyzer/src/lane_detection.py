import cv2
import numpy as np

from .edge_detection import canny_edges


def region_of_interest(edges):
    """
    Selects the lower region of the image
    where road/lane lines are expected.
    """

    height, width = edges.shape

    # Create black mask
    mask = np.zeros_like(edges)

    # Define trapezoidal road region
    polygon = np.array(
        [[
            (int(0.08 * width), height),

            (int(0.42 * width),
             int(0.58 * height)),

            (int(0.58 * width),
             int(0.58 * height)),

            (int(0.92 * width), height)
        ]],
        dtype=np.int32
    )

    # Fill polygon
    cv2.fillPoly(
        mask,
        polygon,
        255
    )

    # Keep only selected region
    masked_edges = cv2.bitwise_and(
        edges,
        mask
    )

    return masked_edges


def draw_line(image, line):
    """
    Draws a detected line on the image.
    """

    x1, y1, x2, y2 = line

    cv2.line(
        image,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (0, 255, 0),
        4
    )


def detect_lanes(image):
    """
    Detects prominent straight lines using:

    Canny Edge Detection
              +
    Hough Line Transform

    Returns:
        output image
        ROI edge image
        number of detected lines
    """

    # Check input
    if image is None or image.size == 0:
        raise ValueError("Empty image provided.")

    # Resize image
    resized = cv2.resize(
        image,
        (640, 360)
    )

    # ----------------------------------------------
    # STEP 1: CANNY EDGE DETECTION
    # ----------------------------------------------

    edges = canny_edges(
        resized
    )

    # ----------------------------------------------
    # STEP 2: REGION OF INTEREST
    # ----------------------------------------------

    roi = region_of_interest(
        edges
    )

    # ----------------------------------------------
    # STEP 3: HOUGH TRANSFORM
    # ----------------------------------------------

    lines = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=35,
        maxLineGap=80
    )

    output = resized.copy()

    line_count = 0

    # ----------------------------------------------
    # STEP 4: FILTER AND DRAW LINES
    # ----------------------------------------------

    if lines is not None:

        for line in lines:

            # Convert whatever shape OpenCV returned
            # into a flat array.
            coordinates = np.asarray(
                line
            ).reshape(-1)

            # Make sure four coordinates exist
            if len(coordinates) < 4:
                continue

            x1, y1, x2, y2 = coordinates[:4]

            # Avoid division by zero
            if x2 == x1:
                continue

            # Calculate slope
            slope = (
                (y2 - y1) /
                (x2 - x1)
            )

            # Keep reasonably lane-like lines
            if 0.35 < abs(slope) < 3.5:

                draw_line(
                    output,
                    (x1, y1, x2, y2)
                )

                line_count += 1

    return (
        output,
        roi,
        line_count
    )