import cv2
import numpy as np


def calculate_optical_flow(
    video_path,
    max_frames=80
):
    """
    Calculates dense Optical Flow using
    the Farneback method.

    Parameters:
        video_path: Path of input video
        max_frames: Maximum number of frames

    Returns:
        Optical Flow visualization
    """

    # ----------------------------------------------
    # OPEN VIDEO
    # ----------------------------------------------

    cap = cv2.VideoCapture(
        video_path
    )

    # ----------------------------------------------
    # READ FIRST FRAME
    # ----------------------------------------------

    ret, first_frame = cap.read()

    if not ret:

        cap.release()

        return None

    # Resize first frame
    first_frame = cv2.resize(
        first_frame,
        (640, 360)
    )

    # Convert first frame to grayscale
    previous_gray = cv2.cvtColor(
        first_frame,
        cv2.COLOR_BGR2GRAY
    )

    # ----------------------------------------------
    # HSV IMAGE FOR FLOW VISUALIZATION
    # ----------------------------------------------

    hsv = np.zeros_like(
        first_frame
    )

    # Maximum saturation
    hsv[..., 1] = 255

    final_output = (
        first_frame.copy()
    )

    frame_count = 1

    # ----------------------------------------------
    # PROCESS VIDEO FRAMES
    # ----------------------------------------------

    while frame_count < max_frames:

        ret, frame = cap.read()

        if not ret:
            break

        # Resize
        frame = cv2.resize(
            frame,
            (640, 360)
        )

        # Convert to grayscale
        current_gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # ------------------------------------------
        # FARNEBACK OPTICAL FLOW
        # ------------------------------------------

        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            current_gray,
            None,

            pyr_scale=0.5,

            levels=3,

            winsize=15,

            iterations=3,

            poly_n=5,

            poly_sigma=1.2,

            flags=0
        )

        # ------------------------------------------
        # FLOW MAGNITUDE + DIRECTION
        # ------------------------------------------

        magnitude, angle = cv2.cartToPolar(
            flow[..., 0],
            flow[..., 1]
        )

        # Direction
        hsv[..., 0] = (
            angle *
            180 /
            np.pi /
            2
        )

        # Magnitude
        hsv[..., 2] = cv2.normalize(
            magnitude,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        )

        # ------------------------------------------
        # CREATE FLOW VISUALIZATION
        # ------------------------------------------

        flow_visualization = cv2.cvtColor(
            hsv,
            cv2.COLOR_HSV2BGR
        )

        # Overlay flow on video frame
        final_output = cv2.addWeighted(
            frame,
            0.55,

            flow_visualization,
            0.45,

            0
        )

        # Current frame becomes previous frame
        previous_gray = current_gray

        frame_count += 1

    # Release video
    cap.release()

    return final_output