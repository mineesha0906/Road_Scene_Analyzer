import cv2
import numpy as np

from sklearn.cluster import KMeans


def kmeans_segment(image, k=4):
    """
    Performs image segmentation using K-Means.

    Parameters:
        image: Input BGR image
        k: Number of clusters

    Returns:
        Segmented image
    """

    # Check input
    if image is None or image.size == 0:
        raise ValueError("Empty image provided.")

    # Check K
    if k < 2:
        raise ValueError(
            "Number of clusters must be at least 2."
        )

    # ----------------------------------------------
    # RESIZE
    # ----------------------------------------------

    resized = cv2.resize(
        image,
        (640, 360)
    )

    # ----------------------------------------------
    # CONVERT BGR TO RGB
    # ----------------------------------------------

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    # ----------------------------------------------
    # CONVERT IMAGE INTO PIXELS
    # ----------------------------------------------

    pixels = rgb.reshape(
        (-1, 3)
    ).astype(
        np.float32
    )

    # ----------------------------------------------
    # CREATE K-MEANS MODEL
    # ----------------------------------------------

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    # ----------------------------------------------
    # FIND CLUSTER FOR EACH PIXEL
    # ----------------------------------------------

    labels = model.fit_predict(
        pixels
    )

    # ----------------------------------------------
    # GET CLUSTER CENTERS
    # ----------------------------------------------

    centers = np.uint8(
        model.cluster_centers_
    )

    # ----------------------------------------------
    # REPLACE PIXELS WITH CLUSTER CENTERS
    # ----------------------------------------------

    segmented = centers[
        labels
    ].reshape(
        rgb.shape
    )

    # ----------------------------------------------
    # RGB → BGR
    # ----------------------------------------------

    segmented_bgr = cv2.cvtColor(
        segmented,
        cv2.COLOR_RGB2BGR
    )

    return segmented_bgr