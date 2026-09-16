import streamlit as st
import cv2
import numpy as np

from PIL import Image

from src.preprocessing import preprocess_image
from src.edge_detection import canny_edges
from src.lane_detection import detect_lanes
from src.segmentation import kmeans_segment
from src.feature_extraction import extract_hog
from src.optical_flow import calculate_optical_flow


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Road Scene Analyzer",
    page_icon="🚗",
    layout="wide"
)


# ==================================================
# TITLE
# ==================================================

st.title(
    "🚗 Road Scene Analyzer"
)

st.write(
    """
    An interactive Computer Vision application
    demonstrating classical image-processing and
    computer-vision techniques.
    """
)

st.caption(
    "Canny • Hough Transform • K-Means • HOG • Optical Flow"
)


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header(
    "Select Analysis"
)

operation = st.sidebar.selectbox(
    "Choose an operation",

    [
        "Preprocessing",

        "Canny Edge Detection",

        "Lane Detection (Hough Transform)",

        "K-Means Segmentation",

        "HOG Feature Extraction",

        "Optical Flow (Video)"
    ]
)


# ==================================================
# K-MEANS CLUSTER CONTROL
# ==================================================

if operation == "K-Means Segmentation":

    k = st.sidebar.slider(
        "Number of Clusters (K)",

        min_value=2,

        max_value=6,

        value=4
    )


# ==================================================
# OPTICAL FLOW
# ==================================================

if operation == "Optical Flow (Video)":

    st.header(
        "🎥 Optical Flow Analysis"
    )

    st.write(
        """
        Upload a short video to visualize apparent
        motion between consecutive frames.
        """
    )

    video_file = st.file_uploader(
        "Upload Video",

        type=[
            "mp4",
            "avi",
            "mov"
        ]
    )

    if video_file is None:

        st.info(
            "Please upload a video."
        )

        st.stop()

    # Save video temporarily
    temp_video_path = (
        "temp_input_video.mp4"
    )

    with open(
        temp_video_path,
        "wb"
    ) as file:

        file.write(
            video_file.read()
        )

    # Process video
    with st.spinner(
        "Calculating Optical Flow..."
    ):

        result = calculate_optical_flow(
            temp_video_path,
            max_frames=80
        )

    # Display result
    if result is None:

        st.error(
            "Unable to process the video."
        )

    else:

        st.subheader(
            "Optical Flow Result"
        )

        st.image(
            cv2.cvtColor(
                result,
                cv2.COLOR_BGR2RGB
            ),

            use_container_width=True
        )

    st.stop()


# ==================================================
# IMAGE UPLOAD
# ==================================================

uploaded_file = st.file_uploader(
    "Upload an Image",

    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ==================================================
# INPUT VALIDATION
# ==================================================

if uploaded_file is None:

    st.warning(
        "Please upload an image to begin."
    )

    st.stop()


# ==================================================
# IMAGE CONVERSION
# ==================================================

try:

    # Read uploaded image
    pil_image = Image.open(
        uploaded_file
    ).convert(
        "RGB"
    )

    # PIL → NumPy
    rgb_image = np.array(
        pil_image
    )

    # RGB → BGR
    image_bgr = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2BGR
    )

except Exception as error:

    st.error(
        f"Unable to read image: {error}"
    )

    st.stop()


# ==================================================
# DISPLAY INPUT
# ==================================================

st.subheader(
    "📷 Input Image"
)

st.image(
    rgb_image,

    use_container_width=True
)


# ==================================================
# PROCESSING
# ==================================================

try:

    # =================================================
    # PREPROCESSING
    # =================================================

    if operation == "Preprocessing":

        st.header(
            "🔧 Image Preprocessing"
        )

        enhanced, gray, blurred = (
            preprocess_image(
                image_bgr
            )
        )

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                cv2.cvtColor(
                    enhanced,
                    cv2.COLOR_BGR2RGB
                ),

                caption="Enhanced Image",

                use_container_width=True
            )

        with col2:

            st.image(
                gray,

                caption="Grayscale Image",

                use_container_width=True
            )

        st.success(
            "Preprocessing completed successfully."
        )

        st.write(
            """
            Processing stages:

            1. Image resizing
            2. Grayscale conversion
            3. Gaussian filtering
            4. CLAHE-based contrast enhancement
            """
        )


    # =================================================
    # CANNY EDGE DETECTION
    # =================================================

    elif operation == "Canny Edge Detection":

        st.header(
            "🔍 Canny Edge Detection"
        )

        edges = canny_edges(
            image_bgr
        )

        st.image(
            edges,

            caption="Canny Edge Map",

            use_container_width=True
        )

        st.success(
            "Canny edge detection completed."
        )

        st.write(
            """
            Canny Edge Detection identifies strong
            intensity boundaries in the image.
            """
        )


    # =================================================
    # HOUGH TRANSFORM
    # =================================================

    elif operation == (
        "Lane Detection (Hough Transform)"
    ):

        st.header(
            "🛣️ Lane / Line Detection"
        )

        result, edge_map, line_count = (
            detect_lanes(
                image_bgr
            )
        )

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                edge_map,

                caption="ROI Edge Map",

                use_container_width=True
            )

        with col2:

            st.image(
                cv2.cvtColor(
                    result,
                    cv2.COLOR_BGR2RGB
                ),

                caption="Hough Line Detection",

                use_container_width=True
            )

        st.metric(
            "Detected Line Segments",

            line_count
        )

        st.write(
            """
            The system first performs Canny edge
            detection, selects a Region of Interest,
            and then applies the Probabilistic Hough
            Line Transform.
            """
        )


    # =================================================
    # K-MEANS SEGMENTATION
    # =================================================

    elif operation == "K-Means Segmentation":

        st.header(
            "🎨 K-Means Image Segmentation"
        )

        segmented = kmeans_segment(
            image_bgr,
            k=k
        )

        st.image(
            cv2.cvtColor(
                segmented,
                cv2.COLOR_BGR2RGB
            ),

            caption=f"K-Means Segmentation (K={k})",

            use_container_width=True
        )

        st.success(
            "Image segmentation completed."
        )

        st.write(
            f"""
            Each pixel is represented using its RGB
            values and assigned to one of {k} clusters.
            """
        )


    # =================================================
    # HOG FEATURE EXTRACTION
    # =================================================

    elif operation == "HOG Feature Extraction":

        st.header(
            "📊 HOG Feature Extraction"
        )

        hog_image, feature_count = (
            extract_hog(
                image_bgr
            )
        )

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                hog_image,

                caption="HOG Visualization",

                use_container_width=True
            )

        with col2:

            st.metric(
                "Feature Vector Length",

                feature_count
            )

            st.write(
                """
                Histogram of Oriented Gradients (HOG)
                represents local shape information using
                distributions of gradient orientations.
                """
            )


# ==================================================
# ERROR HANDLING
# ==================================================

except Exception as error:

    st.error(
        f"Processing failed: {error}"
    )