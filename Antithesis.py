import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from Basically import BasicOperations
from Colour_Change import ColorShifter
from DefinitelyEnhance import ImageSmoother
from Edgy import EdgeDetector
from Featuring import ImageSharpener
from Granuling import Segmenter
from Histogramming import HistogramProcessor
from InversionConversion import ImageTransformer

def main():
    st.title("🖼️ Image Processing Suite")
    
    # Initialize session state
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None
    if 'preview_image' not in st.session_state:
        st.session_state.preview_image = None
    if 'show_histogram' not in st.session_state:
        st.session_state.show_histogram = None
    
    # File upload
    uploaded_file = st.file_uploader("Choose an image", 
                                   type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Initialize image processor
        processor = BasicOperations(uploaded_file)
        
        # Store images in session state
        if st.session_state.original_image is None:
            st.session_state.original_image = processor.image.copy()
            st.session_state.current_image = processor.image.copy()
            
        # Display current image
        st.image(st.session_state.current_image,
                channels="BGR" if len(st.session_state.current_image.shape)==3 else "GRAY",
                use_column_width=True,
                caption="Current Image")
            
        # Sidebar controls
        operation = st.sidebar.selectbox(
            "Select Operation",
            ["None", "Color Shift", "Smooth", "Edge Detect",
             "Sharpen", "Segmentation", "Histogram",
             "Inversion/Transform", "Reset"]
        )
        
        # Operation parameters and apply buttons
        if operation == "Color Shift":
            r_shift = st.sidebar.slider("Red Shift", 0, 255, 0)
            g_shift = st.sidebar.slider("Green Shift", 0, 255, 0)
            b_shift = st.sidebar.slider("Blue Shift", 0, 255, 0)
            
            if st.sidebar.button("Preview"):
                st.session_state.preview_image = ColorShifter.shift_channels(
                    st.session_state.current_image.copy(),
                    r_shift,
                    g_shift,
                    b_shift
                )
            
            if st.sidebar.button("Apply"):
                st.session_state.current_image = ColorShifter.shift_channels(
                    st.session_state.current_image,
                    r_shift,
                    g_shift,
                    b_shift
                )
                st.session_state.preview_image = None
                st.sidebar.success("Color shift applied!")
                
        elif operation == "Smooth":
            kernel_size = st.sidebar.slider("Kernel Size", 1, 15, 3, step=2)
            
            if st.sidebar.button("Preview"):
                st.session_state.preview_image = cv2.GaussianBlur(
                    st.session_state.current_image.copy(),
                    (kernel_size, kernel_size),
                    0
                )
            
            if st.sidebar.button("Apply"):
                st.session_state.current_image = cv2.GaussianBlur(
                    st.session_state.current_image,
                    (kernel_size, kernel_size),
                    0
                )
                st.session_state.preview_image = None
                st.sidebar.success("Smoothing applied!")
                
        elif operation == "Edge Detect":
            threshold1 = st.sidebar.slider("Threshold 1", 0, 255, 100)
            threshold2 = st.sidebar.slider("Threshold 2", 0, 255, 200)
            
            if st.sidebar.button("Preview"):
                st.session_state.preview_image = cv2.Canny(
                    st.session_state.current_image.copy(),
                    threshold1,
                    threshold2
                )
            
            if st.sidebar.button("Apply"):
                st.session_state.current_image = cv2.Canny(
                    st.session_state.current_image,
                    threshold1,
                    threshold2
                )
                st.session_state.preview_image = None
                st.sidebar.success("Edge detection applied!")
                
        elif operation == "Sharpen":
            strength = st.sidebar.slider("Sharpening Strength", 0.1, 3.0, 1.0)
            
            if st.sidebar.button("Preview"):
                st.session_state.preview_image = ImageSharpener.sharpen(
                    st.session_state.current_image.copy(),
                    strength
                )
            
            if st.sidebar.button("Apply"):
                st.session_state.current_image = ImageSharpener.sharpen(
                    st.session_state.current_image,
                    strength
                )
                st.session_state.preview_image = None
                st.sidebar.success("Sharpening applied!")
                
        elif operation == "Segmentation":
            seg_mode = st.sidebar.selectbox(
                "Segmentation Mode",
                ["binary", "otsu"]
            )
            
            if st.sidebar.button("Preview"):
                st.session_state.preview_image = Segmenter.threshold_segmentation(
                    st.session_state.current_image.copy(),
                    seg_mode
                )
            
            if st.sidebar.button("Apply"):
                st.session_state.current_image = Segmenter.threshold_segmentation(
                    st.session_state.current_image,
                    seg_mode
                )
                st.session_state.preview_image = None
                st.sidebar.success("Segmentation applied!")
                
        elif operation == "Histogram":
            st.sidebar.markdown("### Histogram Options")
            
            # Create two columns for the buttons
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                if st.button("Original"):
                    st.session_state.show_histogram = "original"
            
            with col2:
                if st.button("Equalized"):
                    st.session_state.show_histogram = "equalized"
            
            # Display the appropriate histogram
            if st.session_state.show_histogram == "original":
                st.subheader("Original Histogram")
                fig = HistogramProcessor.plot_histogram(st.session_state.current_image)
                st.pyplot(fig)
            elif st.session_state.show_histogram == "equalized":
                equalized_img = HistogramProcessor.equalize_histogram(st.session_state.current_image.copy())
                st.subheader("Equalized Histogram")
                fig = HistogramProcessor.plot_histogram(equalized_img)
                st.pyplot(fig)
                st.info("This is a preview. Changes are not applied to the image.")
            
            # Add apply button if viewing equalized histogram
            if st.session_state.show_histogram == "equalized":
                if st.sidebar.button("Apply Equalization"):
                    st.session_state.current_image = HistogramProcessor.equalize_histogram(
                        st.session_state.current_image
                    )
                    st.sidebar.success("Histogram equalization applied!")
                
        elif operation == "Inversion/Transform":
            transform_mode = st.sidebar.selectbox(
                "Transformation Type",
                ["Invert", "Rotate", "Resize"]
            )
            
            if transform_mode == "Invert":
                if st.sidebar.button("Preview"):
                    st.session_state.preview_image = cv2.bitwise_not(
                        st.session_state.current_image.copy()
                    )
                
                if st.sidebar.button("Apply"):
                    st.session_state.current_image = cv2.bitwise_not(
                        st.session_state.current_image
                    )
                    st.session_state.preview_image = None
                    st.sidebar.success("Inversion applied!")
                    
            elif transform_mode == "Rotate":
                angle = st.sidebar.slider("Rotation Angle", 0, 360, 0)
                
                if st.sidebar.button("Preview"):
                    rows, cols = st.session_state.current_image.shape[:2]
                    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                    st.session_state.preview_image = cv2.warpAffine(
                        st.session_state.current_image.copy(),
                        M,
                        (cols, rows)
                    )
                
                if st.sidebar.button("Apply"):
                    rows, cols = st.session_state.current_image.shape[:2]
                    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                    st.session_state.current_image = cv2.warpAffine(
                        st.session_state.current_image,
                        M,
                        (cols, rows)
                    )
                    st.session_state.preview_image = None
                    st.sidebar.success("Rotation applied!")
                    
            elif transform_mode == "Resize":
                scale = st.sidebar.slider("Scale Percentage", 1, 200, 100)
                
                if st.sidebar.button("Preview"):
                    width = int(st.session_state.current_image.shape[1] * scale / 100)
                    height = int(st.session_state.current_image.shape[0] * scale / 100)
                    dim = (width, height)
                    st.session_state.preview_image = cv2.resize(
                        st.session_state.current_image.copy(),
                        dim,
                        interpolation=cv2.INTER_AREA
                    )
                
                if st.sidebar.button("Apply"):
                    width = int(st.session_state.current_image.shape[1] * scale / 100)
                    height = int(st.session_state.current_image.shape[0] * scale / 100)
                    dim = (width, height)
                    st.session_state.current_image = cv2.resize(
                        st.session_state.current_image,
                        dim,
                        interpolation=cv2.INTER_AREA
                    )
                    st.session_state.preview_image = None
                    st.sidebar.success("Resize applied!")
                    
        elif operation == "Reset":
            if st.sidebar.button("Reset to Original"):
                st.session_state.current_image = st.session_state.original_image.copy()
                st.session_state.preview_image = None
                st.session_state.show_histogram = None
                st.sidebar.success("Image reset to original!")
        
        # Display preview if available
        if st.session_state.preview_image is not None:
            st.image(st.session_state.preview_image,
                    channels="BGR" if len(st.session_state.preview_image.shape)==3 else "GRAY",
                    use_column_width=True,
                    caption="Preview - Click Apply to confirm changes")

if __name__ == "__main__":
    main()
