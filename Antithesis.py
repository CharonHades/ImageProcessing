import streamlit as st
import cv2
import numpy as np
from Basically import BasicOperations
from Colour_Change import ColorShifter
from DefinitelyEnhance import ImageSmoother
from Edgy import EdgeDetector
from Featuring import ImageSharpener

def main():
    st.title("🖼️ Image Processing Suite")
    
    # File upload
    uploaded_file = st.file_uploader("Choose an image", 
                                    type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        processor = BasicOperations(uploaded_file)
        
        if 'original' not in st.session_state:
            st.session_state.original = processor.image.copy()
            
        # Sidebar controls
        operation = st.sidebar.selectbox(
            "Select Operation",
            ["Original", "Color Shift", "Smooth", 
             "Edge Detect", "Sharpen", "Reset"]
        )
        
        # Process based on selection
        if operation == "Color Shift":
            r_shift = st.sidebar.slider("Red Shift", -255, 255, 0)
            g_shift = st.sidebar.slider("Green Shift", -255, 255, 0)
            b_shift = st.sidebar.slider("Blue Shift", -255, 255, 0)
            processor.image = ColorShifter.shift_channels(
                processor.image, r_shift, g_shift, b_shift
            )
            
        elif operation == "Smooth":
            processor.image = ImageSmoother.gaussian_blur(processor.image)
            
        elif operation == "Edge Detect":
            processor.image = EdgeDetector.detect_edges(processor.image)
            
        elif operation == "Sharpen":
            processor.image = ImageSharpener.sharpen(processor.image)
            
        elif operation == "Reset":
            processor.image = st.session_state.original.copy()
        
        # Display
        st.image(processor.image, 
                channels="BGR" if len(processor.image.shape)==3 else "GRAY",
                use_column_width=True)

if __name__ == "__main__":
    main()