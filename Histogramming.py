import cv2
import matplotlib.pyplot as plt
import numpy as np

class HistogramProcessor:
    @staticmethod
    def plot_histogram(image, title="Histogram"):
        """Plot RGB/grayscale histogram with image side by side"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Show the image
        if len(image.shape) == 3:  # Color image
            ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:  # Grayscale
            ax1.imshow(image, cmap='gray')
        ax1.set_title('Image')
        ax1.axis('off')

        # Show the histogram
        if len(image.shape) == 3:  # Color image
            colors = ('b', 'g', 'r')
            for i, color in enumerate(colors):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                ax2.plot(hist, color=color)
            ax2.set_title('Color Histogram')
        else:  # Grayscale
            ax2.hist(image.ravel(), 256, [0, 256])
            ax2.set_title('Grayscale Histogram')
            
        fig.suptitle(title)
        plt.tight_layout()
        return fig

    @staticmethod
    def equalize_histogram(image):
        """Apply histogram equalization and return equalized image"""
        if len(image.shape) == 3:  # Color
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:  # Grayscale
            equalized = cv2.equalizeHist(image)
        return equalized