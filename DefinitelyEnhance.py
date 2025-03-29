import cv2
import numpy as np

class ImageSharpener:
    @staticmethod
    def sharpen(image, strength=1.0):
        # Base sharpening kernel
        base_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
        
        # Adjust kernel based on strength
        identity = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]])
        
        # Blend between identity and sharpening kernel based on strength
        kernel = identity + (base_kernel - identity) * strength
        return cv2.filter2D(image, -1, kernel)
