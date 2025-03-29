import cv2
import numpy as np

class ColorShifter:
    @staticmethod
    def shift_channels(image, r_shift, g_shift, b_shift):
        shifted = image.copy()
        
        # Split channels
        if len(image.shape) == 3:
            b, g, r = cv2.split(shifted)
            
            # Apply shifts with wrap-around
            r = (r + r_shift) % 255
            g = (g + g_shift) % 255
            b = (b + b_shift) % 255
            
            return cv2.merge([b, g, r])
        return shifted