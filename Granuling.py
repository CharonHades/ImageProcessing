import cv2
import numpy as np

class Segmenter:
    @staticmethod
    def threshold_segmentation(image, mode='binary'):
        """Segment image using thresholding"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if mode == 'otsu':
            _, thresh = cv2.threshold(gray, 0, 255, 
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:  # Default binary
            _, thresh = cv2.threshold(gray, 127, 255, 
                                    cv2.THRESH_BINARY)
            
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)