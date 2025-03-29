import cv2
import numpy as np

class ImageTransformer:
    @staticmethod
    def invert_colors(image):
        """Invert image colors"""
        return cv2.bitwise_not(image)

    @staticmethod
    def rotate_image(image, angle):
        """Rotate image by specified angle"""
        h, w = image.shape[:2]
        center = (w//2, h//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))

    @staticmethod
    def resize_image(image, scale_percent):
        """Resize image by percentage"""
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        return cv2.resize(image, (width, height), 
                        interpolation=cv2.INTER_AREA)