import cv2
import numpy as np

class ImageUtils:
    @staticmethod
    def bytes_to_image(file_bytes):
        return cv2.imdecode(np.frombuffer(file_bytes, np.uint8), 
                           cv2.IMREAD_COLOR)