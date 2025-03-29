import cv2
import numpy as np

class BasicOperations:
    def __init__(self, uploaded_file):
        self.file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        self.image = cv2.imdecode(self.file_bytes, cv2.IMREAD_COLOR)
        self.original = self.image.copy()
        
    def show_image(self):
        """For non-Streamlit implementations"""
        cv2.imshow("Image", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()