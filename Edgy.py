import cv2

class EdgeDetector:
    @staticmethod
    def detect_edges(image, threshold1=100, threshold2=200):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, threshold1, threshold2)