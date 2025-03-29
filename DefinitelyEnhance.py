import cv2

class ImageSmoother:
    @staticmethod
    def gaussian_blur(image, kernel_size=(5,5)):
        return cv2.GaussianBlur(image, kernel_size, 0)