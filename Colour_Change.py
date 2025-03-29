import cv2
import numpy as np

class ColorShifter:
    @staticmethod
    def shift_channels(image, r_shift, g_shift, b_shift):
        """
        Shift color channels of an image while preserving the original
        
        Args:
            image: Input BGR image (numpy array)
            r_shift: Red channel shift value (0-255)
            g_shift: Green channel shift value (0-255)
            b_shift: Blue channel shift value (0-255)
            
        Returns:
            A new image with shifted colors, original remains unchanged
        """
        # Create a deep copy to ensure original isn't modified
        shifted = np.copy(image)
        
        if len(shifted.shape) == 3:  # Only process color images
            # Split into individual channels
            b, g, r = cv2.split(shifted)
            
            # Apply shifts with proper 256 modulus
            r_shifted = (r.astype(int) + r_shift) % 256
            g_shifted = (g.astype(int) + g_shift) % 256
            b_shifted = (b.astype(int) + b_shift) % 256
            
            # Merge back into BGR format
            return cv2.merge([
                b_shifted.astype('uint8'),
                g_shifted.astype('uint8'),
                r_shifted.astype('uint8')
            ])
        
        return shifted.copy()  # Return copy of grayscale if not color
