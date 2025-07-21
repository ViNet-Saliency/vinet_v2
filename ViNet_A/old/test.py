import cv2
import numpy as np
print(f"OpenCV version: {cv2.__version__}")

# Test with a simple array
test_img = np.zeros((100, 100, 3), dtype=np.uint8)
resized = cv2.resize(test_img, (50, 50))
print("OpenCV working correctly!")