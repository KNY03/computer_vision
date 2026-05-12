import cv2
import numpy as np

img = cv2.imread('C:/Data/computer_vision/YOLO3/video/cat.jpg')
smaller = cv2.pyrDown(img)
bigger = cv2.pyrUp(smaller)

laplacian = cv2.subtract(img, bigger)
restored = bigger + laplacian

# merged = np.hstack((img, laplacian, bigger, restored))
# cv2.imshow('Laplacian Pyramid', merged)
cv2.imshow('img', img)
cv2.imshow('bigger', bigger)
cv2.imshow('Restored', restored)
cv2.waitKey(0)
cv2.destroyAllWindows()