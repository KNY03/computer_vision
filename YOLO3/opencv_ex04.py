import cv2

image = cv2.imread("C:/Data/computer_vision/YOLO3/video/cat.jpg", cv2.IMREAD_ANYCOLOR)
cv2.imshow("cat", image)
cv2.waitKey(0)
cv2.destroyAllWindows()