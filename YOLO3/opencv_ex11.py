import cv2

src = cv2.imread('C:/Data/computer_vision/YOLO3/video/chess.jpg', cv2.IMREAD_COLOR)

dst = src.copy()
roi = src[100:600, 200:700]
dst[0:500, 0:500] = roi

src = cv2.resize(src, dsize=None, fx=1/2, fy=1/2, interpolation=cv2.INTER_AREA)
dst = cv2.resize(dst, dsize=None, fx=1/2, fy=1/2, interpolation=cv2.INTER_AREA)

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.waitKey()
cv2.destroyAllWindows()