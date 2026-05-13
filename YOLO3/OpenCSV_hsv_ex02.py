import cv2

src = cv2.imread('./video/tomato.jpg', cv2.IMREAD_COLOR)
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

h = cv2.inRange(h, 8, 20)
orange = cv2.bitwise_and(hsv, hsv, mask=h)
orange = cv2.cvtColor(orange, cv2.COLOR_HSV2BGR)

src = cv2.resize(src, None, fx=1/4, fy=1/4, interpolation=cv2.INTER_AREA)
orange = cv2.resize(orange, None, fx=1/4, fy=1/4, interpolation=cv2.INTER_AREA)

cv2.imshow('src', src)
cv2.imshow('orange', orange)
cv2.waitKey()
cv2.destroyAllWindows()

