import cv2

capture = cv2.VideoCapture("C:/Data/computer_vision/YOLO3/video/youquiz4.mp4")

print(capture.get(cv2.CAP_PROP_POS_FRAMES))
print(capture.get(cv2.CAP_PROP_FRAME_COUNT))

while cv2.waitKey(1) < 0:
    if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, frame = capture.read()
    cv2.imshow("Wildlife", frame)

capture.release()
cv2.destroyAllWindows()