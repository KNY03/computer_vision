import cv2
import datetime

capture = cv2.VideoCapture("./video/youquiz4.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
record = False

while True:
    if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.open("./video/youquiz4.mp4")

    ret, frame = capture.read()
    cv2.imshow("VideoFrame", frame)

    now = datetime.datetime.now().strftime("%d_%H-%M-%S")
    key = cv2.waitKey(33)
    print("key =>", key)
    
    if key == 27: # ESC 키
        break
    elif key == 49: # 1번 키
        print("캡쳐")
        cv2.imwrite("./capture/" + str(now) + ".png", frame)
    elif key == 50: # 2번 키
        print("녹화 시작")
        record = True
        video = cv2.VideoWriter("./capture/" + str(now) + ".avi", fourcc, 20.0,
                                (frame.shape[1], frame.shape[0]))
    elif key == 51: # 3번 키
        print("녹화 중지")
        record = False
        video.release()

    if record == True:
        print("녹화 중...")
        video.write(frame)

capture.release()
cv2.destroyAllWindows()