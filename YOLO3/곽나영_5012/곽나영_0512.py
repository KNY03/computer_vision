import cv2
from ultralytics import YOLO
from pathlib import Path

# 폴더 경로 설정
base_dir = Path("C:/Data/computer_vision/YOLO3")

# 저장할 영상 파일명
video_path = str(base_dir / "yolo26.avi")

# YOLO 모델 경로
model_path = str(base_dir / "yolo26n.pt")

# 1. 노트북 카메라로 영상 촬영 및 저장
capture = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
record = False
video = None

# 예외 처리
if not capture.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 사용자 안내 메시지
print("카메라 촬영 화면입니다.")
print("1번 키: 녹화 시작")
print("2번 키: 녹화 중지")
print("ESC 키: 촬영 종료 후 YOLO 탐지 시작")

while True:
    ret, frame = capture.read()

    # 프레임을 읽을 수 없는 경우 예외 처리
    if not ret:
        print("카메라 프레임을 읽을 수 없습니다.")
        break

    cv2.imshow("VideoFrame", frame)

    key = cv2.waitKey(33)

    if key == 27:  # ESC 키
        print("촬영 종료")
        break

    elif key == 49:  # 1번 키
        if record == False:
            print("녹화 시작")
            record = True

            video = cv2.VideoWriter(
                video_path,
                fourcc,
                20.0,
                (frame.shape[1], frame.shape[0])
            )

    elif key == 50:  # 2번 키
        if record == True:
            print("녹화 중지")
            record = False

            if video is not None:
                video.release()
                video = None

    if record == True and video is not None:
        video.write(frame)

capture.release()

if video is not None:
    video.release()

cv2.destroyAllWindows()

print("영상 저장 완료:", video_path)

# 2. 저장된 yolo26.avi 파일을 불러와서 YOLO 객체 탐지
model = YOLO(model_path)

VideoSignal = cv2.VideoCapture(video_path)

# 예외 처리
if not VideoSignal.isOpened():
    print("저장된 yolo26.avi 파일을 열 수 없습니다.")
    exit()

# 사용자 안내 메시지
print("YOLO 객체 탐지를 시작합니다.")
print("아무 키나 누르면 종료됩니다.")

while True:
    ret, frame = VideoSignal.read()

    # 프레임을 읽을 수 없는 경우 예외 처리
    if not ret:
        print("영상 재생이 끝났습니다.")
        break

    # YOLO 객체 탐지 수행
    results = model(frame, verbose=False)

    # 탐지 결과 가져오기
    boxes = results[0].boxes

    for box in boxes:
        # 신뢰도
        confidence = float(box.conf[0])

        if confidence > 0.5:
            # 바운딩박스 좌표
            x1, y1, x2, y2 = box.xyxy[0]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            # 클래스 번호와 객체명
            class_id = int(box.cls[0])
            label = model.names[class_id]

            # 객체명 + 신뢰도
            text = label + " " + str(round(confidence, 2))

            # 바운딩박스 그리기
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

            # 객체명과 confidence 출력
            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_ITALIC,
                0.6,
                (255, 255, 255),
                2
            )

    cv2.imshow("YOLO26 Detection", frame)

    if cv2.waitKey(100) > 0:
        break

VideoSignal.release()
cv2.destroyAllWindows()