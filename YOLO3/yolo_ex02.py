import cv2
import numpy as np

VideoSignal = cv2.VideoCapture(0)

YOLO_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("yolo.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    
layer_names = YOLO_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in YOLO_net.getUnconnectedOutLayers()]

while True:
    ret, frame = VideoSignal.read()
    h, w, c = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
    YOLO_net.setInput(blob)
    outs = YOLO_net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                w_box = int(detection[2] * w)
                h_box = int(detection[3] * h)

                x = int(center_x - w_box / 2)
                y = int(center_y - h_box / 2)

                boxes.append([x, y, w_box, h_box])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h_box = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h_box), color, 2)
            cv2.putText(frame, label, (x, y + 30), cv2.FONT_ITALIC, 3, color, 3)
                
    cv2.imshow("YOLOv3", frame)
    
    if cv2.waitKey(100) > 0:
        break