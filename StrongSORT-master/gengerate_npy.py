#
#

import numpy as np
import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

video_path = "/home/junyan/Documents/2024-01-09.mp4"
output_npy_path = "/home/junyan/Documents/MOT17-02-FRCNN.npy"

cap = cv2.VideoCapture(video_path)

detections = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame,conf=0.5)
    for result in results:
        for box in result.boxes:
            cls = box.cls.item()
            if int(cls) == 0:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                width = x2 - x1
                height = y2 - y1
                conf = box.conf.item()
                detection_row = [cap.get(cv2.CAP_PROP_POS_FRAMES), -1, x1, y1, width, height, conf, -1, -1, -1]
                detections.append(detection_row)

detections_array = np.array(detections, dtype=np.float64)
np.save(output_npy_path, detections_array)

cap.release()

#
# import numpy as np
# import cv2
# from ultralytics import YOLO
# model = YOLO('yolov8n.pt')
#
# video_path = "/home/junyan/Documents/2024-01-09.mp4"
# cap = cv2.VideoCapture(video_path)
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     results = model(frame, conf=0.6)
#     for result in results:
#         for box in result.boxes:
#             cls = box.cls.item()
#             if int(cls) == 0:
#                 x1, y1, x2, y2 = box.xyxy[0].tolist()
#                 width = x2 - x1
#                 height = y2 - y1
#                 score = box.conf.item()
#                 cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#                 label = f'Person: {score:.2f}'
#                 cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#     cv2.imshow('YOLOv8 Detection', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
