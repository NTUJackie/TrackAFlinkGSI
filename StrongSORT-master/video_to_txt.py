import torch
import cv2
import numpy as np
import os
import dill


model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/junyan/Documents/train_model/yolov5/runs/train/exp2/weights/best.pt')
#model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
video_path = '/home/junyan/Documents/test_videos/2024-03-04 15-51-53 NPHL BSL3+-CAM08 ext  13860.mp4'
output_video_path = '/home/junyan/Documents/test_videos/output_video2.mp4'

cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

frame_idx = 0
confidence_threshold = 0.5

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    results = model(frame)

    detections = results.xyxy[0].cpu().numpy()

    for detection in detections:
        xmin, ymin, xmax, ymax, confidence, class_id = detection
        width = xmax - xmin
        height = ymax - ymin

        if int(class_id) == 0 and confidence >= confidence_threshold:
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

            label = f'Person: {confidence:.2f}'
            cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()





#
# cap = cv2.VideoCapture(video_path)
#
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))
#
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
#
# frame_idx = 0
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     frame_idx += 1
#     results = model(frame)
#
#     detections = results.xyxy[0].cpu().numpy()
#
#     for detection in detections:
#         xmin, ymin, xmax, ymax, confidence, class_id = detection
#         width = xmax - xmin
#         height = ymax - ymin
#
#         if int(class_id) == 0:
#             cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
#
#             label = f'Person: {confidence:.2f}'
#             cv2.putText(frame, label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#     out.write(frame)
#
# cap.release()
# out.release()




#
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/junyan/Documents/yolov5/runs/train/exp6/weights/best.pt')
#
# video_path = '/home/junyan/Documents/2024-01-09.mp4'
# #output_txt_path = '/home/junyan/Documents/2024-01-09.txt'
#
#
# cap = cv2.VideoCapture(video_path)
#
# frame_idx = 0
# output_data = []
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     frame_idx += 1
#     results = model(frame)
#
#     detections = results.xyxy[0].cpu().numpy()
#     print(detections)
#     for detection in detections:
#         xmin, ymin, xmax, ymax, confidence, class_id = detection
#         width = xmax - xmin
#         height = ymax - ymin
#
#         if int(class_id) == 0:
#             target_id = int(class_id)
#             output_data.append([frame_idx, target_id, xmin, ymin, width, height, confidence, 0, 0, 0])
#
# cap.release()
#
# np.savetxt(output_txt_path, np.array(output_data), fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d')
#
# cap = cv2.VideoCapture(video_path)
#
# frame_idx = 0
# output_data = []
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     frame_idx += 1
#     results = model(frame)
#
#     detections = results.xyxy[0].cpu().numpy()
#
#     for detection in detections:
#         xmin, ymin, xmax, ymax, confidence, class_id = detection
#         width = xmax - xmin
#         height = ymax - ymin
#
#         target_id = int(class_id)
#         output_data.append([frame_idx, target_id, xmin, ymin, width, height, confidence, 0, 0, 0])
#
# cap.release()
#
# np.savetxt(output_txt_path, np.array(output_data), fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d')
