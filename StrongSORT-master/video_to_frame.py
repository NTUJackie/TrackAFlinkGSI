
# import cv2
# from ultralytics import YOLO
#
# model = YOLO('yolov8n.pt')
# def extract_frames_with_people(video_path, output_dir):
#     cap = cv2.VideoCapture(video_path)
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     frame_interval = fps
#     frame_number = 0
#     saved_frame_count = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if frame_number % frame_interval == 0:
#             results = model(frame)
#             people_detected = any(result.boxes.cls == 0 for result in results)
#             if people_detected:
#                 output_frame_path = f"{output_dir}/frame_{saved_frame_count}.jpg"
#                 cv2.imwrite(output_frame_path, frame)
#                 saved_frame_count += 1
#                 #print(f"Saved frame {saved_frame_count} at {output_frame_path}")
#         frame_number += 1
#     cap.release()


import cv2
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
def extract_frames_with_people(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps
    frame_number = 0
    saved_frame_count = 0

    while cap.isOpened():
        if frame_number > 1:
            break
        ret, frame = cap.read()
        if not ret:
            break
        if frame_number % frame_interval == 0:
            results = model(frame)
            people_detected = False
            for result in results:
                for cls in result.boxes.cls:
                    if int(cls) == 0:
                        people_detected = True
                        break
                if people_detected:
                    break
            if people_detected:
                output_frame_path = f"{output_dir}/frame_{saved_frame_count}.jpg"
                cv2.imwrite(output_frame_path, frame)
                saved_frame_count += 1
                print(f"Saved frame {saved_frame_count} at {output_frame_path}")
        frame_number += 1
    cap.release()

video_path = '/home/junyan/Documents/refrigerator_test/2024-09-03 10-44-14 Main Bacteria Room - CAM10 ext 13872.mp4'
output_dir = '/home/junyan/Documents/sample_img/sample_img23'
extract_frames_with_people(video_path, output_dir)