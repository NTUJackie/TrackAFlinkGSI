import cv2
import os


video_path = '/home/junyan/Documents/palace.mp4'
output_folder = '/home/junyan/Documents/output/img1'


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(video_path)
frame_count = 1
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_filename = os.path.join(output_folder, f'{frame_count:06d}.jpg')
    cv2.imwrite(frame_filename, frame)
    frame_count += 1

cap.release()
