import cv2
import os

video_path = '/home/junyan/Documents/Freezer doorOpening_SAMPLE VIDEOS/2024-09-03 11-23-48 Main Bacteria Room - CAM10 ext 13872.mp4'
output_folder = '/home/junyan/Documents/sample_img/sample_img26'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps)

frame_count = 0
saved_frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    if frame_count % frame_interval == 0:
        frame_name = os.path.join(output_folder, f"frame_{saved_frame_count}.jpg")
        cv2.imwrite(frame_name, frame)
        print(f"Saved frame {saved_frame_count} as {frame_name}")
        saved_frame_count += 1

    frame_count += 1

cap.release()
cv2.destroyAllWindows()