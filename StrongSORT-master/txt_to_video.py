# import cv2
# import numpy as np
#
# def visualize_aflink_tracking(track_file, video_file):
#     tracks = np.loadtxt(track_file, delimiter=',')
#     cap = cv2.VideoCapture(video_file)
#     frame_id = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_id += 1
#
#         current_tracks = tracks[tracks[:, 0] == frame_id]
#         for track in current_tracks:
#             track_id = int(track[1])
#             x1, y1, w, h = map(int, track[2:6])
#             score = track[6]
#
#             cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
#             cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
#
#         cv2.imshow('AFLink Tracking', frame)
#         if cv2.waitKey(60) & 0xFF == ord('q'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#
# visualize_aflink_tracking('/home/junyan/Documents/StrongSORT-master/data/StrongSORT/MOT17-02-FRCNN.txt', '/home/junyan/Documents/2024-01-09.mp4')
#


import cv2
import numpy as np

def visualize_and_save_aflink_tracking(track_file, video_file, output_file):
    tracks = np.loadtxt(track_file, delimiter=',')

    cap = cv2.VideoCapture(video_file)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        current_tracks = tracks[tracks[:, 0] == frame_id]

        for track in current_tracks:
            track_id = int(track[1])
            x1, y1, w, h = map(int, track[2:6])
            score = track[6]

            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        out.write(frame)

        cv2.imshow('AFLink Tracking', frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

visualize_and_save_aflink_tracking('/home/junyan/Documents/bytetrack_output/bytetrack_txt.txt', '/home/junyan/Documents/2024-01-09.mp4', '/home/junyan/Documents/bytetrack_output/bytetrack+AFlink+GSI.mp4')