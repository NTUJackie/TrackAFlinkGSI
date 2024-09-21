# import cv2
# import numpy as np
#
# x1, y1 = 523, 138
# x2, y2 = 838, 806
#
# cap = cv2.VideoCapture('/home/junyan/Documents/2024-01-09.mp4')
#
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#
# video_duration = frame_count / fps
#
# ret, prev_frame = cap.read()
#
# prev_frame = prev_frame[y1:y2, x1:x2]
# prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#
# motion_start_time = None
# motion_end_time = None
# motion_detected = False
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
#     current_time = frame_idx / fps
#     fridge_region = frame[y1:y2, x1:x2]
#     gray = cv2.cvtColor(fridge_region, cv2.COLOR_BGR2GRAY)
#
#     diff = cv2.absdiff(prev_gray, gray)
#     _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
#     change = np.sum(thresh)
#
#     if change > 5000:
#         if not motion_detected:
#             motion_start_time = current_time
#             motion_detected = True
#     else:
#         if motion_detected:
#             motion_end_time = current_time
#             motion_detected = False
#             start_minutes = int(motion_start_time // 60)
#             start_seconds = int(motion_start_time % 60)
#             end_minutes = int(motion_end_time // 60)
#             end_seconds = int(motion_end_time % 60)
#             print(f"refrigerator change time: {start_minutes}:{start_seconds} to {end_minutes}:{end_seconds}")
#
#     prev_gray = gray
#
# if motion_detected:
#     motion_end_time = frame_idx / fps
#     start_minutes = int(motion_start_time // 60)
#     start_seconds = int(motion_start_time % 60)
#     end_minutes = int(motion_end_time // 60)
#     end_seconds = int(motion_end_time % 60)
#     print(f"refrigerator change time: {start_minutes}:{start_seconds} to {end_minutes}:{end_seconds}")
#
# cap.release()
#
#
# import cv2
# import numpy as np
#
# x1, y1 = 523, 138
# x2, y2 = 838, 806
#
# cap = cv2.VideoCapture('/home/junyan/Documents/2024-01-09.mp4')
#
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#
# video_duration = frame_count / fps
#
# ret, prev_frame = cap.read()
#
# prev_frame = prev_frame[y1:y2, x1:x2]
# prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#
# motion_start_time = None
# motion_end_time = None
# motion_detected = False
# recording_motion = False
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
#     current_time = frame_idx / fps
#
#     fridge_region = frame[y1:y2, x1:x2]
#     gray = cv2.cvtColor(fridge_region, cv2.COLOR_BGR2GRAY)
#
#     diff = cv2.absdiff(prev_gray, gray)
#     _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
#
#     change = np.sum(thresh)
#
#     status = "refrigerator closed"
#
#     if change > 200000:
#         status = "refrigerator opened"
#         if not motion_detected and not recording_motion:
#             motion_start_time = current_time
#             motion_detected = True
#             recording_motion = True
#             start_minutes = int(motion_start_time // 60)
#             start_seconds = int(motion_start_time % 60)
#             print(f"refrigerator change begin time: {start_minutes}:{start_seconds}")
#     else:
#         if motion_detected:
#             motion_end_time = current_time
#             motion_detected = False
#             end_minutes = int(motion_end_time // 60)
#             end_seconds = int(motion_end_time % 60)
#             print(f"refrigerator change close time: {end_minutes}:{end_seconds}")
#             recording_motion = False
#
#     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     cv2.putText(frame, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#     cv2.putText(frame, f"Position: ({x1}, {y1}), ({x2}, {y2})", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#
#     cv2.imshow("Fridge Status", frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
#     prev_gray = gray
#
# if recording_motion:
#     motion_end_time = frame_idx / fps
#     start_minutes = int(motion_start_time // 60)
#     start_seconds = int(motion_start_time % 60)
#     end_minutes = int(motion_end_time // 60)
#     end_seconds = int(motion_end_time % 60)
#     print(f"refrigerator change time: {start_minutes}:{start_seconds} to {end_minutes}:{end_seconds}")
#
# cap.release()
# cv2.destroyAllWindows()

#
# import cv2
# import numpy as np
#
# x1, y1 = 523, 138
# x2, y2 = 838, 806
#
# cap = cv2.VideoCapture('/home/junyan/Documents/2024-01-09.mp4')
#
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
# output_video = cv2.VideoWriter('/home/junyan/Documents/output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
#
# video_duration = frame_count / fps
#
# ret, prev_frame = cap.read()
#
# prev_frame = prev_frame[y1:y2, x1:x2]
# prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#
# motion_start_time = None
# motion_end_time = None
# motion_detected = False
# recording_motion = False
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
#     current_time = frame_idx / fps
#
#     fridge_region = frame[y1:y2, x1:x2]
#     gray = cv2.cvtColor(fridge_region, cv2.COLOR_BGR2GRAY)
#
#     diff = cv2.absdiff(prev_gray, gray)
#
#     _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
#
#     change = np.sum(thresh)
#
#     status = "refrigerator closed"
#
#     if change > 5000:
#         status = "refrigerator opened"
#         if not motion_detected and not recording_motion:
#             motion_start_time = current_time
#             motion_detected = True
#             recording_motion = True
#             start_minutes = int(motion_start_time // 60)
#             start_seconds = int(motion_start_time % 60)
#             print(f"refrigerator change begin time: {start_minutes}:{start_seconds}")
#     else:
#         if motion_detected:
#             motion_end_time = current_time
#             motion_detected = False
#             end_minutes = int(motion_end_time // 60)
#             end_seconds = int(motion_end_time % 60)
#             print(f"refrigerator change close time: {end_minutes}:{end_seconds}")
#             recording_motion = False
#
#     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#     cv2.putText(frame, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#     #cv2.putText(frame, f"Position: ({x1}, {y1}), ({x2}, {y2})", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#
#     cv2.imshow("Fridge Status", frame)
#
#     output_video.write(frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
#     prev_gray = gray
#
# if recording_motion:
#     motion_end_time = frame_idx / fps
#     start_minutes = int(motion_start_time // 60)
#     start_seconds = int(motion_start_time % 60)
#     end_minutes = int(motion_end_time // 60)
#     end_seconds = int(motion_end_time % 60)
#     print(f"refrigerator change time: {start_minutes}:{start_seconds} to {end_minutes}:{end_seconds}")
#
# cap.release()
# output_video.release()
# cv2.destroyAllWindows()









#
#
# import cv2
# import numpy as np
# from skimage.metrics import structural_similarity as ssim
#
# x1, y1 = 523, 138
# x2, y2 = 838, 806
#
# cap = cv2.VideoCapture('/home/junyan/Documents/2024-01-09.mp4')
#
# ret, first_frame = cap.read()
# if not ret:
#     print("can't get video")
#     exit()
#
# first_frame_region = first_frame[y1:y2, x1:x2]
# first_gray = cv2.cvtColor(first_frame_region, cv2.COLOR_BGR2GRAY)
#
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
#
# output = cv2.VideoWriter('/home/junyan/Documents/output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     current_region = frame[y1:y2, x1:x2]
#     current_gray = cv2.cvtColor(current_region, cv2.COLOR_BGR2GRAY)
#
#     similarity, _ = ssim(first_gray, current_gray, full=True)
#
#     if similarity < 0.9:
#         status = "open"
#     else:
#         status = "close"
#
#     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     cv2.putText(frame, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#
#     output.write(frame)
#     cv2.imshow("Fridge Status", frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# output.release()
# cv2.destroyAllWindows()
#







#
# import cv2
# import numpy as np
# from skimage.metrics import structural_similarity as ssim
#
# region_1 = [964, 272, 1024, 325]
# region_2 = [1083, 216, 1111, 257]
# region_3 = [890, 215, 1020, 246]
# fridge_x1, fridge_y1, fridge_x2, fridge_y2 = 870, 66, 1325, 870
#
# orb = cv2.ORB_create()
#
# cap = cv2.VideoCapture('/home/junyan/Documents/refrigerator_test/2024-09-03 Main Bacteria.mp4')
#
# ret, first_frame = cap.read()
#
# if not ret:
#     print("can't get video")
#     exit()
#
# first_region_1 = first_frame[region_1[1]:region_1[3], region_1[0]:region_1[2]]
# first_region_2 = first_frame[region_2[1]:region_2[3], region_2[0]:region_2[2]]
# first_region_3 = first_frame[region_3[1]:region_3[3], region_3[0]:region_3[2]]
#
# first_gray_1 = cv2.cvtColor(first_region_1, cv2.COLOR_BGR2GRAY)
# first_gray_2 = cv2.cvtColor(first_region_2, cv2.COLOR_BGR2GRAY)
# first_gray_3 = cv2.cvtColor(first_region_3, cv2.COLOR_BGR2GRAY)
#
# kp_1, des_1 = orb.detectAndCompute(first_gray_1, None)
# kp_2, des_2 = orb.detectAndCompute(first_gray_2, None)
# kp_3, des_3 = orb.detectAndCompute(first_gray_3, None)
#
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
#
# output = cv2.VideoWriter('/home/junyan/Documents/refigerator5.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
#
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     current_region_1 = frame[region_1[1]:region_1[3], region_1[0]:region_1[2]]
#     current_region_2 = frame[region_2[1]:region_2[3], region_2[0]:region_2[2]]
#     current_region_3 = frame[region_3[1]:region_3[3], region_3[0]:region_3[2]]
#
#     current_gray_1 = cv2.cvtColor(current_region_1, cv2.COLOR_BGR2GRAY)
#     current_gray_2 = cv2.cvtColor(current_region_2, cv2.COLOR_BGR2GRAY)
#     current_gray_3 = cv2.cvtColor(current_region_3, cv2.COLOR_BGR2GRAY)
#
#     kp_curr_1, des_curr_1 = orb.detectAndCompute(current_gray_1, None)
#     kp_curr_2, des_curr_2 = orb.detectAndCompute(current_gray_2, None)
#     kp_curr_3, des_curr_3 = orb.detectAndCompute(current_gray_3, None)
#
#     matches_1 = bf.match(des_1, des_curr_1)
#     matches_2 = bf.match(des_2, des_curr_2)
#     matches_3 = bf.match(des_3, des_curr_3)
#
#     def compute_displacement(matches, kp_base, kp_current):
#         displacements = []
#         for match in matches:
#             base_pt = kp_base[match.queryIdx].pt
#             current_pt = kp_current[match.trainIdx].pt
#             displacement = np.linalg.norm(np.array(base_pt) - np.array(current_pt))
#             displacements.append(displacement)
#         return np.mean(displacements) if len(displacements) > 0 else 0
#
#     displacement_1 = compute_displacement(matches_1, kp_1, kp_curr_1)
#     displacement_2 = compute_displacement(matches_2, kp_2, kp_curr_2)
#     displacement_3 = compute_displacement(matches_3, kp_3, kp_curr_3)
#
#     if np.abs(displacement_1 - displacement_2) < 1000 and np.abs(displacement_1 - displacement_3) < 1000:
#         print("Detected shaking, updating region positions...")
#         region_1 = [int(region_1[0] + displacement_1), int(region_1[1] + displacement_1), int(region_1[2] + displacement_1), int(region_1[3] + displacement_1)]
#         region_2 = [int(region_2[0] + displacement_2), int(region_2[1] + displacement_2), int(region_2[2] + displacement_2), int(region_2[3] + displacement_2)]
#         region_3 = [int(region_3[0] + displacement_3), int(region_3[1] + displacement_3), int(region_3[2] + displacement_3), int(region_3[3] + displacement_3)]
#
#     similarity_1, _ = ssim(first_gray_1, current_gray_1, full=True)
#     similarity_2, _ = ssim(first_gray_2, current_gray_2, full=True)
#     similarity_3, _ = ssim(first_gray_3, current_gray_3, full=True)
#
#     if similarity_1 > 0.8 or similarity_2 > 0.8 or similarity_3 > 0.8:
#         status = "close"
#     else:
#         status = "open"
#
#     cv2.rectangle(frame, (fridge_x1, fridge_y1), (fridge_x2, fridge_y2), (0, 255, 0), 2)
#     cv2.putText(frame, status, (fridge_x1, fridge_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#     output.write(frame)
#     cv2.imshow("Fridge Status", frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# output.release()
# cv2.destroyAllWindows()
#
# import cv2
# import numpy as np
# from skimage.metrics import structural_similarity as ssim
#
# region_1 = [964, 272, 1024, 325]
# region_2 = [1083, 216, 1111, 257]
# region_3 = [890, 215, 1020, 246]
# fridge_x1, fridge_y1, fridge_x2, fridge_y2 = 870, 66, 1325, 870
#
# orb = cv2.ORB_create()
#
# cap = cv2.VideoCapture('/home/junyan/Documents/refrigerator_test/2024-09-03 Main Bacteria.mp4')
#
# baseline_image_path = '/home/junyan/Documents/sample_img/sample_img22/frame_12.jpg'
# first_frame = cv2.imread(baseline_image_path)
#
#
# first_region_1 = first_frame[region_1[1]:region_1[3], region_1[0]:region_1[2]]
# first_region_2 = first_frame[region_2[1]:region_2[3], region_2[0]:region_2[2]]
# first_region_3 = first_frame[region_3[1]:region_3[3], region_3[0]:region_3[2]]
#
# first_gray_1 = cv2.cvtColor(first_region_1, cv2.COLOR_BGR2GRAY)
# first_gray_2 = cv2.cvtColor(first_region_2, cv2.COLOR_BGR2GRAY)
# first_gray_3 = cv2.cvtColor(first_region_3, cv2.COLOR_BGR2GRAY)
#
# kp_1, des_1 = orb.detectAndCompute(first_region_1, None)
# kp_2, des_2 = orb.detectAndCompute(first_gray_2, None)
# kp_3, des_3 = orb.detectAndCompute(first_gray_3, None)
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
#
# output = cv2.VideoWriter('/home/junyan/Documents/refigerator5.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
#
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     current_region_1 = frame[region_1[1]:region_1[3], region_1[0]:region_1[2]]
#     current_region_2 = frame[region_2[1]:region_2[3], region_2[0]:region_2[2]]
#     current_region_3 = frame[region_3[1]:region_3[3], region_3[0]:region_3[2]]
#
#     current_gray_1 = cv2.cvtColor(current_region_1, cv2.COLOR_BGR2GRAY)
#     current_gray_2 = cv2.cvtColor(current_region_2, cv2.COLOR_BGR2GRAY)
#     current_gray_3 = cv2.cvtColor(current_region_3, cv2.COLOR_BGR2GRAY)
#
#     kp_curr_1, des_curr_1 = orb.detectAndCompute(current_gray_1, None)
#     kp_curr_2, des_curr_2 = orb.detectAndCompute(current_gray_2, None)
#     kp_curr_3, des_curr_3 = orb.detectAndCompute(current_gray_3, None)
#
#     #print("des_1,des_2,des_3",des_1,des_2,des_3)
#
#     matches_1 = bf.match(des_1, des_curr_1)
#     matches_2 = bf.match(des_2, des_curr_2)
#     matches_3 = bf.match(des_3, des_curr_3)
#
#     def compute_average_direction(matches, kp_base, kp_current):
#         directions = []
#         for match in matches:
#             base_pt = np.array(kp_base[match.queryIdx].pt)
#             current_pt = np.array(kp_current[match.trainIdx].pt)
#             direction = current_pt - base_pt
#             norm = np.linalg.norm(direction)
#             if norm > 0:
#                 direction = direction / norm
#             directions.append(direction)
#         return np.mean(directions, axis=0) if len(directions) > 0 else np.array([0, 0])
#
#     # print("matches_1,matches_2,matches_3",matches_1,matches_2,matches_3)
#     # print("kp_1,kp_2,kp_3",kp_1,kp_2,kp_3)
#     # print("kp_curr_1,kp_curr_2,kp_curr_3",kp_curr_1,kp_curr_2,kp_curr_3)
#
#     direction_1 = compute_average_direction(matches_1, kp_1, kp_curr_1)
#     direction_2 = compute_average_direction(matches_2, kp_2, kp_curr_2)
#     direction_3 = compute_average_direction(matches_3, kp_3, kp_curr_3)
#     #print("direction_1,direction_2,direction_1",direction_1,direction_2,direction_3)
#     def are_directions_similar(dir1, dir2, dir3, threshold=0.2):
#         cos_sim_1_2 = np.dot(dir1, dir2) / (np.linalg.norm(dir1) * np.linalg.norm(dir2))
#         cos_sim_1_3 = np.dot(dir1, dir3) / (np.linalg.norm(dir1) * np.linalg.norm(dir3))
#         cos_sim_2_3 = np.dot(dir2, dir3) / (np.linalg.norm(dir2) * np.linalg.norm(dir3))
#         #print("cos_sim_1_2,cos_sim_1_3,cos_sim_2_3", cos_sim_1_2,cos_sim_1_3,cos_sim_1_3)
#         return cos_sim_1_2 > (1 - threshold) and cos_sim_1_3 > (1 - threshold) and cos_sim_2_3 > (1 - threshold)
#
#     if are_directions_similar(direction_1, direction_2, direction_3):
#         print("Detected shaking")
#         region_1 = [int(region_1[0] + direction_1[0]), int(region_1[1] + direction_1[1]), int(region_1[2] + direction_1[0]), int(region_1[3] + direction_1[1])]
#         region_2 = [int(region_2[0] + direction_2[0]), int(region_2[1] + direction_2[1]), int(region_2[2] + direction_2[0]), int(region_2[3] + direction_2[1])]
#         region_3 = [int(region_3[0] + direction_3[0]), int(region_3[1] + direction_3[1]), int(region_3[2] + direction_3[0]), int(region_3[3] + direction_3[1])]
#
#     similarity_1, _ = ssim(first_gray_1, current_gray_1, full=True)
#     similarity_2, _ = ssim(first_gray_2, current_gray_2, full=True)
#     similarity_3, _ = ssim(first_gray_3, current_gray_3, full=True)
#
#     if similarity_1 > 0.8 or similarity_2 > 0.8 or similarity_3 > 0.8:
#         status = "close"
#     else:
#         status = "open"
#
#     # 绘制结果
#     cv2.rectangle(frame, (fridge_x1, fridge_y1), (fridge_x2, fridge_y2), (0, 255, 0), 2)
#     cv2.rectangle(frame, (region_1[0], region_1[1]), (region_1[2], region_1[3]), (0, 255, 0), 2)
#     cv2.rectangle(frame, (region_2[0], region_2[1]), (region_2[2], region_2[3]), (0, 255, 0), 2)
#     cv2.rectangle(frame, (region_3[0], region_3[1]), (region_3[2], region_3[3]), (0, 255, 0), 2)
#
#     cv2.putText(frame, status, (fridge_x1, fridge_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#     output.write(frame)
#     cv2.imshow("Fridge Status", frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# output.release()
# cv2.destroyAllWindows()




#
# import cv2
# import numpy as np
# from skimage.metrics import structural_similarity as ssim
#
# region_1 = [964, 272, 1024, 325]
# region_2 = [1083, 216, 1111, 257]
# region_3 = [890, 215, 1020, 246]
# fridge_x1, fridge_y1, fridge_x2, fridge_y2 = 870, 66, 1325, 870
#
# sift = cv2.SIFT_create()
#
# cap = cv2.VideoCapture('/home/junyan/Documents/refrigerator_test/2024-09-03 Main Bacteria.mp4')
# baseline_image_path = '/home/junyan/Documents/sample_img/sample_img22/frame_12.jpg'
# first_frame = cv2.imread(baseline_image_path)
#
# first_region_1 = first_frame[region_1[1]:region_1[3], region_1[0]:region_1[2]]
# first_region_2 = first_frame[region_2[1]:region_2[3], region_2[0]:region_2[2]]
# first_region_3 = first_frame[region_3[1]:region_3[3], region_3[0]:region_3[2]]
#
# first_gray_1 = cv2.cvtColor(first_region_1, cv2.COLOR_BGR2GRAY)
# first_gray_2 = cv2.cvtColor(first_region_2, cv2.COLOR_BGR2GRAY)
# first_gray_3 = cv2.cvtColor(first_region_3, cv2.COLOR_BGR2GRAY)
#
# kp_1, des_1 = sift.detectAndCompute(first_gray_1, None)
# kp_2, des_2 = sift.detectAndCompute(first_gray_2, None)
# kp_3, des_3 = sift.detectAndCompute(first_gray_3, None)
#
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
#
# output = cv2.VideoWriter('/home/junyan/Documents/refigerator5.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
#
# bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     current_region_1 = frame[region_1[1]:region_1[3], region_1[0]:region_1[2]]
#     current_region_2 = frame[region_2[1]:region_2[3], region_2[0]:region_2[2]]
#     current_region_3 = frame[region_3[1]:region_3[3], region_3[0]:region_3[2]]
#
#     current_gray_1 = cv2.cvtColor(current_region_1, cv2.COLOR_BGR2GRAY)
#     current_gray_2 = cv2.cvtColor(current_region_2, cv2.COLOR_BGR2GRAY)
#     current_gray_3 = cv2.cvtColor(current_region_3, cv2.COLOR_BGR2GRAY)
#
#     kp_curr_1, des_curr_1 = sift.detectAndCompute(current_gray_1, None)
#     kp_curr_2, des_curr_2 = sift.detectAndCompute(current_gray_2, None)
#     kp_curr_3, des_curr_3 = sift.detectAndCompute(current_gray_3, None)
#
#     matches_1 = bf.match(des_1, des_curr_1)
#     matches_2 = bf.match(des_2, des_curr_2)
#     matches_3 = bf.match(des_3, des_curr_3)
#
#     def compute_average_direction(matches, kp_base, kp_current):
#         directions = []
#         for match in matches:
#             base_pt = np.array(kp_base[match.queryIdx].pt)
#             current_pt = np.array(kp_current[match.trainIdx].pt)
#             direction = current_pt - base_pt
#             norm = np.linalg.norm(direction)
#             if norm > 0:
#                 direction = direction / norm
#             directions.append(direction)
#         return np.mean(directions, axis=0) if len(directions) > 0 else np.array([0, 0])
#
#     direction_1 = compute_average_direction(matches_1, kp_1, kp_curr_1)
#     direction_2 = compute_average_direction(matches_2, kp_2, kp_curr_2)
#     direction_3 = compute_average_direction(matches_3, kp_3, kp_curr_3)
#
#     def are_directions_similar(dir1, dir2, dir3, threshold=0.2):
#         cos_sim_1_2 = np.dot(dir1, dir2) / (np.linalg.norm(dir1) * np.linalg.norm(dir2))
#         cos_sim_1_3 = np.dot(dir1, dir3) / (np.linalg.norm(dir1) * np.linalg.norm(dir3))
#         cos_sim_2_3 = np.dot(dir2, dir3) / (np.linalg.norm(dir2) * np.linalg.norm(dir3))
#         return cos_sim_1_2 > (1 - threshold) and cos_sim_1_3 > (1 - threshold) and cos_sim_2_3 > (1 - threshold)
#
#     if are_directions_similar(direction_1, direction_2, direction_3):
#         print("Detected shaking")
#         region_1 = [int(region_1[0] + direction_1[0]), int(region_1[1] + direction_1[1]), int(region_1[2] + direction_1[0]), int(region_1[3] + direction_1[1])]
#         region_2 = [int(region_2[0] + direction_2[0]), int(region_2[1] + direction_2[1]), int(region_2[2] + direction_2[0]), int(region_2[3] + direction_2[1])]
#         region_3 = [int(region_3[0] + direction_3[0]), int(region_3[1] + direction_3[1]), int(region_3[2] + direction_3[0]), int(region_3[3] + direction_3[1])]
#
#     similarity_1, _ = ssim(first_gray_1, current_gray_1, full=True)
#     similarity_2, _ = ssim(first_gray_2, current_gray_2, full=True)
#     similarity_3, _ = ssim(first_gray_3, current_gray_3, full=True)
#
#     if similarity_1 > 0.8 or similarity_2 > 0.8 or similarity_3 > 0.8:
#         status = "close"
#     else:
#         status = "open"
#
#     cv2.rectangle(frame, (fridge_x1, fridge_y1), (fridge_x2, fridge_y2), (0, 255, 0), 2)
#     cv2.rectangle(frame, (region_1[0], region_1[1]), (region_1[2], region_1[3]), (0, 255, 0), 2)
#     cv2.rectangle(frame, (region_2[0], region_2[1]), (region_2[2], region_2[3]), (0, 255, 0), 2)
#     cv2.rectangle(frame, (region_3[0], region_3[1]), (region_3[2], region_3[3]), (0, 255, 0), 2)
#     cv2.putText(frame, status, (fridge_x1, fridge_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#     output.write(frame)
#     cv2.imshow("Fridge Status", frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# output.release()
# cv2.destroyAllWindows()

#



















# import cv2
# import numpy as np
# from skimage.metrics import structural_similarity as ssim
#
# region_1 = (973, 627, 1042, 700)
# region_2 = (1008, 993, 1046, 1027)
# region_3 = (1211, 931, 1264, 974)
# fridge_x1, fridge_y1, fridge_x2, fridge_y2 = 868, 516, 1424, 1079
#
# baseline_image_path = '/home/junyan/Documents/sample_img/sample_img23/frame_11.jpg'
# baseline_frame = cv2.imread(baseline_image_path)
#
#
# first_region_1 = baseline_frame[region_1[1]:region_1[3], region_1[0]:region_1[2]]
# first_region_2 = baseline_frame[region_2[1]:region_2[3], region_2[0]:region_2[2]]
# first_region_3 = baseline_frame[region_3[1]:region_3[3], region_3[0]:region_3[2]]
#
# first_gray_1 = cv2.cvtColor(first_region_1, cv2.COLOR_BGR2GRAY)
# first_gray_2 = cv2.cvtColor(first_region_2, cv2.COLOR_BGR2GRAY)
# first_gray_3 = cv2.cvtColor(first_region_3, cv2.COLOR_BGR2GRAY)
#
# cap = cv2.VideoCapture('/home/junyan/Documents/refrigerator_test/2024-09-04.mp4')
#
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
#
# output = cv2.VideoWriter('/home/junyan/Documents/refigerator2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     current_region_1 = frame[region_1[1]:region_1[3], region_1[0]:region_1[2]]
#     current_region_2 = frame[region_2[1]:region_2[3], region_2[0]:region_2[2]]
#     current_region_3 = frame[region_3[1]:region_3[3], region_3[0]:region_3[2]]
#
#     current_gray_1 = cv2.cvtColor(current_region_1, cv2.COLOR_BGR2GRAY)
#     current_gray_2 = cv2.cvtColor(current_region_2, cv2.COLOR_BGR2GRAY)
#     current_gray_3 = cv2.cvtColor(current_region_3, cv2.COLOR_BGR2GRAY)
#
#     similarity_1, _ = ssim(first_gray_1, current_gray_1, full=True)
#     similarity_2, _ = ssim(first_gray_2, current_gray_2, full=True)
#     similarity_3, _ = ssim(first_gray_3, current_gray_3, full=True)
#
#     if similarity_1 < 0.7 or (similarity_2 < 0.7 and similarity_3 < 0.7):
#         status = "open"
#     else:
#         status = "close"
#
#     cv2.rectangle(frame, (fridge_x1, fridge_y1), (fridge_x2, fridge_y2), (0, 255, 0), 2)
#
#     cv2.putText(frame, status, (fridge_x1, fridge_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#     output.write(frame)
#     cv2.imshow("Fridge Status", frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# output.release()
# cv2.destroyAllWindows()

# upper_regions = [(973, 627, 1042, 700), (837, 651, 927, 710), (1119, 624, 1149, 655)]        //2024-09-04
# lower_regions = [(1008, 993, 1046, 1027), (1211, 931, 1264, 974), (1115, 1004, 1169, 1043)]
# fridge_x1, fridge_y1, fridge_x2, fridge_y2 = 805, 519, 1431, 1079

# upper_regions = [(931, 200, 970, 242), (773, 193, 859, 260), (632, 214, 719, 252)]
# lower_regions = [(972, 492, 1036, 542), (778, 529, 824, 560), (886, 587, 922, 622)]      #2024-09-03
# fridge_x1, fridge_y1, fridge_x2, fridge_y2 = 609, 90, 1215, 820

# upperRegion = [(1002, 83, 1068, 136), (1074, 95, 1100, 125)]
# lowerRegion = [(1081, 404, 1101, 424)]
# fridge_x1, fridge_y1, fridge_x2, fridge_y2 = 983, 37, 1200, 573

# basePath = '/home/junyan/Documents/sample_img/sample_img19/frame_18.jpg'
# baseFrame = cv2.imread(basePath)





# region_1 = (529, 865, 618, 901)
# region_2 = (805, 793, 868, 852)
# region_3 = (633, 596, 820, 661)
# fridge_x1, fridge_y1, fridge_x2, fridge_y2 = 504, 146, 1005, 992

# region_1 = (827, 184, 874, 224)
# region_2 = (942, 238, 999, 265)
# region_3 = (905, 297, 1020, 350)
# fridge_x1, fridge_y1, fridge_x2, fridge_y2 = 802, 104, 1128, 670
# region_1 = (809, 783, 869, 843)
# region_2 = (769, 868, 914, 901)
# region_3 = (525, 861, 619, 897)
# fridge_x1, fridge_y1, fridge_x2, fridge_y2 = 504, 146, 1005, 992

# region_1 = (964, 272, 1024, 325)
# region_2 = (1083, 216, 1111, 257)
# region_3 = (890, 215, 1020, 246)
# fridge_x1, fridge_y1, fridge_x2, fridge_y2 = 870, 66, 1325, 870
# ret, first_frame = cap.read()
#
# if not ret:
#     print("can't get video")
#     exit()


#
# import cv2
# import numpy as np
# from skimage.metrics import structural_similarity as ssim
# region_1 = (1002, 83, 1068, 136)
# region_2 = (1074, 95, 1100, 125)
# region_3 = (878, 214, 1041, 251)
# fridge_x1, fridge_y1, fridge_x2, fridge_y2 = 865, 62, 1327, 874
#
# cap = cv2.VideoCapture('/home/junyan/Documents/Freezer doorOpening_SAMPLE VIDEOS/2024-09-03 11-01-08 Main Bacteria Room - CAM10 ext 13872.mp4')
#
# baseline_image_path = '/home/junyan/Documents/sample_img/sample_img22/frame_12.jpg'
# first_frame = cv2.imread(baseline_image_path)
#
# first_region_1 = first_frame[region_1[1]:region_1[3], region_1[0]:region_1[2]]
# first_region_2 = first_frame[region_2[1]:region_2[3], region_2[0]:region_2[2]]
# first_region_3 = first_frame[region_3[1]:region_3[3], region_3[0]:region_3[2]]
#
# first_gray_1 = cv2.cvtColor(first_region_1, cv2.COLOR_BGR2GRAY)
# first_gray_2 = cv2.cvtColor(first_region_2, cv2.COLOR_BGR2GRAY)
# first_gray_3 = cv2.cvtColor(first_region_3, cv2.COLOR_BGR2GRAY)
#
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
#
# output = cv2.VideoWriter('/home/junyan/Documents/Freezer doorOpening_SAMPLE VIDEOS result/refigerator2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     current_region_1 = frame[region_1[1]:region_1[3], region_1[0]:region_1[2]]
#     current_region_2 = frame[region_2[1]:region_2[3], region_2[0]:region_2[2]]
#     current_region_3 = frame[region_3[1]:region_3[3], region_3[0]:region_3[2]]
#
#     current_gray_1 = cv2.cvtColor(current_region_1, cv2.COLOR_BGR2GRAY)
#     current_gray_2 = cv2.cvtColor(current_region_2, cv2.COLOR_BGR2GRAY)
#     current_gray_3 = cv2.cvtColor(current_region_3, cv2.COLOR_BGR2GRAY)
#
#     similarity_1, _ = ssim(first_gray_1, current_gray_1, full=True)
#     similarity_2, _ = ssim(first_gray_2, current_gray_2, full=True)
#     similarity_3, _ = ssim(first_gray_3, current_gray_3, full=True)
#
#     if similarity_1 > 0.5 or similarity_2 > 0.5 or similarity_3 > 0.5:
#         status = "open"
#     else:
#         status = "close"
#
#     cv2.rectangle(frame, (fridge_x1, fridge_y1), (fridge_x2, fridge_y2), (0, 255, 0), 2)
#
#     cv2.putText(frame, status, (fridge_x1, fridge_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     output.write(frame)
#     cv2.imshow("Fridge Status", frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# output.release()
# cv2.destroyAllWindows()




#
# import cv2
# import numpy as np
# from skimage.metrics import structural_similarity as ssim
#
# upperRegion = [(786, 201, 854, 264), (937, 210, 970, 243), (745, 315, 795, 360)]
# lowerRegion = [(983, 493, 1040, 542), (787, 533, 824, 564)]
# fridge_x1, fridge_y1, fridge_x2, fridge_y2 = 599, 100, 1220, 834
#
# cap = cv2.VideoCapture('/home/junyan/Documents/Freezer doorOpening_SAMPLE VIDEOS/2024-09-03 10-54-09 Main Bacteria Room - CAM10 ext 13872.mp4')
#
# ret, baseFrame = cap.read()
# if not ret:
#     print("can't get video")
#     exit()
#
# upper_region = [baseFrame[y1:y2, x1:x2] for (x1, y1, x2, y2) in upperRegion]
# lower_region = [baseFrame[y1:y2, x1:x2] for (x1, y1, x2, y2) in lowerRegion]
#
# upper = [cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) for region in upper_region]
# lower = [cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) for region in lower_region]
#
#
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = cap.get(cv2.CAP_PROP_FPS)
#
# output = cv2.VideoWriter('/home/junyan/Documents/Freezer doorOpening_SAMPLE VIDEOS result/refigerator8.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     curUpperRegion = [frame[y1:y2, x1:x2] for (x1, y1, x2, y2) in upperRegion]
#     curLowerRegion = [frame[y1:y2, x1:x2] for (x1, y1, x2, y2) in lowerRegion]
#
#     cur_upper = [cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) for region in curUpperRegion]
#     cur_lower = [cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) for region in curLowerRegion]
#
#     upperSimilarity = [ssim(base, cur, full=True)[0] for base, cur in zip(upper, cur_upper)]
#     lowerSimilarity = [ssim(base, cur, full=True)[0] for base, cur in zip(lower, cur_lower)]
#
#     upperClosed = any(similarity > 0.6 for similarity in upperSimilarity)
#     upperOpen = not upperClosed
#
#     lowerClosed = any(similarity > 0.5 for similarity in lowerSimilarity)
#     lowerOpen = not lowerClosed
#
#     if upperOpen or lowerOpen:
#         status = "open"
#     else:
#         status = "close"
#
#     cv2.rectangle(frame, (fridge_x1, fridge_y1), (fridge_x2, fridge_y2), (0, 255, 0), 2)
#     cv2.putText(frame, status, (fridge_x1, fridge_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#     output.write(frame)
#     cv2.imshow("Fridge Status", frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# output.release()
# cv2.destroyAllWindows()




import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

region_1 = (955, 263, 1036, 340)
region_2 = (979, 448, 1191, 532)
region_3 = (878, 214, 1041, 251)
fridge_x1, fridge_y1, fridge_x2, fridge_y2 = 865, 62, 1327, 874

cap = cv2.VideoCapture('/home/junyan/Documents/Freezer doorOpening_SAMPLE VIDEOS/2024-09-03 11-01-08 Main Bacteria Room - CAM10 ext 13872.mp4')

baseline_image_path = '/home/junyan/Documents/sample_img/sample_img22/frame_12.jpg'
first_frame = cv2.imread(baseline_image_path)

first_region_1 = first_frame[region_1[1]:region_1[3], region_1[0]:region_1[2]]
first_region_2 = first_frame[region_2[1]:region_2[3], region_2[0]:region_2[2]]
first_region_3 = first_frame[region_3[1]:region_3[3], region_3[0]:region_3[2]]

first_gray_1 = cv2.cvtColor(first_region_1, cv2.COLOR_BGR2GRAY)
first_gray_2 = cv2.cvtColor(first_region_2, cv2.COLOR_BGR2GRAY)
first_gray_3 = cv2.cvtColor(first_region_3, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

kp1_1, des1_1 = sift.detectAndCompute(first_gray_1, None)
kp1_2, des1_2 = sift.detectAndCompute(first_gray_2, None)
kp1_3, des1_3 = sift.detectAndCompute(first_gray_3, None)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

output = cv2.VideoWriter('/home/junyan/Documents/refigerator2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_region_1 = frame[region_1[1]:region_1[3], region_1[0]:region_1[2]]
    current_region_2 = frame[region_2[1]:region_2[3], region_2[0]:region_2[2]]
    current_region_3 = frame[region_3[1]:region_3[3], region_3[0]:region_3[2]]

    current_gray_1 = cv2.cvtColor(current_region_1, cv2.COLOR_BGR2GRAY)
    current_gray_2 = cv2.cvtColor(current_region_2, cv2.COLOR_BGR2GRAY)
    current_gray_3 = cv2.cvtColor(current_region_3, cv2.COLOR_BGR2GRAY)

    kp2_1, des2_1 = sift.detectAndCompute(current_gray_1, None)
    kp2_2, des2_2 = sift.detectAndCompute(current_gray_2, None)
    kp2_3, des2_3 = sift.detectAndCompute(current_gray_3, None)

    def align_region(kp1, des1, kp2, des2, current_region):
        if des1 is not None and des2 is not None and len(des1) > 4 and len(des2) > 4:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) >= 4:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                aligned_region = cv2.warpPerspective(current_region, matrix, (current_region.shape[1], current_region.shape[0]))
                return aligned_region
        return current_region

    current_region_1 = align_region(kp1_1, des1_1, kp2_1, des2_1, current_region_1)
    current_region_2 = align_region(kp1_2, des1_2, kp2_2, des2_2, current_region_2)
    current_region_3 = align_region(kp1_3, des1_3, kp2_3, des2_3, current_region_3)

    similarity_1, _ = ssim(first_gray_1, current_gray_1, full=True)
    similarity_2, _ = ssim(first_gray_2, current_gray_2, full=True)
    similarity_3, _ = ssim(first_gray_3, current_gray_3, full=True)

    if similarity_1 > 0.5 or similarity_2 > 0.5 or similarity_3 > 0.5:
        status = "open"
    else:
        status = "close"

    cv2.rectangle(frame, (fridge_x1, fridge_y1), (fridge_x2, fridge_y2), (0, 255, 0), 2)
    cv2.putText(frame, status, (fridge_x1, fridge_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    output.write(frame)
    cv2.imshow("Fridge Status", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output.release()
cv2.destroyAllWindows()