import numpy as np

from yolox.tracker.byte_tracker import BYTETracker
import cv2
import os
import torch
import argparse
from yolox.utils.visualize import plot_tracking


import argparse
import os
import os.path as osp
import time
import cv2
import torch

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
#
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/junyan/Documents/train_model/yolov5/runs/train/exp2/weights/best.pt')
#
# def get_detections(frame):
#     results = model(frame)
#
#     dets = results.xyxy[0]
#
#     class_conf, _ = results.pred[0][:, 5:].max(1)
#     class_pred = results.pred[0][:, 5:].argmax(1)
#     print("Class Confidence: ", class_conf)
#     dets_with_class_info = torch.cat((dets, class_conf.unsqueeze(1), class_pred.unsqueeze(1).float()), 1)
#     return dets_with_class_info


model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/junyan/Documents/train_model/yolov5/runs/train/exp2/weights/best.pt')

def get_detections(frame):
    results = model(frame)
    dets = results.xyxy[0]
    class_conf, _ = results.pred[0][:, 5:].max(1)
    class_pred = results.pred[0][:, 5:].argmax(1)
    dets_with_class_info = torch.cat((dets, class_conf.unsqueeze(1), class_pred.unsqueeze(1).float()), 1)
    #dets_with_class_info = dets_with_class_info[class_conf > 0.5]
    print("Class Confidence: ", class_conf)
    return dets_with_class_info


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="./videos/bytetrack_example.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser

args = make_parser().parse_args()
tracker = BYTETracker(args,frame_rate=30)

def track_and_save_video(video_path, output_video_path, output_txt_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_id = 0
    results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        dets = get_detections(frame)
        print(dets)
        online_targets = tracker.update(dets, [height, width], (height, width))
        print(online_targets)
        online_tlwhs = []
        online_ids = []
        online_scores = []

        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(t.score)

            results.append(f"{frame_id},{tid},{tlwh[0]},{tlwh[1]},{tlwh[2]},{tlwh[3]},{t.score:.2f},-1,-1,-1\n")
            #print(results)
        online_im = plot_tracking(frame, online_tlwhs, online_ids, frame_id=frame_id, fps=fps)
        vid_writer.write(online_im)

    cap.release()
    vid_writer.release()

    with open(output_txt_path, 'w') as f:
        f.writelines(results)

if args.fuse:
    logger.info("\tFusing model...")
    model = fuse_model(model)

if args.fp16:
    model = model.half()  # to FP16


track_and_save_video('/home/junyan/Documents/2024-01-09.mp4', '/home/junyan/Documents/bytetrack_output/bytetrack_video_latest.mp4', '/home/junyan/Documents/bytetrack_output/bytetrack_txt_latest.txt')