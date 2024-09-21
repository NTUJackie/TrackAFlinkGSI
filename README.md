

## Dependencies
* PyTorch = 1.12.0

## Quick Start

- python strong_sort.py MOT17 val --AFLink --GSI

Introduction:

In order to improve the tracking accuracy for featureless people, this algorithm has modified the strongsort++ algorithm. The first step is to replace the original strongsort++ tracking model with bytetrack. In order to adapt to featureless people, we replaced bytetrack's yolox model with the self-trained target detection model yolov5m, and modified some of the original bytetrack code to adapt to the new model. At the same time, we replaced the kalman filter in the original bytetrack code with the NSA Kalman filter to improve accuracy. Then the bytetrack code running results are brought into AFlink and GSI to further improve accuracy.

Note:

This project involves many other functions, such as judging the refrigerator switch function, which does not use deep learning, but only image processing. The code can be found in refrigerator_change_detection.py. There is also a fall detection function, which is to improve the detection of people standing under the camera as falling. This code trains the new fall detection model to improve the accuracy.

```
