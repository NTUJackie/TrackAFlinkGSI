

## Dependencies
* PyTorch = 1.12.0

## Quick Start

- python strong_sort.py MOT17 val --AFLink --GSI

Introduction:

In order to improve the tracking accuracy for featureless people, this algorithm has modified the strongsort++ algorithm. The first step is to replace the original strongsort++ tracking model with bytetrack. In order to use featureless people, we replaced bytetrack's yolox model with the self-trained target detection model yolov5m, and modified some of the original bytetrack code to adapt to the new model. At the same time, we replaced the kalman filter in the original bytetrack code with the NSA Kalman filter to improve accuracy. Then the bytetrack code running results are brought into AFlink and GSI to further improve accuracy.



```
