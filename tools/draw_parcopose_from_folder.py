"""Generate Keypoints

Example script that generate the resulting 2D keypoints of Realsense images from a file stream.
"""

import os
import sys
import time
import json
import numpy as np
import datetime
import argparse
import pandas as pd
import cv2

from submodule import DNN, DrawSkeleton


parser = argparse.ArgumentParser(description="ParcoPose from folder", epilog="PARCO")
parser.add_argument("--folder", 
                    "-f", 
                    dest="folder", 
                    required=True, 
                    help="Folder with images")
args = parser.parse_args()


def get_elapsed(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)

def init():
    print("Maeve initialization")
    start = time.time()
    dnn = DNN(kind="densenet", suffix="parco").load()
    # dnn = DNN(kind="densenet").load()
    end = time.time()
    print("initialization elapsed time: {}".format(get_elapsed(start, end)))
    return dnn


def loop(dnn : DNN, folderpath):
    print("Maeve loop")

    new_folderpath = folderpath + "_parcopose"
    # new_folderpath = folderpath + "_trtpose"
    if not os.path.exists(new_folderpath):
        os.mkdir(new_folderpath)
    
    loop_start = time.time()
    i = 0
    while True:
        # Get frame
        filepath = os.path.join(folderpath, "frame_{}.jpg".format(i))
        if not os.path.exists(filepath):
            break
        
        color_frame = cv2.imread(filepath)
        if color_frame is None: 
            break
        
        # Get 2D keypoints
        s = time.time()
        scene = dnn.exec_kp2d(color_frame)
        e = time.time()
        color_frame = DrawSkeleton.on_2d_cv(color_frame, scene, i)
        print("elapsed: {:.5f} ms; framerate: {:.3f}".format((e - s) * 1000.0,  1 / (e - s)))

        cv2.imwrite(os.path.join(new_folderpath, "frame_{}.jpg".format(i)), color_frame)

        print(i, end="\r")
        i += 1
    loop_end = time.time()
    print("loop elapsed time: {}".format(get_elapsed(loop_start, loop_end)))
    print("amount of frame processed: {}".format(i))


def main():
    loop(init(), args.folder)


if __name__ == "__main__":
    main()
