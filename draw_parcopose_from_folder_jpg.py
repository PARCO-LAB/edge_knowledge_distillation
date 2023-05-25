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
parser.add_argument("--model-name", 
                    "-n", 
                    dest="name", 
                    required=True, 
                    help="Model name (parcopose or trtpose)")
args = parser.parse_args()


def get_elapsed(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)

def init(model_name):
    print("Maeve initialization")
    start = time.time()
    if model_name == "trtpose":
        dnn = DNN(kind="densenet").load()
    elif model_name == "parcoposeh36m":
        dnn = DNN(kind="densenet", suffix="parcoh36m").load()
    else:
        dnn = DNN(kind="densenet", suffix="parco").load()
    end = time.time()
    print("initialization elapsed time: {}".format(get_elapsed(start, end)))
    return dnn


def loop(dnn : DNN, folderpath, name, stream_type="video"):
    print("Maeve loop")
    
    loop_start = time.time()
    i = 0
    out_stream = None
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

        if i == 10: 
            cv2.imwrite("prova.png", color_frame)

        if stream_type == "folder":
            if i == 0:
                folder_name = folderpath + "_{}".format(name)
                if not os.path.isdir(folder_name):
                    os.mkdir(folder_name)
            cv2.imwrite(os.path.join(folder_name, "{}.png".format(i)), color_frame)
        else: 
            if i == 0:
                filename = "{}.mp4".format(folderpath + "_{}".format(name))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                w, h = color_frame.shape[1], color_frame.shape[0]
                out_stream = cv2.VideoWriter(filename, fourcc, 15.0, (w, h))
            if out_stream:
                color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGRA2BGR)
                out_stream.write(color_frame)

        print(i, end="\r")
        i += 1
    loop_end = time.time()
    print("loop elapsed time: {}".format(get_elapsed(loop_start, loop_end)))
    print("amount of frame processed: {}".format(i))


def main():
    loop(init(args.name), args.folder, args.name, "video")
    # loop(init(args.name), args.folder, args.name, "folder")


if __name__ == "__main__":
    main()
