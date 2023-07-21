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

from submodule import DNN

FRAMERATE = 50

H36M_HUMAN_PARTS_MAP = {
    "Hip": "mid_hip",
    "RHip": "right_hip",
    "RKnee": "right_knee",
    "RAnkle": "right_ankle",
    "LHip": "left_hip",
    "LKnee": "left_knee",
    "LAnkle": "left_ankle",
    "LShoulder": "left_shoulder",
    "LElbow": "left_elbow",
    "LWrist": "left_wrist",
    "RShoulder": "right_shoulder",
    "RElbow": "right_elbow",
    "RWrist": "right_wrist",
}

# H36M_2D_COLS = [
#     v for k in H36M_HUMAN_PARTS_MAP for v in ["{}:U".format(k), "{}:V".format(k)]
# ]

H36M_2D_COLS = [
    v for k in H36M_HUMAN_PARTS_MAP for v in ["{}:U".format(k), "{}:V".format(k), "{}:ACC".format(k)]
]


def get_elapsed(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)

def skeletons_to_row(kp2d_dict): 
    if len(kp2d_dict) == 0:
        return pd.DataFrame()
    skeletons_length = np.array([len(skeleton) for skeleton in kp2d_dict])
    skeleton_with_more_kp = kp2d_dict[np.argmax(skeletons_length)]

    # Calculate mid_hip
    if "right_hip" in skeleton_with_more_kp and "left_hip" in skeleton_with_more_kp:
        skeleton_with_more_kp["mid_hip"] = [
            (skeleton_with_more_kp["right_hip"][i] + skeleton_with_more_kp["right_hip"][i]) / 2
            for i in range(3)
        ]

    data = []
    for part in H36M_HUMAN_PARTS_MAP:
        parcopose_part = H36M_HUMAN_PARTS_MAP[part]
        if parcopose_part in skeleton_with_more_kp: 
            #data.extend(skeleton_with_more_kp[parcopose_part][:2])
            data.extend(skeleton_with_more_kp[parcopose_part][:3])
        else:
            #data.extend([np.nan] * 2)
            data.extend([np.nan] * 3)
        
    return pd.DataFrame(np.array([data]), columns=H36M_2D_COLS)


def init(model_name):
    print("Maeve initialization")
    start = time.time()
    if "parco" in model_name:
        dnn = DNN(kind="densenet", suffix=model_name, enable_opt=True).load()
    else:
        dnn = DNN(kind="densenet", enable_opt=True).load()
    end = time.time()
    print("initialization elapsed time: {}".format(get_elapsed(start, end)))
    return dnn


def loop(dnn : DNN, folderpath):
    print("Maeve loop")
    
    loop_start = time.time()
    i = 0
    df = pd.DataFrame(columns=["time", "frame"] + H36M_2D_COLS)
    while True:
        # Get frame
        #filepath = os.path.join(folderpath, "frame_{}.jpg".format(i))
        filepath = os.path.join(folderpath, "{}.png".format(i))
        if not os.path.exists(filepath):
            break
        
        color_frame = cv2.imread(filepath)
        if color_frame is None: 
            break
        
        # Get 2D keypoints
        scene = dnn.exec_kp2d(color_frame)
        scene_df = skeletons_to_row(scene)
        scene_df["frame"] = i
        scene_df["time"] = i * (1.0 / FRAMERATE)
        df = pd.concat([df, scene_df], axis=0)

        print(i, end="\r")
        # if i > 100: 
        #     break
        i += 1
    loop_end = time.time()
    print("loop elapsed time: {}".format(get_elapsed(loop_start, loop_end)))
    print("amount of frame processed: {}".format(i))
    return df


def main():
    parser = argparse.ArgumentParser(description="ParcoPose from folder", epilog="PARCO")
    parser.add_argument("--folder", 
                        "-f", 
                        dest="folder", 
                        required=True, 
                        nargs="+",
                        help="Folder with images")
    parser.add_argument("--output-folder", 
                        "-o", 
                        dest="output_folder", 
                        required=False,
                        default=None, 
                        help="Output folder with csv")
    parser.add_argument("--model-name", 
                        "-n", 
                        dest="name", 
                        required=True, 
                        help="Model name (parcopose or trtpose)")
    args = parser.parse_args()
    m = init(args.name)
    for folder in args.folder:
        res = loop(m, folder)
        res = res[["time", "frame"] + H36M_2D_COLS]
        if folder[-1] == "/":
            folder = folder[:-1]
        if args.output_folder is None: 
            res.to_csv("{}.csv".format(folder), index=False)
        else: 
            name = os.path.basename(folder)
            res.to_csv(os.path.join(args.output_folder, "{}.csv".format(name)), index=False)


if __name__ == "__main__":
    main()
