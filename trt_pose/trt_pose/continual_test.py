import argparse
import os
import json
import pprint

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from submodule import DNN
import glob
from parcopose_from_folder import H36M_2D_COLS, skeletons_to_row, FRAMERATE
from error import calculate_mpjpe, calculate_mAP, h36m_kps, h36m_kp_names
import pandas as pd
import cv2


def write_log_entry_error(logfile, error_info, chunk_i):
    with open(logfile, 'a+') as f:
        jpe_report = ""
        for kp_name in h36m_kps: 
            jpe_report += "{}: {:.1f}; ".format(kp_name, error_info["{} JPE".format(kp_name)].mean())
        ap_report = ""
        for kp_name in h36m_kps: 
            ap_report += "{}: {:.1f}; ".format(kp_name, error_info["{} AP".format(kp_name)].mean())
        report = "{}: {{ MPJPE: {}; mAP: {}; ".format(chunk_i, error_info["MPJPE"].mean(), error_info["mAP"].mean()) + jpe_report + ap_report + "}}"
        print(report)
        f.write(report + '\n')
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument("--chunk-id",
                        "-i",
                        dest="chunk_id",
                        required=True,
                        help="Chunk id")
    args = parser.parse_args()

    chunk_idx_input = int(args.chunk_id)
    
    print('Loading config %s' % args.config)
    with open(args.config, 'r') as f:
        config = json.load(f)
        pprint.pprint(config)
        
    logfile_path = args.config + '.log'
    
    checkpoint_dir = args.config + '.checkpoints'
    if not os.path.exists(checkpoint_dir):
        print('Creating checkpoint directory % s' % checkpoint_dir)
        os.mkdir(checkpoint_dir)
    
    subjects = config["ground_truth"]["subjects"]
    cameras = config["ground_truth"]["cameras"]
    ground_truth_model = config["ground_truth"]["model"]
    ground_truth_folder = config["ground_truth"]["folder"]

    print("Maeve initialization at chunk {}".format(chunk_idx_input))
    if os.path.exists(os.path.join(checkpoint_dir, 'chunk_%d_trt.pth' % chunk_idx_input)):
        os.remove(os.path.join(checkpoint_dir, 'chunk_%d_trt.pth' % chunk_idx_input))
    dnn = DNN(kind="densenet", suffix="parco", model_fp=os.path.join(checkpoint_dir, 'chunk_%d.pth' % chunk_idx_input)).load()
    print("Maeve inference")
    inference_data = {}
    for cam in cameras: 
        inference_data[cam] = {}
        for sub in subjects:
            inference_data[cam][sub] = {}
            folder_with_actions = os.path.join(ground_truth_folder, sub, cam)
            actions = glob.glob(os.path.join(folder_with_actions, "*" + cam + "*"))
            actions = [a for a in actions if os.path.isdir(a) and "ALL" not in a]
            for a in actions: 
                print(a, end="\r")
                i = 0
                df = pd.DataFrame(columns=["time", "frame"] + H36M_2D_COLS)
                while True:
                    # Get frame
                    filepath = os.path.join(a, "{}.png".format(i))
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
                    i += 1
                    if i > 4: 
                        break
                inference_data[cam][sub][os.path.basename(a).replace(" ", "")] = df

    print("Calculate distance")
    results_dist = {
        "camera": [],
        "subject": [],
        "action": [],
        "MPJPE": [],
        "mAP": [],
    }
    for kp in h36m_kps: 
        results_dist["{} JPE".format(kp)] = []
        results_dist["{} AP".format(kp)] = []
    for cam in inference_data:
        for sub in inference_data[cam]:
            for a in inference_data[cam][sub]:
                print(a, end="\r")
                df_train_model = inference_data[cam][sub][a][h36m_kp_names]

                action_name = a.split(".")[0]
                fp = os.path.join(
                    ground_truth_folder, sub, ground_truth_model, 
                    "{}.{}.csv".format(action_name, cam))
                if any(c.isdigit() for c in action_name) and not os.path.exists(fp):
                    fp = os.path.join(
                        ground_truth_folder, sub, ground_truth_model, 
                        "{} {}.{}.csv".format(action_name[:-1], action_name[-1], cam))
                if not os.path.exists(fp): 
                    print("Error: file {} not exist".format(fp))
                    continue

                df_ground_truth = pd.read_csv(fp)[h36m_kp_names]

                df_ground_truth = df_ground_truth.iloc[:min(len(df_ground_truth.index), len(df_train_model.index))]
                df_train_model = df_train_model.iloc[:min(len(df_ground_truth.index), len(df_train_model.index))]
                df_ground_truth_reshape = df_ground_truth.values.reshape((len(df_ground_truth.index), -1, 2))
                df_train_model_reshape = df_train_model.values.reshape((len(df_train_model.index), -1, 2))

                mpjpe, jpe = calculate_mpjpe(df_ground_truth_reshape, df_train_model_reshape)
                map, ap = calculate_mAP(df_ground_truth_reshape, df_train_model_reshape)
                results_dist["camera"].append(cam)
                results_dist["subject"].append(sub)
                results_dist["action"].append(action_name)
                results_dist["MPJPE"].append(mpjpe)
                results_dist["mAP"].append(map)
                for i, e in enumerate(jpe): 
                    results_dist["{} JPE".format(h36m_kps[i])].append(e)
                for i, e in enumerate(ap): 
                    results_dist["{} AP".format(h36m_kps[i])].append(e)
    results_dist_df = pd.DataFrame(results_dist)
    results_dist_df.to_csv(os.path.join(checkpoint_dir, "res_dist_{}.csv".format(chunk_idx_input)))
    write_log_entry_error(logfile_path, results_dist_df, chunk_idx_input)