import argparse
import os
import json

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from submodule import DNN
import glob
from parcopose_from_folder import H36M_2D_COLS, skeletons_to_row, FRAMERATE
from error import calculate_mpjpe, calculate_mAP, h36m_kps, h36m_kp_names
import pandas as pd
import numpy as np
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
    parser.add_argument("--baseline",
                        dest="baseline",
                        action="store_true",
                        help="Enable baseline execution")
    args = parser.parse_args()

    chunk_idx_input = int(args.chunk_id)
    
    print('Loading config %s' % args.config)
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    logfile_path = args.config + '.log'
    
    checkpoint_dir = args.config + '.checkpoints'
    if not os.path.exists(checkpoint_dir):
        print('Creating checkpoint directory % s' % checkpoint_dir)
        os.mkdir(checkpoint_dir)

    # Get chunk amount
    train_chunk_size = config["train_loader"]["batch_size"]
    train_annotations_fp = config["train_dataset"]["annotations_file"]
    with open(train_annotations_fp, 'r') as f:
        train_annotations_json = json.load(f)
    train_images = train_annotations_json["images"]
    print("[Train] len: {} chunk_size: {}".format(len(train_images), train_chunk_size))
    chunk_amount = np.ceil(len(train_images) / train_chunk_size)
    
    # Get test images
    test_chunk_size = config["test_loader"]["batch_size"]
    test_annotations_fp = config["test_dataset"]["annotations_file"]
    with open(test_annotations_fp, 'r') as f:
        test_annotations_json = json.load(f)
    test_images = test_annotations_json["images"]
    all_actions = list(dict.fromkeys([e["file_name"].split("/")[2] for e in test_images]))
    print("[Test] len: {} chunk_size: {}".format(len(test_images), test_chunk_size))

    # Get action of current chunk in test set
    if chunk_idx_input <= 0:
        print("Exit: cannot evalutate catastrofic forgettin in the first chunk")
        exit()
    curr_train_images = train_images[chunk_idx_input*train_chunk_size:(chunk_idx_input+1)*train_chunk_size]
    curr_actions = list(dict.fromkeys([e["file_name"].split("/")[2] for e in curr_train_images]))
    chunk_idx_past = chunk_idx_input - 1
    past_train_images = train_images[chunk_idx_past*train_chunk_size:(chunk_idx_past+1)*train_chunk_size]
    past_actions = list(dict.fromkeys([e["file_name"].split("/")[2] for e in past_train_images]))
    test_actions = []
    for a in past_actions: 
        if a in curr_actions:
            break
        test_actions.append(a)

    if args.baseline: 
        filename = os.path.join(checkpoint_dir, "resforgetactionval_base_dist.csv")
    else: 
        filename = os.path.join(checkpoint_dir, "resforgetactionval_dist.csv")

    if len(test_actions) == 0: 
        if os.path.exists(filename) and chunk_idx_input == 1:
            os.remove(filename)
        print("Exit: no forgetting test_actions in chunk {}".format(chunk_idx_input))
        exit()

    test_images = [e["file_name"] for e in test_images for a in test_actions if a in e["file_name"]]
    test_images_dir = config["test_dataset"]["images_dir"]

    ground_truth_model = config["ground_truth"]["model"]
    ground_truth_folder = config["ground_truth"]["folder"]

    print("Maeve initialization at chunk {}".format(chunk_idx_input))
    if os.path.exists(os.path.join(checkpoint_dir, 'chunk_%d_trt.pth' % chunk_idx_input)):
        os.remove(os.path.join(checkpoint_dir, 'chunk_%d_trt.pth' % chunk_idx_input))
    if args.baseline or chunk_idx_input == -1: 
        dnn = DNN(kind="densenet", suffix="parco", 
                  enable_opt=True).load()
    else:
        dnn = DNN(kind="densenet", suffix="parco", 
                  model_fp=os.path.join(checkpoint_dir, 'chunk_%d.pth' % chunk_idx_input), 
                  enable_opt=True).load()
    print("Maeve inference")
    inference_data = pd.DataFrame()
    time = all_actions.index(test_actions[0]) * len(test_images) * (1.0 / FRAMERATE)
    for id in test_images:
        sub = id.split("/")[0]
        cam = id.split("/")[1]
        action = id.split("/")[2].split(".")[0]
        frame_id = int(id.split("/")[3].split(".")[0])
        print(id, end="\r")
        # Get frame
        filepath = os.path.join(test_images_dir, id)
        
        color_frame = cv2.imread(filepath)
        if color_frame is None: 
            print("Error: empty color frame in {}".format(filepath))
            break
        
        # Get 2D keypoints
        scene = dnn.exec_kp2d(color_frame)
        scene_df = skeletons_to_row(scene)
        scene_df["chunk"] = chunk_idx_input
        scene_df["time"] = time
        scene_df["frames"] = frame_id
        scene_df["sub"] = sub
        scene_df["cam"] = cam
        scene_df["action"] = action
        scene_df["id"] = id
        scene_df["fp"] = filepath
        time += (1.0 / FRAMERATE)
        df = pd.DataFrame(scene_df, columns=["time", "chunk", "frames", "sub", "cam", "action", "id", "fp"] + H36M_2D_COLS)
        inference_data = pd.concat([inference_data, df], axis=0).reset_index(drop=True)

    print("Calculate distance")
    for cam in inference_data["cam"].unique():
        for sub in inference_data["sub"].unique():
            for a in inference_data["action"].unique():
                print(a, end="\r")
                curr_inference_data = inference_data.loc[
                    (inference_data["cam"] == cam) & 
                    (inference_data["sub"] == sub) & 
                    (inference_data["action"] == a)]
                df_train_model = curr_inference_data[h36m_kp_names]
                
                # Read ground truth
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
                df_ground_truth = pd.read_csv(fp)
                df_ground_truth = df_ground_truth.loc[df_ground_truth["frames"].isin(curr_inference_data["frames"]), h36m_kp_names]

                # Prepare data numpy for mpjpe 
                df_train_model = df_train_model.reset_index(drop=True)
                df_ground_truth = df_ground_truth.reset_index(drop=True)

                df_train_model_reshape = df_train_model.values.reshape((1, -1, 2))
                df_ground_truth_reshape = df_ground_truth.values.reshape((1, -1, 2))

                # Calculate mpjpe
                mpjpe, jpe = calculate_mpjpe(df_ground_truth_reshape, df_train_model_reshape)
                map, ap = calculate_mAP(df_ground_truth_reshape, df_train_model_reshape)

                inference_data.loc[(inference_data["cam"] == cam) & (inference_data["sub"] == sub) & (inference_data["action"] == a), 
                                    "MPJPE"] = mpjpe
                inference_data.loc[(inference_data["cam"] == cam) & (inference_data["sub"] == sub) & (inference_data["action"] == a), 
                                    "mAP"] = map
                for i, e in enumerate(jpe): 
                    inference_data.loc[(inference_data["cam"] == cam) & (inference_data["sub"] == sub) & (inference_data["action"] == a), 
                                        "{} JPE".format(h36m_kps[i])] = e
                for i, e in enumerate(ap): 
                    inference_data.loc[(inference_data["cam"] == cam) & (inference_data["sub"] == sub) & (inference_data["action"] == a), 
                                        "{} AP".format(h36m_kps[i])] = e
    
    if os.path.exists(filename):
        if chunk_idx_input == 1: 
            data = pd.DataFrame()
        else: 
            data = pd.read_csv(filename, index_col=0)
    else: 
        data = pd.DataFrame()

    data = pd.concat([data, inference_data], axis=0).reset_index(drop=True)
    data.to_csv(filename)
    write_log_entry_error(logfile_path, inference_data, chunk_idx_input)