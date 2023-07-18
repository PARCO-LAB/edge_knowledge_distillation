import argparse
import os
import json

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from submodule import DNN
import glob
from parcopose_from_folder import H36M_2D_COLS, skeletons_to_row, FRAMERATE
# from error import calculate_mpjpe, calculate_mAP, h36m_kps, h36m_kp_names
import pandas as pd
import torch

from coco import CocoDataset, CocoHumanPoseEval
import tqdm
import torchvision
import cv2
import numpy as np

h36m_kp_names = [
    "LShoulder:U", "LShoulder:V", "RShoulder:U", "RShoulder:V",
    "LElbow:U", "LElbow:V", "RElbow:U", "RElbow:V", 
    "LWrist:U", "LWrist:V", "RWrist:U", "RWrist:V", 
    "LHip:U", "LHip:V", "RHip:U", "RHip:V", 
    "LKnee:U", "LKnee:V","RKnee:U", "RKnee:V", 
    "LAnkle:U", "LAnkle:V", "RAnkle:U", "RAnkle:V"
]
h36m_kps = list(dict.fromkeys(([kp.split(":")[0] for kp in h36m_kp_names])))


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

def test(config_file, outfolder, order, baseline=False):
    print('Loading config %s' % config_file)
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    checkpoint_dir = config['checkpoint_dir']
    if not os.path.exists(checkpoint_dir):
        print('Creating checkpoint directory % s' % checkpoint_dir)
        os.mkdir(checkpoint_dir)

    # Get test images
    test_dataset_kwargs = config["test_dataset"]
    test_images_dir = config["test_dataset"]["images_dir"]
    test_dataset_kwargs['transforms'] = np.array
    # avoid transformation, but keep it here
    # test_dataset_kwargs['transforms'] = torchvision.transforms.Compose([
    #         # torchvision.transforms.ColorJitter(**config['color_jitter']),
    #         torchvision.transforms.ToTensor()
    #         # torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    # custom return to bypass the logic of coco dataset and do a simple "cv2.imread"
    test_dataset_kwargs["custom_return"] = True
    test_dataset = CocoDataset(**test_dataset_kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        **config["test_loader"]
    )

    path_to_model = ""
    if "initial_state_dict_name" in config:
        path_to_model  = f"{checkpoint_dir}/{config['initial_state_dict_name']}"
    elif "initial_state_dict" in config['model']:
        path_to_model = config['model']['initial_state_dict']
    print(f"LOADING: {path_to_model}")
    dnn = DNN(kind="densenet", suffix="parco", 
                model_fp=path_to_model,
                enable_opt=False).load()
    inference_data = pd.DataFrame()
    
    # for image_test, cmap_test, paf_test, mask_test, id, ground_truth in tqdm.tqdm(iter(test_loader)): # Chunk
    for image_test, id in tqdm.tqdm(iter(test_loader)): # Chunk
        # Get frame
        # filepath=''

        # the eventual batch size
        for image_id in range(image_test.shape[0]):
            # color_frame = image_test
            filepath = os.path.join(test_images_dir, id[image_id])
            color_frame = np.asarray(image_test)[image_id,:,:,:]
            if color_frame is None: 
                print("Error: empty color frame in {}".format(filepath))
                break
            current_id = id[image_id]

            # Get 2D keypoints
            scene = dnn.exec_kp2d(color_frame)
            # this to factor we are already resizing
            scene_df = skeletons_to_row(scene)
            sub = current_id.split("/")[0]
            cam = current_id.split("/")[1]
            action = current_id.split("/")[2].split(".")[0]
            frame_id = int(current_id.split("/")[3].split(".")[0])
            scene_df["global_id"] = order[current_id]
            scene_df["frames"] = frame_id
            scene_df["sub"] = sub
            scene_df["cam"] = cam
            scene_df["action"] = action
            scene_df["id"] = current_id
            scene_df["time"] = int(order[current_id])/50
            df = pd.DataFrame(scene_df, columns=["global_id","frames","sub","cam","action","id","time"] + H36M_2D_COLS)
            inference_data = pd.concat([inference_data, df], axis=0).reset_index(drop=True)
    
    filename = os.path.join(outfolder, os.path.splitext(os.path.basename(config_file))[0] + ".csv")
    
    # choose to overwrite previous value
    # if os.path.exists(filename):
    #     data = pd.read_csv(filename, index_col=0)
    # else: 
    #     data = pd.DataFrame()
    data = pd.DataFrame()
    
    data = pd.concat([data, inference_data], axis=0).reset_index(drop=True)
    data.to_csv(filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help="Folder to load configuration")
    parser.add_argument('output', help="Folder to store CSV output (same name as input chunk)")
    parser.add_argument('order', help="File for ordering")
    parser.add_argument('--file', help="If specified, load only THAT configuration",
                        default=argparse.SUPPRESS)
    args = parser.parse_args()


    import csv
    order_indexes = {}
    order_indexes_arr = []
    file_indexes = {}
    file_indexes_arr = []
    with open(args.order) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            order_indexes[row[1]] = row[0]
            order_indexes_arr.append(row[1])
    
    if 'file' in args:
        test(os.path.join(args.config, args.file), args.output, order_indexes)
    else:
        # get all the json and to the train for each one
        import glob
        # sorted for lexicographic ordering
        experiment_files = sorted(glob.glob(os.path.join(args.config, "*.json")))
        for experiment in experiment_files:
            test(experiment, args.output, order_indexes)