import argparse
import subprocess
import torch
import torchvision
import os
import torch.optim
import tqdm
import apex
import apex.amp as amp
import time
import json
import pprint
import torch.nn.functional as F
from coco import CocoDataset, CocoHumanPoseEval
from models import MODELS
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from submodule import DNN
import glob
from parcopose_from_folder import H36M_2D_COLS, skeletons_to_row, FRAMERATE
from error import calculate_mpjpe, calculate_mAP, h36m_kps, h36m_kp_names
import pandas as pd
import cv2

OPTIMIZERS = {
    'SGD': torch.optim.SGD,
    'Adam': torch.optim.Adam
}

EPS = 1e-6

def set_lr(optimizer, lr):
    for p in optimizer.param_groups:
        p['lr'] = lr
        
        
def save_checkpoint(model, directory, chunk):
    if not os.path.exists(directory):
        os.mkdir(directory)
    filename = os.path.join(directory, 'chunk_%d.pth' % chunk)
    print('Saving checkpoint to %s' % filename)
    torch.save(model.state_dict(), filename)

    
def write_log_entry(logfile, epoch, train_loss, lr, chunk_i):
    with open(logfile, 'a+') as f:
        logline = '%d: %d, %f, %f' % (chunk_i, epoch, train_loss, lr)
        print(logline)
        f.write(logline + '\n')

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
        
device = torch.device('cuda')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    
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
        
    # LOAD DATASETS
    
    train_dataset_kwargs = config["train_dataset"]
    train_dataset_kwargs['transforms'] = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(**config['color_jitter']),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset_kwargs = config["test_dataset"]
    test_dataset_kwargs['transforms'] = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if 'evaluation' in config:
        evaluator = CocoHumanPoseEval(**config['evaluation'])
    
    train_dataset = CocoDataset(**train_dataset_kwargs)
    test_dataset = CocoDataset(**test_dataset_kwargs)
    
    part_type_counts = test_dataset.get_part_type_counts().float().cuda()
    part_weight = 1.0 / part_type_counts
    part_weight = part_weight / torch.sum(part_weight)
    paf_type_counts = test_dataset.get_paf_type_counts().float().cuda()
    paf_weight = 1.0 / paf_type_counts
    paf_weight = paf_weight / torch.sum(paf_weight)
    paf_weight /= 2.0
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        **config["train_loader"]
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        **config["test_loader"]
    )
    
    model = MODELS[config['model']['name']](**config['model']['kwargs']).to(device)
    
    if "initial_state_dict" in config['model']:
        print('Loading initial weights from %s' % config['model']['initial_state_dict'])
        model.load_state_dict(torch.load(config['model']['initial_state_dict']))
    
    optimizer = OPTIMIZERS[config['optimizer']['name']](model.parameters(), **config['optimizer']['kwargs'])
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if 'mask_unlabeled' in config and config['mask_unlabeled']:
        print('Masking unlabeled annotations')
        mask_unlabeled = True
    else:
        mask_unlabeled = False
    lr = config['optimizer']['kwargs']["lr"]
    set_lr(optimizer, lr)
    # for epoch in range(config["epochs"]):
        
    train_chunk_size = config["train_loader"]["batch_size"]
    chunk_i = 0
    for image_train, cmap_train, paf_train, mask_train in tqdm.tqdm(iter(train_loader)): # Chunk
        train_loss = 0.0
        model = model.train()
        for epoch in range(config["epochs"]):
            for batch_i in range(int(train_chunk_size / config["batch_size"])):
                image = image_train[batch_i*config["batch_size"]:(batch_i + 1)*config["batch_size"]].to(device)
                cmap = cmap_train[batch_i*config["batch_size"]:(batch_i + 1)*config["batch_size"]].to(device)
                paf = paf_train[batch_i*config["batch_size"]:(batch_i + 1)*config["batch_size"]].to(device)
                
                if mask_unlabeled:
                    mask = mask_train[batch_i*config["batch_size"]:(batch_i + 1)*config["batch_size"]].to(device).float()
                else:
                    mask = torch.ones_like(mask_train)[batch_i*config["batch_size"]:(batch_i + 1)*config["batch_size"]].to(device).float()
                optimizer.zero_grad()
                cmap_out, paf_out = model(image)
                #print("out: ", cmap_out.shape, paf_out.shape, image.shape)
                #print("gt: ", cmap.shape, paf.shape)

                cmap_mse = torch.mean(mask * (cmap_out - cmap)**2)
                paf_mse = torch.mean(mask * (paf_out - paf)**2)
            
                loss = cmap_mse + paf_mse
                
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()

                optimizer.step() # TODO: try to untab

                train_loss += float(loss)
            
            train_loss /= int(train_chunk_size / config["batch_size"])
            
            write_log_entry(logfile_path, epoch, train_loss, lr, chunk_i)

        save_checkpoint(model, checkpoint_dir, chunk_i)
        print("Maeve initialization at chunk {}".format(chunk_i))
        if os.path.exists(os.path.join(checkpoint_dir, 'chunk_%d_trt.pth' % chunk_i)):
            os.remove(os.path.join(checkpoint_dir, 'chunk_%d_trt.pth' % chunk_i))
        dnn = DNN(kind="densenet", suffix="parco", model_fp=os.path.join(checkpoint_dir, 'chunk_%d.pth' % chunk_i)).load()
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
        results_dist_df.to_csv(os.path.join(checkpoint_dir, "res_dist_{}.csv".format(chunk_i)))
        write_log_entry_error(logfile_path, results_dist_df, chunk_i)
        chunk_i += 1
        