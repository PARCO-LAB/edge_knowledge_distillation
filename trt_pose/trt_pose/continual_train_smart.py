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
import random
import pandas as pd

SEED=213445
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

OPTIMIZERS = {
    'SGD': torch.optim.SGD,
    'Adam': torch.optim.Adam
}

EPS = 1e-6

def set_lr(optimizer, lr):
    for p in optimizer.param_groups:
        p['lr'] = lr
        
        
def save_checkpoint(model, directory, chunk, selector):
    if not os.path.exists(directory):
        os.mkdir(directory)
    filename = os.path.join(directory, 'chunk_{}_{}.pth'.format(selector, chunk))
    print('Saving checkpoint to %s' % filename)
    torch.save(model.state_dict(), filename)

    
def write_log_entry(logfile, epoch, train_loss, lr, chunk_i):
    with open(logfile, 'a+') as f:
        logline = '%d: %d, %f, %f' % (chunk_i, epoch, train_loss, lr)
        print(logline)
        f.write(logline + '\n')

        
device = torch.device('cuda')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument("--chunk-id",
                        "-i",
                        dest="chunk_id",
                        required=True,
                        help="Chunk id")
    parser.add_argument("--selector",
                        dest="selector",
                        required=True,
                        help="Metric selector")
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
    res_val = pd.read_csv(os.path.join(checkpoint_dir, 'resval_{}.csv'.format(args.selector)))

    
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
    
    if 'evaluation' in config:
        evaluator = CocoHumanPoseEval(**config['evaluation'])
    
    train_dataset = CocoDataset(**train_dataset_kwargs)
    
    model = MODELS[config['model']['name']](**config['model']['kwargs']).to(device)
    
    if "initial_state_dict" in config['model'] and chunk_idx_input == 0:
        print('Loading initial weights from %s' % config['model']['initial_state_dict'])
        model.load_state_dict(torch.load(config['model']['initial_state_dict']))
    else: 
        fp = os.path.join(checkpoint_dir, 'chunk_{}_{}.pth'.format(args.selector, chunk_idx_input-1))
        print('Loading initial weights from %s' % fp)
        model.load_state_dict(torch.load(fp))
    if args.selector == "svd":
        if res_val['eigen'].iloc[-1] < 14:
            save_checkpoint(model, checkpoint_dir, chunk_idx_input, args.selector)
            exit()
    else:
        if np.mean(res_val['MPJPE'].iloc[-1500:-1]) < 10:
            save_checkpoint(model, checkpoint_dir, chunk_idx_input, args.selector)
            exit()

    
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
    start_frame = chunk_idx_input * train_chunk_size
    end_frame = (chunk_idx_input + 1) * train_chunk_size

    image_train, cmap_train, paf_train, mask_train = [], [], [], []
    for i, (image_i, cmap_i, paf_i, mask_i) in enumerate(train_dataset): # Chunk
        if i >= end_frame:
            break
        print(i, end="\r")
        if i >= start_frame: 
            image_train.append(image_i.numpy())
            cmap_train.append(cmap_i.numpy())
            paf_train.append(paf_i.numpy())
            mask_train.append(mask_i.numpy())

    image_train, cmap_train, paf_train, mask_train = np.array(image_train), np.array(cmap_train), np.array(paf_train), np.array(mask_train)
    image_train, cmap_train, paf_train, mask_train = torch.FloatTensor(image_train), torch.FloatTensor(cmap_train), torch.FloatTensor(paf_train), torch.FloatTensor(mask_train)
    train_loss = 0.0
    model = model.train()
    for epoch in range(config["epochs"]):
        for batch_i in range(max(1, int(train_chunk_size / config["batch_size"]))):
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
        
        train_loss /= max(1, int(train_chunk_size / config["batch_size"]))
        
        write_log_entry(logfile_path, epoch, train_loss, lr, chunk_idx_input)

    save_checkpoint(model, checkpoint_dir, chunk_idx_input, args.selector) 
        