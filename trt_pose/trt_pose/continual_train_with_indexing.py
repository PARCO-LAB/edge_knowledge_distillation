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
        
        
def save_checkpoint_name(model, directory, name):
    if not os.path.exists(directory):
        os.mkdir(directory)
    filename = os.path.join(directory, name)
    print('Saving checkpoint to %s' % filename)
    torch.save(model.state_dict(), filename)

    
def write_log_entry(logfile, epoch, train_loss, lr, chunk_i):
    with open(logfile, 'a+') as f:
        logline = '%d: %d, %f, %f' % (chunk_i, epoch, train_loss, lr)
        print(logline)
        f.write(logline + '\n')

        
device = torch.device('cuda')

def train(config_file):
    print('Loading config %s' % config_file)
    with open(config_file, 'r') as f:
        config = json.load(f)
        pprint.pprint(config)

    # custom add: train only if it is trainable (i.e.: not last chunk)
    if 'trainable' in config and config['trainable'] != 1:
        exit()
        
    logfile_path = config_file + '.log'
    
    checkpoint_dir = config['checkpoint_dir']
    if not os.path.exists(checkpoint_dir):
        print('Creating checkpoint directory % s' % checkpoint_dir)
        os.mkdir(checkpoint_dir)
    
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
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        **config["train_loader"]
    )
    
    model = MODELS[config['model']['name']](**config['model']['kwargs']).to(device)
    
    if "initial_state_dict_name" in config['model']:
        path_weights = f"{checkpoint_dir}/{config['model']['initial_state_dict']}"
        print('Loading initial weights from fname %s' % path_weights)
        model.load_state_dict(torch.load(path_weights))
    elif "initial_state_dict" in config['model']:
        print('Loading initial weights from dict %s' % config['model']['initial_state_dict'])
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
    first_time = True
    for image_train, cmap_train, paf_train, mask_train in tqdm.tqdm(iter(train_loader)): # Chunk
        if first_time:
            first_time = False
        else:
            # it should NEVER do two iterations. Notify so user is aware
            raise Exception("Error: two iterations of the data loader, but we are in the continual domain, so this is probably a config error!")
        
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

        save_checkpoint_name(model, checkpoint_dir, config['final_state_dict_name']) 
    
    # here if some cleanup is needed
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()

    # get all the json and to the train for each one
    import glob
    # sorted for lexicographic ordering
    experiment_files = sorted(glob.glob(os.path.join(args.config, "*.json")))
    for experiment in experiment_files:
        train(experiment)