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

        
device = torch.device('cuda')

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
    
    if "initial_state_dict" in config['model'] and chunk_idx_input == 0:
        print('Loading initial weights from %s' % config['model']['initial_state_dict'])
        model.load_state_dict(torch.load(config['model']['initial_state_dict']))
    else: 
        fp = os.path.join(checkpoint_dir, 'chunk_%d.pth' % (chunk_idx_input-1))
        print('Loading initial weights from %s' % fp)
        model.load_state_dict(torch.load(fp))
    
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
        if chunk_i != chunk_idx_input:
            chunk_i += 1
            continue
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
        exit()
        