#!/usr/bin/python

import random
import argparse

parser = argparse.ArgumentParser(description="Generate index for sampling", epilog="PARCO")

parser.add_argument("--input", help="Blueprint soruce file", default="blueprint.json")
parser.add_argument("--output", help="Output folder for blueprint generation")
parser.add_argument("--train-annotations", default=argparse.SUPPRESS, help="Folder where to find annotation")
parser.add_argument("--test-annotations", default=argparse.SUPPRESS, help="Folder where to find annotation")
parser.add_argument("--checkpoint-dir", help="Where to store intermediate .pth files")
parser.add_argument("--initial-state-dict", default=argparse.SUPPRESS, help="Initial model")
parser.add_argument("--chunk-size", type=int, default=argparse.SUPPRESS)
parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
parser.add_argument("--learning-rate", type=float, default=argparse.SUPPRESS)
parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS)

parser.add_argument("--num-chunks", type=int, default=argparse.SUPPRESS, help="Number of chunks to generate")

args = parser.parse_args()

import json
import os
with open(args.input) as f:
    j = json.load(f)

if 'chunk_size' in args:
    j['train_loader']['batch_size'] = args.chunk_size

if 'batch_size' in args:
    j['batch_size'] = args.batch_size

if 'epochs' in args:
    j['epochs'] = args.epochs

if 'initial_state_dict' in args:
        j['model']['initial_state_dict'] = args.initial_state_dict

if 'learning_rate' in args:
        j['optimizer']['kwargs']['lr'] = args.learning_rate

if 'checkpoint_dir' in args:
    j['checkpoint_dir'] = args.checkpoint_dir

for i in range(args.num_chunks):
    chunk_id = str(i).zfill(3)

    if 'train_annotations' in args:
        j['train_dataset']['annotations_file'] = os.path.join(args.train_annotations, f"chunk-{chunk_id}.json")
    if 'test_annotations' in args:
        j['test_dataset']['annotations_file'] = os.path.join(args.test_annotations, f"chunk-{chunk_id}.json")

    j['final_state_dict_name'] = f"chunk-{chunk_id}.pth"

    # we are mixing train and test: verify we can actually train!
    j['trainable'] = os.path.exists(j['train_dataset']['annotations_file'])

    with open(os.path.join(args.output, f"chunk-{chunk_id}.json"), 'w') as f:
        json.dump(j, f, indent=4)

    j['initial_state_dict_name'] = j['final_state_dict_name']