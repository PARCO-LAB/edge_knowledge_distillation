import json
import glob
import pandas as pd
import argparse

parser = argparse.ArgumentParser("Generate order description for H36M")

parser.add_argument("--subject", default="S1")
parser.add_argument("--base-path", default="/home/shared/nas/KnowledgeDistillation/h36m/")
parser.add_argument("--output", default="lexicographic_order.csv")
parser.add_argument("--actions", nargs="*" , default=None)

args = parser.parse_args()

out = {}
annotations = []
images = []
camera = '55011271'

file_order = []

actions = args.actions

print(f"Generating order set")
for sub in [ args.subject ]:#, 'S11']: #'S1','S5', 'S6', 'S7', 'S8']:#
    base_path = args.base_path +  sub + '/vicon/'

    files = glob.glob(base_path + '*' + camera + '*')

    ## If you want to change the order action

    for filename in files:
        filename_replaced = filename.split('/')[-1].replace(" ","_")

        try:
            filename_replaced = filename_replaced.replace('_','.')
            take = '.'.join(filename_replaced.split('/')[-1].split('_')[0].split('.')[0:-1])
            # print(take)
        except:
            take = '.'.join(filename_replaced.split('/')[-1].split('_')[0].split('.')[0:-1])
            # print(take)
        cam = ' '.join(take.split('.')[:-1])
        if actions is not None and cam not in actions:
            continue
        print(f"Generating: {cam}")

        file = pd.read_csv(filename)
        
        for i, r in file.iterrows(): 
            cam = take.split('.')[-1]
            action = ''.join(take.split('.')[:-1])
            dirname =  f'{cam}/{action}.{cam}'
            image_id = "{}/{}/{}.png".format(sub, dirname, i)
            file_order.append(image_id)


with open(args.output, 'w') as f:
    i=0
    for ff in file_order:
        f.write(str(i) + ',' + ff + '\n')
        i = i+1