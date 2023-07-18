import json
import glob
import pandas as pd
import argparse
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from parcopose_from_folder import H36M_2D_COLS

parser = argparse.ArgumentParser("Generate order description for H36M")

parser.add_argument("--subject", default="S1")
parser.add_argument("--base-path", default="dataset")
parser.add_argument("--output", default="lexicographic_order.csv")
parser.add_argument("--output-gt", default="vicon_new.csv")
parser.add_argument("--percentage", default=0.8, type=float, help="Percentage of action to take")
parser.add_argument("--actions", nargs="*" , default=None)
parser.add_argument("--teacher", default="vicon")
parser.add_argument("--output-adeguate-folder", default="adeguate", help="Folder to output GT adeguated")
parser.add_argument("--adeguate", nargs="*", default=[], help="Methods to use the same indexing")

args = parser.parse_args()

out = {}
annotations = []
images = []
camera = '55011271'

file_order = []

actions = args.actions

global_count = 0

print(f"Generating order set")
for sub in [ args.subject ]:#, 'S11']: #'S1','S5', 'S6', 'S7', 'S8']:#
    base_path = os.path.join(args.base_path,  sub, args.teacher)
    other_base_path = [os.path.join(args.base_path,  sub, a) for a in args.adeguate]

    ## If you want to change the order action

    def process_action(data, other_data, action=None):
        if action is None:
            files = glob.glob(os.path.join(base_path, '*' + camera + '*'))
        else:
            globData = '*' + action + "." + camera + '*'
            files = glob.glob(os.path.join(base_path, globData))

        df_global_id = []
        df_frames = []
        df_sub = []
        df_cam = []
        df_action = []
        df_id = []
        df_time = []

        new_data = pd.DataFrame()
        new_other_data = [pd.DataFrame()] * len(args.adeguate)

        for filename in files:
            filename_original = filename.split('/')[-1]

            filename_replaced = filename.split('/')[-1].replace(" ","_")

            try:
                filename_replaced = filename_replaced.replace('_','.')
                take = '.'.join(filename_replaced.split('/')[-1].split('_')[0].split('.')[0:-1])
            except:
                take = '.'.join(filename_replaced.split('/')[-1].split('_')[0].split('.')[0:-1])

            cam = ' '.join(take.split('.')[:-1])
            # if action is not None and cam != action:
            #     continue
            print(f"Generating: {cam}")

            file = pd.read_csv(filename)
            other_files = [pd.read_csv(os.path.join(p, filename_original)) 
                           if os.path.exists(os.path.join(p, filename_original))
                           else pd.read_csv(os.path.join(p, filename_original.replace(" ", "")))
                           for p in other_base_path]

            # limit pandas
            max_n = int(file.shape[0]*args.percentage)
            data_to_iter = file.iloc[:max_n] 
            new_data = pd.concat([data_to_iter, new_data], axis=0).reset_index(drop=True)
            for i in range(len(new_other_data)):
                new_other_data[i] = pd.concat([other_files[i].iloc[:max_n], new_other_data[i]], axis=0).reset_index(drop=True)

            for i, r in data_to_iter.iterrows():
                cam = take.split('.')[-1]
                action = ''.join(take.split('.')[:-1])
                dirname =  f'{cam}/{action}.{cam}'
                image_id = "{}/{}/{}.png".format(sub, dirname, i)

                current_id = dirname
                # sub = current_id.split("/")[0]
                frame_id = i#int(current_id.split("/")[3].split(".")[0])

                df_global_id.append(len(file_order))
                df_frames.append(frame_id)
                df_sub.append(sub)
                df_cam.append(cam)
                df_action.append(action)
                df_id.append(current_id)
                df_time.append(len(file_order)/50)

                file_order.append(image_id)
        
        new_data["global_id"] = df_global_id
        new_data["frames"] = df_frames
        new_data["sub"] = df_sub
        new_data["cam"] = df_cam
        new_data["action"] = df_action
        new_data["id"] = df_id
        new_data["time"] = df_time
        data = pd.concat([data, new_data], axis=0).reset_index(drop=True)

        for i in range(len(new_other_data)):
            new_other_data[i]["global_id"] = df_global_id
            new_other_data[i]["frames"] = df_frames
            new_other_data[i]["sub"] = df_sub
            new_other_data[i]["cam"] = df_cam
            new_other_data[i]["action"] = df_action
            new_other_data[i]["id"] = df_id
            new_other_data[i]["time"] = df_time

            other_data[i] = pd.concat([other_data[i], new_other_data[i]], axis=0).reset_index(drop=True)
        return data, other_data
    
    keep_all_actions = actions is None
    if keep_all_actions:
        data, other_data = process_action(pd.DataFrame())
    else:
        data = pd.DataFrame()
        other_data = [pd.DataFrame()] * len(args.adeguate)
        for action in actions:
            data, other_data = process_action(data, other_data, action)
    data.to_csv(args.output_gt)

    for i, a in enumerate(args.adeguate):
        other_data[i].to_csv(os.path.join(args.output_adeguate_folder, a + ".csv"))


with open(args.output, 'w') as f:
    i=0
    for ff in file_order:
        f.write(str(i) + ',' + ff + '\n')
        i = i+1