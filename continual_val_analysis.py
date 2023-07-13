import argparse
import os
import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np


def compute_checkpoint(checkpoint_fp, prefix, train=True):
    with open(checkpoint_fp, "r") as f:
        config = json.load(f)
    chunk_size = config["train_loader"]["batch_size"]
    # train_annotations_fp = config["train_dataset"]["annotations_file"]
    # with open(train_annotations_fp, 'r') as f:
    #     train_annotations_json = json.load(f)

    fp = os.path.join("{}.checkpoints".format(checkpoint_fp), "{}.csv".format(prefix))
    if not os.path.exists(fp):
        print("Error: checkpoint {} doesn't exist".format(fp))
        exit()

    df_checkpoint = pd.read_csv(fp, index_col=0)
    train_idx = np.arange(0, len(df_checkpoint.index), chunk_size)
    df_checkpoint["train"] = False
    df_checkpoint.loc[train_idx, "train"] = train
    df_checkpoint.loc[0, "train"] = False
    
    return df_checkpoint


def compute_baseline(baseline_fp, x_samples):
    df = pd.read_csv(baseline_fp, index_col=0)
    df["action"] = None
    df_baseline = pd.DataFrame()
    for i in [0, len(x_samples) - 1]:
        df["time"] = x_samples[i]
        df_baseline = pd.concat([df_baseline, df], axis=0)
    return df_baseline


def time_coherency(data):
    action_time_accumulator = 0
    for a in enumerate(data["action"].unique()): 
        data.loc[data["action"] == a, "time"] = action_time_accumulator + data.loc[data["action"] == a, "frames"]
        action_time_accumulator += data.loc[data["action"] == a, "frames"].max()
    return data


def main(checkpoints, baselines, test_name=None):
    data = {}

    # Checkpoints preprocessing
    scatter = {}
    for c_i, checkpoint_fp in enumerate(checkpoints):
        checkpoint_name = "Continual Learning ({})".format("_".join(os.path.basename(checkpoint_fp).split("_")[1:-6]))
        print("Compute {}".format(checkpoint_name))
        data[checkpoint_name] = compute_checkpoint(checkpoint_fp, "resval_dist")
        scatter[checkpoint_name] = {"color": "orange", "marker": "D", "s": 50} if c_i == 0 else {"color": "red", "marker": "D", "s": 60}

    # Baseline preprocessing
    core_baseline = None
    for baseline_fp in baselines:
        if os.path.splitext(baseline_fp)[1] == ".json":
            checkpoint_name = "Baseline ({})".format("_".join(os.path.basename(baseline_fp).split("_")[1:-6]))
            print("Compute {}".format(checkpoint_name))
            data[checkpoint_name] = compute_checkpoint(baseline_fp, "resval_base_dist", train=False)
            if core_baseline is None:
                core_baseline = data[checkpoint_name]
            scatter[checkpoint_name] = {"color": "orange", "marker": "D", "s": 50} 

    # Prepare plot data
    common_max_time = min([data[name]["time"].max() for name in data])
    df_data = pd.DataFrame()
    for name in data: 
        data[name]["model"] = name
        data[name] = data[name].loc[data[name]["time"] <= common_max_time]
        df_data = pd.concat([df_data, data[name]], axis=0).reset_index(drop=True)

    df_data = time_coherency(df_data)
    
    for baseline_fp in baselines:
        if os.path.splitext(baseline_fp)[1] == ".csv":
            baseline_name = "Baseline ({})".format("_".join(baseline_fp.split("_")[1:-1]))
            print("Compute {}".format(baseline_name))
            data[baseline_name] = compute_baseline(baseline_fp, df_data["time"].unique())
            data[baseline_name]["model"] = baseline_name
            df_data = pd.concat([df_data, data[baseline_name]], axis=0).reset_index(drop=True)

    print("Plot MPJPE ... ")
    df_data = df_data.sort_values(by=["time"])
    fig, ax = plt.subplots(1, 1, figsize=(40, 25))
    sns.lineplot(
        data=df_data, x="time", y="MPJPE", hue="model", ax=ax,
        err_style="band", errorbar=("sd", 0.2), drawstyle="steps", zorder=1)
    for name in scatter:
        sns.scatterplot(
            data=df_data.loc[(df_data["model"] == name) & (df_data["train"] == True)], 
            x="time", y="MPJPE", ax=ax, **scatter[name], zorder=3, label="Train step ({})".format(name))
        
    ax.set_ylim([0, 35])
    ax.tick_params(axis="x", rotation=90)
    prev_action = None
    for i, point in df_data.iterrows(): 
        if (prev_action is None or prev_action != point['action']) and point['action'] is not None:
            ax.axvline(x=point['time'], color='black', label='Action change' if prev_action is None else None)
            ax.text(point['time'], point['MPJPE']+4, str(point['action']), 
                    rotation=20, size=15, zorder=5)
            prev_action = point['action']

    ax.legend()
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    fig.savefig("continual_val_analysis_mpjpe{}.png".format("" if test_name is None else "_{}".format(test_name)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Continual analysis", 
        epilog="Mirco De Marchi")
    parser.add_argument("--checkpoints",
                        "-c",
                        dest="checkpoints",
                        required=True,
                        nargs="+",
                        help="File paths with checkpoints")
    parser.add_argument("--baselines",
                        "-b",
                        dest="baselines",
                        required=True,
                        nargs="+",
                        help="File paths with baselines")
    parser.add_argument("--name",
                        "-n",
                        dest="name",
                        required=False,
                        default=None, 
                        help="Test name")
    args = parser.parse_args()
    main(args.checkpoints, args.baselines, args.name)
