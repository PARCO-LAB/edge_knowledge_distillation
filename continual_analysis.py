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
    train_annotations_fp = config["train_dataset"]["annotations_file"]
    with open(train_annotations_fp, 'r') as f:
        train_annotations_json = json.load(f)

    i = 0
    df_checkpoint = pd.DataFrame()
    prev_action_list = None
    prev_action = None
    while True:
        fp = os.path.join("{}.checkpoints".format(checkpoint_fp), "{}_{}.csv".format(prefix, i))
        if not os.path.exists(fp):
            break
        
        last_idx_of_chunk = min((i + 1) * chunk_size, len(train_annotations_json["images"])-1)
        last_id_of_chunk = train_annotations_json["images"][last_idx_of_chunk]["id"]

        df = pd.read_csv(fp, index_col=0)
        action = last_id_of_chunk.split("/")[-2].split(".")[0]
        frame = int(last_id_of_chunk.split("/")[-1])
        df["chunk"] = i 
        curr_actions = df["action"].copy().tolist()
        # print(action, curr_actions)
        df["frame"] = [frame if i == 0 else 0 for i in range(len(curr_actions))] 
        df["train"] = [train if i == 0 else False for i in range(len(curr_actions))] 
        df["action"] = [action if i == 0 else a_df for i, a_df in enumerate(curr_actions)] 
        df["time"] = ["{} [{}]".format(action, frame) if i == 0 else "{} [{}]".format(a_df, 0) 
                      for i, a_df in enumerate(curr_actions)] 
        if prev_action is None or prev_action != action: 
            df_new_action = df.iloc[[0]].copy()
            df_new_action["train"] = False
            df_new_action["time"] = "{} [{}]".format(action, 0)
            df_new_action["frame"] = 0
            df_checkpoint = pd.concat([df_checkpoint, df_new_action], axis=0).reset_index(drop=True)

        if prev_action_list is None or prev_action_list != curr_actions:
            df_checkpoint = pd.concat([df_checkpoint, df], axis=0).reset_index(drop=True)
        else: 
            df_checkpoint = pd.concat([df_checkpoint, df.iloc[[0]]], axis=0).reset_index(drop=True)
        prev_action = action
        prev_action_list = curr_actions
        i += 1
    return df_checkpoint


def compute_baseline(baseline_fp, x_samples):
    df = pd.read_csv(baseline_fp, index_col=0)
    df["chunk"] = np.nan
    df["frame"] = np.nan
    df["action"] = np.nan
    df_baseline = pd.DataFrame()
    for i in range(len(x_samples)):
        df["time"] = x_samples[i]
        df_baseline = pd.concat([df_baseline, df], axis=0)
    return df_baseline


def main(checkpoints, baselines):
    data = {}

    # Checkpoints preprocessing
    scatter = {}
    for c_i, checkpoint_fp in enumerate(checkpoints):
        checkpoint_name = "C.L. ({})".format("_".join(os.path.basename(checkpoint_fp).split("_")[1:-6]))
        data[checkpoint_name] = compute_checkpoint(checkpoint_fp, "resval_dist")
        scatter[checkpoint_name] = {"color": "orange", "marker": "D", "s": 30} if c_i == 0 else {"color": "red", "marker": "D", "s": 50}

    # Baseline preprocessing
    core_baseline = None
    for baseline_fp in baselines:
        if os.path.splitext(baseline_fp)[1] == ".json":
            checkpoint_name = "Baseline ({})".format("_".join(os.path.basename(baseline_fp).split("_")[1:-6]))
            data[checkpoint_name] = compute_checkpoint(baseline_fp, "resval_base_dist", train=False)
            if core_baseline is None:
                core_baseline = data[checkpoint_name]
            data[checkpoint_name] = pd.concat([core_baseline.iloc[[0]], data[checkpoint_name]], axis=0).reset_index(drop=True)
            data[checkpoint_name].loc[0, "chunk"] = -1
            scatter[checkpoint_name] = {"color": "orange", "marker": "D", "s": 30} 

    if core_baseline is not None:
        for name in data: 
            if "C.L." in name:
                # Add baseline actions in front of continual learning
                first_continual_action = data[name]["action"].iloc[0]
                for a in core_baseline["action"].drop_duplicates().tolist():
                    if a == first_continual_action:
                        break
                    data[name] = pd.concat([core_baseline.loc[core_baseline["action"] == a], data[name]], axis=0).reset_index(drop=True)
                
                data[name] = pd.concat([core_baseline.iloc[[0]], data[name]], axis=0).reset_index(drop=True)
                data[name].loc[0, "chunk"] = -1
                data[name].loc[0, "frame"] = -1
                data[name].loc[0, "time"] = "{} [pre]".format(data[name]["action"].iloc[0])
                

    # Prepare plot data
    df_data = pd.DataFrame()
    for name in data: 
        data[name]["model"] = name
        df_data = pd.concat([df_data, data[name]], axis=0).reset_index(drop=True)
    
    for baseline_fp in baselines:
        if os.path.splitext(baseline_fp)[1] == ".csv":
            baseline_name = "Baseline ({})".format("_".join(baseline_fp.split("_")[1:-1]))
            data[baseline_name] = compute_baseline(baseline_fp, df_data["time"].unique())
            data[baseline_name]["model"] = baseline_name
            df_data = pd.concat([df_data, data[baseline_name]], axis=0).reset_index(drop=True)

    print("Plot MPJPE ... ")
    # df_data = df_data.sort_values(by=["action", "frame"])
    idxs = []
    for a in core_baseline["action"].drop_duplicates().tolist():
        idxs.extend(df_data.loc[df_data["action"] == a].sort_values(by=["frame"]).index.tolist())
    df_data = df_data.iloc[idxs]
    
    fig, ax = plt.subplots(1, 1, figsize=(40, 25))
    sns.lineplot(
        data=df_data, x="time", y="MPJPE", hue="model", ax=ax,
        err_style="band", errorbar=("sd", 0.2), drawstyle="steps", zorder=1)
    sns.scatterplot(
        data=df_data.loc[df_data["frame"] == 0], x="time", y="MPJPE", color="black", ax=ax, zorder=2, label="Changing action")
    for name in scatter:
        sns.scatterplot(
            data=df_data.loc[(df_data["model"] == name) & (df_data["train"] == True)], 
            x="time", y="MPJPE", ax=ax, **scatter[name], zorder=3, label="Train step ({})".format(name))
        
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    ax.set_ylim([0, 35])
    ax.tick_params(axis="x", rotation=90)
    if core_baseline is not None:
        for i, point in core_baseline.iterrows(): 
            if point["frame"] == 0:
                ax.text(point['time'], point['MPJPE']+4, str(point['action']), 
                        rotation=20, size=15)

    fig.savefig("continual_analysis_mpjpe.png")

    # TODO: for action
    # print("Plot MPJPE for each action ... ")
    # ax = sns.relplot(data=continual_df, x="chunk", y="MPJPE", kind="line", 
    #                  label="Continual learning", 
    #                  color=palette[len(baselines)], col="action", col_wrap=5)
    # for a_i, action in enumerate(actions):
    #     for b_i, b in enumerate(baselines):
    #         df = pd.read_csv(b, index_col=0)
    #         df["action"] = [a.replace(" ", "") for a in df["action"]]
    #         df = df.loc[df["action"] == action]
    #         df_baseline = pd.DataFrame()
    #         b_name = " ".join(b.split("_")[2:-1])
    #         for c_i in range(i): 
    #             df["chunk"] = c_i
    #             df_baseline = pd.concat([df_baseline, df]).reset_index(drop=True)
    #         sns.lineplot(data=df_baseline, x="chunk", y="MPJPE", ax=ax.axes[a_i], color=palette[b_i], label=b_name)
    # plt.savefig("continual_analysis_mpjpe_action.png")

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Wearable and Camera magnitude plot", 
        epilog="Mirco De Marchi & Cristian Turetta")
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
    args = parser.parse_args()
    main(args.checkpoints, args.baselines)
