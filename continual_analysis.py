import argparse
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np


def main(checkpoints_fp, baselines):
    palette = sns.color_palette("Set1", len(baselines) + 2)
    
    i = 0
    continual_df = pd.DataFrame()
    while True:

        fp = os.path.join(checkpoints_fp, "resval_dist_{}.csv".format(i))
        if not os.path.exists(fp):
            break

        df = pd.read_csv(fp, index_col=0)
        df["chunk"] = i 
        continual_df = pd.concat([continual_df, df], axis=0).reset_index(drop=True)
        i += 1

    actions = set(continual_df["action"].unique())

    print("Computed checkpoints: {}".format(i))

    i = 0
    base_continual_df = pd.DataFrame()
    while True:

        fp = os.path.join(checkpoints_fp, "resval_base_dist_{}.csv".format(i))
        if not os.path.exists(fp):
            break

        df = pd.read_csv(fp, index_col=0)
        df["chunk"] = i
        base_continual_df = pd.concat([base_continual_df, df], axis=0).reset_index(drop=True)
        i += 1

    print("Plot MPJPE ... ")
    continual_df = pd.concat([base_continual_df.iloc[[0]], continual_df], axis=0).reset_index(drop=True)
    base_continual_df = pd.concat([base_continual_df.iloc[[0]], base_continual_df], axis=0).reset_index(drop=True)
    continual_df.loc[0, "chunk"] = -1
    base_continual_df.loc[0, "chunk"] = -1

    
    ax = sns.relplot(data=continual_df, x="chunk", y="MPJPE", kind="line", 
                     label="Continual learning", 
                     color=palette[len(baselines)+1], 
                     height=10, aspect=2, errorbar=('sd', 0.2), drawstyle="steps-post")
    sns.lineplot(data=base_continual_df, x="chunk", y="MPJPE", ax=ax.ax, color=palette[len(baselines)], label="ParcoPose", err_style="band", errorbar=('sd', 0.2), drawstyle="steps-post")
    curr_action = None
    for i, point in base_continual_df.iterrows(): 
        if curr_action is None or curr_action != str(point['action']):
            ax.ax.text(point['chunk'], point['MPJPE']+.1, str(point['action']), rotation=20, size=15)
        curr_action = str(point['action'])


    for b_i, b in enumerate(baselines):
        df = pd.read_csv(b, index_col=0)
        df_baseline = pd.DataFrame()
        b_name = " ".join(b.split("_")[2:-1])
        for c_i in range(i): 
            df["chunk"] = c_i
            df_baseline = pd.concat([df_baseline, df]).reset_index(drop=True)
        # sns.lineplot(data=df_baseline, x="chunk", y="MPJPE", ax=ax.ax, color=palette[b_i], label=b_name, err_style="band", errorbar=('sd', 0.2))

        actions = actions.intersection(set([action.replace(" ", "") for action in df["action"].unique()]))

    plt.savefig("continual_analysis_mpjpe.png")

    print("Plot MPJPE for each action ... ")
    ax = sns.relplot(data=continual_df, x="chunk", y="MPJPE", kind="line", 
                     label="Continual learning", 
                     color=palette[len(baselines)], col="action", col_wrap=5)
    for a_i, action in enumerate(actions):
        for b_i, b in enumerate(baselines):
            df = pd.read_csv(b, index_col=0)
            df["action"] = [a.replace(" ", "") for a in df["action"]]
            df = df.loc[df["action"] == action]

            df_baseline = pd.DataFrame()
            b_name = " ".join(b.split("_")[2:-1])
            for c_i in range(i): 
                df["chunk"] = c_i
                df_baseline = pd.concat([df_baseline, df]).reset_index(drop=True)
            sns.lineplot(data=df_baseline, x="chunk", y="MPJPE", ax=ax.axes[a_i], color=palette[b_i], label=b_name)
    plt.savefig("continual_analysis_mpjpe_action.png")

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Wearable and Camera magnitude plot", 
        epilog="Mirco De Marchi & Cristian Turetta")
    parser.add_argument("--checkpoints",
                        "-c",
                        dest="checkpoints",
                        required=True,
                        help="Folder path with checkpoints")
    parser.add_argument("--baselines",
                        "-b",
                        dest="baselines",
                        required=True,
                        nargs="+",
                        help="Folder path with baselines")
    args = parser.parse_args()
    main(args.checkpoints, args.baselines)