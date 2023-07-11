import argparse
import os
import glob
import itertools

import numpy as np
import pandas as pd
from scipy.linalg import svd

import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

h36m_kp_names = [
    "LShoulder:U", "LShoulder:V", "RShoulder:U", "RShoulder:V",
    "LElbow:U", "LElbow:V", "RElbow:U", "RElbow:V", 
    "LWrist:U", "LWrist:V", "RWrist:U", "RWrist:V", 
    "LHip:U", "LHip:V", "RHip:U", "RHip:V", 
    "LKnee:U", "LKnee:V","RKnee:U", "RKnee:V", 
    "LAnkle:U", "LAnkle:V", "RAnkle:U", "RAnkle:V"
]
h36m_kps = list(dict.fromkeys(([kp.split(":")[0] for kp in h36m_kp_names])))

FORMAT = "jpg"

def first_derivative(df: pd.DataFrame, cols_from, cols_to):
    t_delta = (df["time"] - df["time"].shift(1, fill_value=0).reset_index(drop=True))
    t_delta.iloc[0] = 0

    df_1 = (df[cols_from] - df[cols_from].shift(1, fill_value=0).reset_index(drop=True)).div(t_delta, axis=0)
    df_1.iloc[0] = np.array([0] * len(df_1.columns))
    df_1 = df_1.rename(columns=dict(zip(df_1.columns, cols_to)))
    df_1 = pd.concat([df[["time"]], df_1], axis=1)
    return df_1


def calculate_mpjpe(skeleton1, skeleton2):
    """
    Calculate the Mean Per Joint Position Error (MPJPE) between two skeleton series.
    
    Args:
        skeleton1 (numpy.ndarray): The first skeleton series of shape (num_frames, num_joints, 3).
        skeleton2 (numpy.ndarray): The second skeleton series of shape (num_frames, num_joints, 3).
    
    Returns:
        float: The MPJPE value.
    """
    assert skeleton1.shape == skeleton2.shape, "Skeleton series must have the same shape."
    
    num_frames, num_joints, _ = skeleton1.shape
    
    # Calculate the Euclidean distance between corresponding joints in each frame
    errors = np.linalg.norm(skeleton1 - skeleton2, axis=2)

    # Exclude NaN values from the calculation
    valid_errors = np.where(np.isnan(errors), 0, errors)
    
    # Calculate the mean of errors across all frames and joints
    mpjpe = np.mean(valid_errors)
    
    return mpjpe, np.mean(valid_errors, axis=0)


def calculate_mAP(ground_truth_series, predicted_series, threshold=3):
    assert len(ground_truth_series) == len(predicted_series), "Number of ground truth series must be equal to number of predicted series"

    num_series = len(ground_truth_series)
    num_joints = ground_truth_series[0].shape[0]

    # Initialize variables for true positives, false positives, and false negatives
    true_positives = np.zeros((num_series, num_joints))
    false_positives = np.zeros((num_series, num_joints))
    false_negatives = np.zeros((num_series, num_joints))

    # Calculate precision and recall for each joint in each pose
    for series_idx in range(num_series):
        ground_truth_pose = ground_truth_series[series_idx]
        predicted_pose = predicted_series[series_idx]

        assert ground_truth_pose.shape == predicted_pose.shape, "Shape of ground truth poses must be equal to shape of predicted poses"
        for joint_idx in range(num_joints):
            ground_truth_joint = ground_truth_pose[joint_idx]
            predicted_joint = predicted_pose[joint_idx]

            # Calculate Euclidean distance between ground truth and predicted joint
            distance = np.linalg.norm(ground_truth_joint - predicted_joint)

            # Determine true positives, false positives, and false negatives
            if distance <= threshold:
                true_positives[series_idx, joint_idx] += 1
            else:
                false_positives[series_idx, joint_idx] += 1
                false_negatives[series_idx, joint_idx] += 1

    # Calculate precision and recall for each joint in each series
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    # Calculate average precision for each joint in each series
    average_precision = np.mean(precision, axis=0)

    # Calculate mean average precision (mAP) across all joints and series
    mAP = np.mean(average_precision)

    return mAP, average_precision



def calculate_svd(skeleton_series):
    # Convert the skeleton series into a numpy array
    skeleton_array = np.array(skeleton_series)
    
    # Replace NaN values with zeros
    skeleton_array = np.nan_to_num(skeleton_array)
    
    # Apply SVD to the skeleton array
    u, s, vt = svd(skeleton_array.T, full_matrices=False)

    return (u, s, vt)


def calculate_svd_reconstruction(skeleton_series, svd_info, num_components):
    u, s, vt = svd_info

    # Convert the skeleton series into a numpy array
    skeleton_array = np.array(skeleton_series)
    
    # Replace NaN values with zeros
    skeleton_array = np.nan_to_num(skeleton_array)
    
    # Truncate the singular values and corresponding matrices
    s_truncated = np.diag(s[:num_components])
    u_truncated = u[:, :num_components]
    vt_truncated = vt[:num_components, :]
    
    # Reconstruct the smoothed skeleton series
    smoothed_skeleton_array = u_truncated @ s_truncated @ vt_truncated
    
    # Transpose the array to get back the original shape
    smoothed_skeleton_series = smoothed_skeleton_array.T

    reconstruction_error = np.linalg.norm(skeleton_array - smoothed_skeleton_series)
    total_error = np.linalg.norm(skeleton_array)
    percentage_reconstruction = (1 - reconstruction_error / total_error) * 100
    
    return percentage_reconstruction



def find_pairs_h36m(s1_path, s2_path, camera):
    s1_files = glob.glob(os.path.join(s1_path, "*" + camera + "*"))
    ret = []
    for s1_f in s1_files: 
        s1_basename = os.path.basename(s1_f).replace(" ", "")
        s1_action = s1_basename.split(".")[0]
        if any(c.isdigit() for c in s1_action):
            actions = [s1_action, "{} {}".format(s1_action[:-1], s1_action[-1])]
        else: 
            actions = [s1_action]
        if "WalkDog" in actions:
            actions.append("WalkingDog")
        if "WalkDog1" in actions:
            actions.append("WalkingDog1")
        if "Photo" in actions:
            actions.append("TakingPhoto")
        if "Photo1" in actions:
            actions.append("TakingPhoto1")
        for a in itertools.product(actions, actions):
            s1_fp = os.path.join(s1_path, "{}.{}.csv".format(a[0], camera))
            s2_fp = os.path.join(s2_path, "{}.{}.csv".format(a[1], camera))
            if os.path.exists(s1_fp) and os.path.exists(s2_fp):
                ret.append((s1_fp, s2_fp))
                break
    return ret


def main_h36m(folder, s1, s2):
    train_subjects = ["S1", "S5", "S6", "S7", "S8"]
    test_subjects = ["S9", "S11"]
    subjects = {
        # "Train subjects": (train_subjects, "train"),
        "Test subjects": (test_subjects, "test")
    }
    cameras = ["55011271"]

    results = {}
    for t_sub in subjects:
        table_fp = "error_{}_{}_{}h36m.csv".format(s1, s2, subjects[t_sub][1])
        if os.path.exists(table_fp):
            print("Table between {} and {} on {} data already generated".format(s1, s2, subjects[t_sub][1]))
            results[t_sub] = pd.read_csv(table_fp, index_col=0)
            continue 

        print("Table between {} and {} on {} data generating...".format(s1, s2, subjects[t_sub][1]))
        print("{}".format(t_sub))
        results_t_sub = {
            "camera": [],
            "subject": [],
            "action": [],
            "MPJPE": [],
            "mAP": [],
        }
        for kp in h36m_kps: 
            results_t_sub["{} JPE".format(kp)] = []
            results_t_sub["{} AP".format(kp)] = []
        for camera in cameras: 
            for sub in subjects[t_sub][0]:
                s1_path = os.path.join(folder, sub, s1)
                s2_path = os.path.join(folder, sub, s2)
                sub_pairs = find_pairs_h36m(s1_path, s2_path, camera)

                for sub_pair in sub_pairs:
                    action = os.path.basename(sub_pair[0]).split(".")[0]
                    s1_df = pd.read_csv(sub_pair[0])[h36m_kp_names]
                    s2_df = pd.read_csv(sub_pair[1])[h36m_kp_names]
                    s1_df = s1_df.iloc[:min(len(s1_df.index), len(s2_df.index))]
                    s2_df = s2_df.iloc[:min(len(s1_df.index), len(s2_df.index))]
                    s1_reshape = s1_df.values.reshape((len(s1_df.index), -1, 2))
                    s2_reshape = s2_df.values.reshape((len(s2_df.index), -1, 2))

                    mpjpe, jpe = calculate_mpjpe(s1_reshape, s2_reshape)
                    map, ap = calculate_mAP(s1_reshape, s2_reshape)
                    print("[SUB {} CAM {} ACTION {:15}] MPJPE: {:.3f} mAP: {:.5f}".format(
                        sub, camera, action, mpjpe, map))
                    results_t_sub["camera"].append(camera)
                    results_t_sub["subject"].append(sub)
                    results_t_sub["action"].append(action)
                    results_t_sub["MPJPE"].append(mpjpe)
                    results_t_sub["mAP"].append(map)
                    jpe_report = ""
                    for i, e in enumerate(jpe): 
                        jpe_report += "{}: {:.1f}; ".format(h36m_kps[i], e)
                        results_t_sub["{} JPE".format(h36m_kps[i])].append(e)
                    print("JPE: {{ {}}}".format(jpe_report))
                    ap_report = ""
                    for i, e in enumerate(ap): 
                        ap_report += "{}: {:.4f}; ".format(h36m_kps[i], e)
                        results_t_sub["{} AP".format(h36m_kps[i])].append(e)
                    print("AP: {{ {}}}".format(ap_report))
        results[t_sub] = pd.DataFrame(results_t_sub)
        results[t_sub].to_csv(table_fp)

    print("==============================================")
    for t_sub in results: 
        print("RESULTS SUMMARY on {}".format(t_sub))
        error_for_list = [
            "camera",
            "subject", 
            "action"
        ]
        error_cols = ["MPJPE", "mAP"] + ["{} JPE".format(kp) for kp in h36m_kps] + ["{} AP".format(kp) for kp in h36m_kps]
        for error_for in error_for_list:
            print(" - Error for {}:".format(error_for))
            for index, row in results[t_sub].groupby(error_for)[error_cols].mean().iterrows():
                print("{:20}: MPJPE: {:.2f} mAP: {:.4f}".format(index, row["MPJPE"], row["mAP"]))
                jpe_report = ""
                for kp in h36m_kps: 
                    jpe_report += "{}: {:.1f}; ".format(kp, row["{} JPE".format(kp)])
                print("{:20}  JPE: {{ {}}}".format(" ", jpe_report))
                ap_report = ""
                for kp in h36m_kps: 
                    ap_report += "{}: {:.3f}; ".format(kp, row["{} AP".format(kp)])
                print("{:20}  AP: {{ {}}}".format(" ", ap_report))
        print("==============================================")


def plot(s1, s2_list):
    for t_sub in ["test"]:
        data_plot = {
            "model": [], 
            "MPJPE": [], 
            "mAP": [], 
        }
        for s2 in s2_list: 
            df = pd.read_csv("error_{}_{}_{}h36m.csv".format(s1, s2, t_sub), index_col=0)
            data_plot["model"].extend([s2] * len(df.index))
            data_plot["MPJPE"].extend(df["MPJPE"])
            data_plot["mAP"].extend(df["mAP"])
        data_plot = pd.DataFrame(data_plot)
        
        plot_name = "error_barplot_{}_{}.{}".format(s1, t_sub, FORMAT)
        print(" - {}".format(plot_name))
        fig, ax = plt.subplots(2, 1, figsize=(14, 20))
        sns.barplot(data=data_plot, x="model", y="MPJPE", ax=ax[0])
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=30, fontsize=7)
        ax[0].set_ylim(0, 12)
        sns.barplot(data=data_plot, x="model", y="mAP", ax=ax[1])
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=30, fontsize=7)
        fig.savefig(plot_name)
        plt.close()

        data_plot = {
            "model": [], 
            "keypoint": [],
            "JPE": [], 
            "AP": [], 
        }
        for s2 in s2_list: 
            df = pd.read_csv("error_{}_{}_{}h36m.csv".format(s1, s2, t_sub), index_col=0)
            for kp in h36m_kps: 
                data_plot["model"].extend([s2] * len(df.index))
                data_plot["keypoint"].extend([kp] * len(df.index))
                data_plot["JPE"].extend(df["{} JPE".format(kp)])
                data_plot["AP"].extend(df["{} AP".format(kp)])
        data_plot = pd.DataFrame(data_plot)

        plot_name = "error_boxplot_{}_{}_kp.{}".format(s1, t_sub, FORMAT)
        print(" - {}".format(plot_name))
        fig, ax = plt.subplots(2, 1, figsize=(25, 14))
        sns.boxplot(data=data_plot, x="keypoint", hue="model", y="JPE", ax=ax[0])
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=30, fontsize=7)
        sns.boxplot(data=data_plot, x="keypoint", hue="model", y="AP", ax=ax[1])
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=30, fontsize=7)
        fig.savefig(plot_name)
        plt.close()

        data_plot = {
            "model": [], 
            "action": [],
            "MPJPE": [], 
            "mAP": [], 
        }
        for s2 in s2_list: 
            df = pd.read_csv("error_{}_{}_{}h36m.csv".format(s1, s2, t_sub), index_col=0)
            data_plot["model"].extend([s2] * len(df.index))
            data_plot["action"].extend(df["action"])
            data_plot["MPJPE"].extend(df["MPJPE"])
            data_plot["mAP"].extend(df["mAP"])
        data_plot = pd.DataFrame(data_plot)

        plot_name = "error_boxplot_{}_{}_action.{}".format(s1, t_sub, FORMAT)
        print(" - {}".format(plot_name))
        fig, ax = plt.subplots(2, 1, figsize=(50, 20))
        sns.barplot(data=data_plot, x="action", hue="model", y="MPJPE", ax=ax[0])
        ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=30, fontsize=7)
        sns.barplot(data=data_plot, x="action", hue="model", y="mAP", ax=ax[1])
        ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=30, fontsize=7)
        fig.savefig(plot_name)
        plt.close()


def plot_svd_h36m(folder, s1, s2_list, num_eigen_vector=30, batch_size=None, data_type="pos"):
    model_list = [s1] + s2_list
    train_subjects = ["S1", "S5", "S6", "S7", "S8"]
    test_subjects = ["S9", "S11"]
    subjects = {
        # "Train subjects": (train_subjects, "train"),
        "Test subjects": (test_subjects, "test")
    }
    cameras = ["55011271"]
    if batch_size is None: 
        batch_size = num_eigen_vector

    threshold_perc_svd = 99.8

    plot_results = {}
    eigen_plot_results = {}
    for t_sub in subjects:
        table_fp = "error_svd_{}_{}_{}h36m.csv".format(s1, data_type, subjects[t_sub][1])
        eigen_table_fp = "error_eigensvd_{}_{}_{}h36m.csv".format(s1, data_type, subjects[t_sub][1])
        if os.path.exists(table_fp) and os.path.exists(eigen_table_fp):
            print("Table {} SVD on {} ref and {} data already generated".format(data_type, s1, subjects[t_sub][1]))
            plot_results[t_sub] = pd.read_csv(table_fp, index_col=0)
            eigen_plot_results[t_sub] = pd.read_csv(eigen_table_fp, index_col=0)
            continue 

        data_plot = {
            "eigen_vector": [],
            "model": [],
            "MPJPE": [], 
            "mAP": [],
            "perc": [],
        }

        eigen_data_plot = {
            "first_good_eigen_vector": [],
            "model": [],
            "MPJPE": [], 
            "mAP": [],
        }

        for model in model_list: 
            for camera in cameras: 
                for sub in subjects[t_sub][0]:
                    s1_path = os.path.join(folder, sub, s1)
                    model_path = os.path.join(folder, sub, model)
                    model_pairs = find_pairs_h36m(s1_path, model_path, camera)
                    model_pairs = [f_pair for f_pair in model_pairs if "ALL" not in f_pair[0]]
                    for model_f in model_pairs:
                        print(model_f[1], end="\n")
                        s1_df = pd.read_csv(model_f[0])[["time"] + h36m_kp_names]
                        model_df = pd.read_csv(model_f[1])[["time"] + h36m_kp_names]
                        s1_df = s1_df.iloc[:min(len(s1_df.index), len(model_df.index))]
                        model_df = model_df.iloc[:min(len(s1_df.index), len(model_df.index))]

                        if data_type == "vel":
                            s1_vel_df = first_derivative(s1_df, h36m_kp_names, h36m_kp_names)[h36m_kp_names]
                            model_vel_df = first_derivative(model_df, h36m_kp_names, h36m_kp_names)[h36m_kp_names]
                        else: 
                            s1_vel_df = s1_df[h36m_kp_names]
                            model_vel_df = model_df[h36m_kp_names]
                        
                        model_df = model_df[h36m_kp_names]
                        for i, data_window in enumerate(model_df.rolling(batch_size)): 
                            if len(data_window.index) < batch_size:
                                continue
                            
                            s1_vel_reshape = s1_vel_df.iloc[data_window.index].values.reshape((len(data_window.index), -1, 2))
                            model_vel_reshape = model_vel_df.iloc[data_window.index].values.reshape((len(data_window.index), -1, 2))
                            mpjpe, _ = calculate_mpjpe(s1_vel_reshape, model_vel_reshape)
                            map, _ = calculate_mAP(s1_vel_reshape, model_vel_reshape)

                            svd_info = calculate_svd(data_window.values)
                            first_good_eigen_vector = None
                            for e_i in range(1, num_eigen_vector + 1):
                                perc = calculate_svd_reconstruction(data_window.values, svd_info, e_i)
                                data_plot["eigen_vector"].append(e_i)
                                data_plot["model"].append(model)
                                data_plot["MPJPE"].append(mpjpe)
                                data_plot["mAP"].append(map)
                                data_plot["perc"].append(perc)
                                if first_good_eigen_vector is None and perc > threshold_perc_svd:
                                    first_good_eigen_vector = e_i
                                print(e_i, model, perc, end="\r")
                            if first_good_eigen_vector is None:
                                first_good_eigen_vector = num_eigen_vector
                            eigen_data_plot["first_good_eigen_vector"].append(first_good_eigen_vector)
                            eigen_data_plot["model"].append(model)
                            eigen_data_plot["MPJPE"].append(mpjpe)
                            eigen_data_plot["mAP"].append(map)
        data_plot = pd.DataFrame(data_plot)
        plot_results[t_sub] = data_plot
        plot_results[t_sub].to_csv(table_fp)
        eigen_data_plot = pd.DataFrame(eigen_data_plot)
        eigen_plot_results[t_sub] = eigen_data_plot
        eigen_plot_results[t_sub].to_csv(eigen_table_fp)

    for t_sub in plot_results:
        data_plot = plot_results[t_sub]
        eigen_data_plot = eigen_plot_results[t_sub]

        data_plot = data_plot.loc[
            (data_plot["eigen_vector"] >= 3) & (data_plot["eigen_vector"] <= 6) 
            & (data_plot["model"] != "vicon")
            # & ((data_plot["model"] == "trtpose_PARCO") | (data_plot["model"] == "trtpose_retrained"))
        ]
        eigen_data_plot = eigen_data_plot.loc[
            (eigen_data_plot["model"] != "vicon")
            # (eigen_data_plot["model"] == "trtpose_PARCO") | (eigen_data_plot["model"] == "trtpose_retrained")
        ]

        # Plot reconstruction
        plot_name = "error_lineplot_reconstruction_{}_{}h36m.{}".format(data_type, subjects[t_sub][1], FORMAT)
        print(" - {}".format(plot_name))
        sns.lineplot(data=data_plot, x="eigen_vector", y="perc", hue="model")
        plt.savefig(plot_name)
        plt.close()
        
        # Plot SVD scatter
        plot_name = "error_scatterplot_svd_MPJPE_{}_{}h36m.{}".format(data_type, subjects[t_sub][1], FORMAT)
        print(" - {}".format(plot_name))
        sns.relplot(
            data=data_plot, x="perc", y="MPJPE", col="eigen_vector", hue="model",
            kind="scatter", col_wrap=2
        )
        plt.savefig(plot_name)
        plt.close()

        plot_name = "error_scatterplot_svd_mAP_{}_{}h36m.{}".format(data_type, subjects[t_sub][1], FORMAT)
        print(" - {}".format(plot_name))
        sns.relplot(
            data=data_plot, x="perc", y="mAP", col="eigen_vector", hue="model",
            kind="scatter", col_wrap=2
        )
        plt.savefig(plot_name)
        plt.close()

        plot_name = "error_scatterplot_eigen_MPJPE_{}_{}h36m.{}".format(data_type, subjects[t_sub][1], FORMAT)
        print(" - {}".format(plot_name))
        sns.catplot(
            data=eigen_data_plot, x="first_good_eigen_vector", y="MPJPE", hue="model",
            kind="box", linewidth=0.4, height=10, aspect=1.2, showfliers = False
        )
        plt.xlabel(f"Minimum eigen-vector amount to obtain {threshold_perc_svd} %")
        plt.savefig(plot_name)
        plt.close()

        plot_name = "error_scatterplot_eigen_mAP_{}_{}h36m.{}".format(data_type, subjects[t_sub][1], FORMAT)
        print(" - {}".format(plot_name))
        sns.catplot(
            data=eigen_data_plot, x="first_good_eigen_vector", y="mAP", hue="model",
            kind="box", linewidth=0.4, height=10, aspect=1.2, showfliers = False
        )
        plt.xlabel(f"Minimum eigen-vector amount to obtain {threshold_perc_svd} %")
        plt.savefig(plot_name)
        plt.close()


def main(folder, s1, s2_list, data_type):
    if not os.path.exists(folder):
        print("Error: folder path {} doesn't exist")
        return
    
    if data_type not in ["h36m", "coco"]:
        print("Error: data_type {} is not recognized")
        return
    
    if data_type == "h36m":
        for s2 in s2_list:
            main_h36m(folder, s1, s2)
    
        print("Plot reconstruction")
        # plot_svd_h36m(folder, s1, s2_list, num_eigen_vector=15, batch_size=30, data_type="pos")
        # plot_svd_h36m(folder, s1, s2_list, num_eigen_vector=15, batch_size=30, data_type="vel")

    print("Plot general")
    plot(s1, s2_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Error between sources of data", epilog="PARCO")
    parser.add_argument("--folder", 
                        "-f", 
                        dest="folder", 
                        required=True, 
                        help="Folder with first sources")
    parser.add_argument("--reference", 
                        "-r", 
                        dest="ref", 
                        required=True, 
                        help="Reference data")
    parser.add_argument("--sources", 
                        "-s", 
                        dest="sources", 
                        nargs="+", 
                        required=True, 
                        help="Sources")
    parser.add_argument("--type", 
                        "-t", 
                        dest="type", 
                        required=False,
                        default="h36m", 
                        help="Dataset type name")
    args = parser.parse_args()
    main(args.folder, args.ref, args.sources, args.type)