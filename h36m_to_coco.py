import argparse
import json
import os
import pandas as pd
import numpy as np
import glob
import time
import random
import math

SEED=213445
random.seed(SEED)

from error import calculate_svd, calculate_svd_reconstruction, calculate_mpjpe, h36m_kp_names

TEACHERS = ["vicon", "openpose1", "CPN"]
SAMPLINGS = ["fixedmean_confidence", "fixedconfidence", "fixedmpjpe", "mean_confidence", "confidence", "mpjpe", "uniform", "fixedrandom", "random", "action", "parco"]
PERCENTAGES = [0.01, 0.05, 0.1, 0.2, 0.4]
WINDOW = int(50 * 30)

IMAGE_ID_LIMIT = 700
CONTINUAL_TRAIN_PERC = 0.8
W, H = 1000, 1000

cameras = ["55011271"]

trt_kp_names = [
    "left_shoulder", "right_shoulder", 
    "left_elbow", "right_elbow", 
    "left_wrist", "right_wrist", 
    "left_hip", "right_hip", 
    "left_knee", "right_knee", 
    "left_ankle", "right_ankle"
]
vicon_kp_names = [
    "LShoulder:U", "LShoulder:V", "RShoulder:U", "RShoulder:V",
    "LElbow:U", "LElbow:V", "RElbow:U", "RElbow:V", 
    "LWrist:U", "LWrist:V", "RWrist:U", "RWrist:V", 
    "LHip:U", "LHip:V", "RHip:U", "RHip:V", 
    "LKnee:U", "LKnee:V","RKnee:U", "RKnee:V", 
    "LAnkle:U", "LAnkle:V", "RAnkle:U", "RAnkle:V"
]
h36m_kps = list(dict.fromkeys(([kp.split(":")[0] for kp in vicon_kp_names])))


def get_annotation(id, keypoints, image_id):
    
    keypoints_out = keypoints[vicon_kp_names].tolist()
    kp_final = [0] * (5*3)
    for kp in range(0,len(keypoints_out),2):
        kp_final.append(keypoints_out[kp])
        kp_final.append(keypoints_out[kp+1])
        kp_final.append(2)
        
    ret = {
            "segmentation": [],
            "num_keypoints": len(trt_kp_names),
            "area": 0,
            "iscrowd": 0,
            "keypoints": kp_final,
            "image_id": image_id,
            "bbox": [0,0,W,H],
            "category_id": 1,
            "id": id
        }
    return ret

def get_image(id, height, width, date="", url="", format="png"):
    ret = {
        "license": 1,
        "file_name": "{}.{}".format(id, format),
        "coco_url": str(url),
        "height": int(height),
        "width": int(width),
        "date_captured": str(date),
        "flickr_url": str(url),
        "id": id
    }
    return ret


def get_dataset_amount(subjects, h36m_folder): 
    amount_dict = {}
    for camera in cameras:
        amount_dict[camera] = {}
        for sub in subjects:
            base_path = os.path.join(h36m_folder, sub, camera)
            files = glob.glob(os.path.join(base_path, "*" + camera + "*"))
            files = [f for f in files if "ALL" not in f]
            amount = 0
            for f in files: 
                print(f, end="\r")
                if IMAGE_ID_LIMIT is not None:
                    if os.path.exists(os.path.join(f, "{}.png".format(int(IMAGE_ID_LIMIT - 1)))):
                        amount += IMAGE_ID_LIMIT
                        continue

                    for i in range(IMAGE_ID_LIMIT - 1, -1, -1):
                        if os.path.exists(os.path.join(f, "{}.png".format(int(i)))):
                            amount += i
                        else: 
                            break
                else: 
                    i = 0
                    while True: 
                        if not os.path.exists(os.path.join(f, "{}.png".format(int(i)))):
                            break
                        i += 1
                    amount += i
            amount_dict[camera][sub] = amount
    return amount_dict


def random_sampling(files, num_samples):
    ret = {}
    all_samples = []
    for fp in files: 
        ret[fp] = []
        limit = IMAGE_ID_LIMIT if IMAGE_ID_LIMIT is not None else len(pd.read_csv(fp).index)
        for i in range(limit):
            all_samples.append({
                "idx": i,
                "fp": fp
            })

    random_sampling = list(random.sample(all_samples, num_samples))

    for e in random_sampling: 
        ret[e["fp"]].append(e["idx"])
    for fp in ret: 
        ret[fp] = sorted(ret[fp])
        print("{}: key-frame amount {}{:20}".format(fp, len(ret[fp]), " "))
    return ret


def fixedrandom_sampling(files, w_perc):
    ret = {}
    
    all_samples = []
    for fp in files: 
        ret[fp] = []
        limit = IMAGE_ID_LIMIT if IMAGE_ID_LIMIT is not None else len(pd.read_csv(fp).index)
        for i in range(limit):
            all_samples.append({
                "idx": i,
                "fp": fp
            })
    for w_i in range(math.ceil(len(all_samples) / WINDOW)):
        start_w = w_i*WINDOW
        end_w = min((w_i+1)*WINDOW, len(all_samples))
        window_size = end_w - start_w
        num_samples_for_window = int(w_perc * window_size)
        curr_window = all_samples[start_w:end_w]
        curr_window_sampling = list(random.sample(curr_window, num_samples_for_window))
        print("{}: fixed key-frame amount {} (window_size: {}) {:20}".format(w_i, len(curr_window_sampling), end_w-start_w, " "))
        for e in curr_window_sampling: 
            ret[e["fp"]].append(e["idx"])

    for fp in files: 
        ret[fp] = sorted(ret[fp])
        print("{}: key-frame amount {}{:20}".format(fp, len(ret[fp]), " "))

    return ret


def uniform_sampling(files, num_samples):
    ret = {}
    all_samples = []
    for fp in files: 
        ret[fp] = []
        limit = IMAGE_ID_LIMIT if IMAGE_ID_LIMIT is not None else len(pd.read_csv(fp).index)
        for i in range(limit):
            all_samples.append({
                "idx": i,
                "fp": fp
            })

    uniform_sampling = list(range(0, len(all_samples), len(all_samples) // num_samples))[:num_samples]

    for i in uniform_sampling: 
        e = all_samples[i]
        ret[e["fp"]].append(e["idx"])
    for fp in ret: 
        ret[fp] = sorted(ret[fp])
        print("{}: key-frame amount {}{:20}".format(fp, len(ret[fp]), " "))
    return ret


def action_sampling(files, num_samples):
    ret = {}

    def check_action(f):
        for a in ["Directions", "Discussion", "Phoning", "Smoking", "Photo", "Waiting"]:
            if a in f: 
                return True
        return False
    
    for fp in files: 
        ret[fp] = []
    files = [f for f in files if check_action(f)]
    all_samples = []
    for fp in files: 
        limit = IMAGE_ID_LIMIT if IMAGE_ID_LIMIT is not None else len(pd.read_csv(fp).index)
        for i in range(limit):
            all_samples.append({
                "idx": i,
                "fp": fp
            })

    uniform_sampling = list(range(0, len(all_samples), len(all_samples) // num_samples))[:num_samples]

    for i in uniform_sampling: 
        e = all_samples[i]
        ret[e["fp"]].append(e["idx"])
    for fp in ret: 
        ret[fp] = sorted(ret[fp])
        print("{}: key-frame amount {}{:20}".format(fp, len(ret[fp]), " "))
    return ret


def parco_keyframe_sampling(files, num_samples, num_eigen_vector=15, batch_size=30):
    ret = {}
    if batch_size is None: 
        batch_size = num_eigen_vector
    first_good_eigen_vector_for_data_index = []
    for fp in files: 
        print(fp, end="\r")
        df = pd.read_csv(fp)

        THRESHOLD_PREC_SVD = 99.8
        for data_window in df.rolling(batch_size): 
            if len(data_window) != batch_size:
                continue
            if IMAGE_ID_LIMIT is not None and int(data_window.iloc[0]["frame"]) >= IMAGE_ID_LIMIT: 
                break
            svd_info = calculate_svd(data_window.values)
            first_good_eigen_vector = None
            for e_i in range(1, num_eigen_vector + 1):
                perc = calculate_svd_reconstruction(data_window.values, svd_info, e_i)
                if first_good_eigen_vector is None and perc > THRESHOLD_PREC_SVD:
                    first_good_eigen_vector = e_i
            if first_good_eigen_vector is None:
                first_good_eigen_vector = num_eigen_vector
            first_good_eigen_vector_for_data_index.append({
                "fp": fp,
                "data_idx": int(data_window.iloc[0]["frame"]),
                "eigen_vector_idx": first_good_eigen_vector
            })
    first_good_eigen_vector_for_data_index = sorted(
        first_good_eigen_vector_for_data_index, key=lambda x: x["eigen_vector_idx"], reverse=True)
    for fp in files: 
        ret[fp] = [e["data_idx"] for e in first_good_eigen_vector_for_data_index[:num_samples] if e["fp"] == fp]

    for fp in files: 
        ret[fp].sort()
        print("{}: key-frame amount {}{:20}".format(fp, len(ret[fp]), " "))
    return ret



def fixedparco_keyframe_sampling(files, w_perc, num_eigen_vector=15, batch_size=30):
    ret = {}
    if batch_size is None: 
        batch_size = num_eigen_vector
    
    all_samples = []
    for fp in files: 
        ret[fp] = []
        print(fp, end="\r")
        df = pd.read_csv(fp)

        THRESHOLD_PREC_SVD = 99.8
        for data_window in df.rolling(batch_size): 
            if len(data_window) != batch_size:
                continue
            if IMAGE_ID_LIMIT is not None and int(data_window.iloc[0]["frame"]) >= IMAGE_ID_LIMIT: 
                break
            svd_info = calculate_svd(data_window.values)
            first_good_eigen_vector = None
            for e_i in range(1, num_eigen_vector + 1):
                perc = calculate_svd_reconstruction(data_window.values, svd_info, e_i)
                if first_good_eigen_vector is None and perc > THRESHOLD_PREC_SVD:
                    first_good_eigen_vector = e_i
            if first_good_eigen_vector is None:
                first_good_eigen_vector = num_eigen_vector
            all_samples.append({
                "fp": fp,
                "data_idx": int(data_window.iloc[0]["frame"]),
                "eigen_vector_idx": first_good_eigen_vector
            })
    
    for w_i in range(math.ceil(len(all_samples) / WINDOW)):
        start_w = w_i*WINDOW
        end_w = min((w_i+1)*WINDOW, len(all_samples))
        window_size = end_w - start_w
        num_samples_for_window = int(w_perc * window_size)
        curr_window = all_samples[start_w:end_w]
        curr_window_sampling = sorted(curr_window, key=lambda x: x["eigen_vector_idx"], reverse=True)[:num_samples_for_window]
        print("{}: fixed key-frame amount {} (window_size: {}) {:20}".format(w_i, len(curr_window_sampling), end_w-start_w, " "))
        for e in curr_window_sampling: 
            ret[e["fp"]].append(e["idx"])

    for fp in files: 
        ret[fp] = sorted(ret[fp])
        print("{}: key-frame amount {}{:20}".format(fp, len(ret[fp]), " "))

    return ret



def mpjpe_keyframe_sampling(files, num_samples, teacher, mpjpe_tresh=10):
    ret = {}
    all_actions_df = pd.DataFrame()
    file_teacher_map = {}
    for fp in files: 
        fp = fp.replace(" ", "")
        basename = os.path.basename(fp)
        action = basename.split(".")[0]
        if any(c.isdigit() for c in action):
            actions = [action, "{} {}".format(action[:-1], action[-1])]
        else: 
            actions = [action]
        if "WalkingDog" in actions:
            actions.extend(["WalkDog"])
        if "WalkingDog1" in actions:
            actions.extend(["WalkDog1", "WalkDog 1"])
        if "TakingPhoto" in actions:
            actions.extend(["Photo"])
        if "TakingPhoto1" in actions:
            actions.extend(["Photo1", "Photo 1"])
        folder_sub_dirname = os.path.dirname(os.path.dirname(fp))
        exist_teacher_fp = False
        for a in actions: 
            teacher_fp = os.path.join(folder_sub_dirname, teacher, "{}.{}.csv".format(a, basename.split(".")[1]))
            if os.path.exists(teacher_fp):
                exist_teacher_fp = True
                break
        if exist_teacher_fp: 
            file_teacher_map[fp] = teacher_fp
        else: 
            print("Warning: unable to calculate MPJPE of file {} with {} teacher".format(fp, teacher))

    for fp in file_teacher_map: 
        df_teacher = pd.read_csv(file_teacher_map[fp])[h36m_kp_names].iloc[:IMAGE_ID_LIMIT]
        df = pd.read_csv(fp)[h36m_kp_names].iloc[:IMAGE_ID_LIMIT]

        common_length = min(len(df.index), len(df_teacher.index))
        df = df.iloc[:common_length]
        df_teacher = df_teacher.iloc[:common_length]

        df_reshape = df.values.reshape((len(df.index), -1, 2))
        df_teacher_reshape = df_teacher.values.reshape((len(df_teacher.index), -1, 2))

        mpjpe = np.linalg.norm(df_reshape - df_teacher_reshape, axis=2)
        mpjpe = np.where(np.isnan(mpjpe), 0, mpjpe)
        df["AVG"] = mpjpe.mean(axis=1)

        df["fp"] = fp
        all_actions_df = pd.concat([all_actions_df, df], axis=0)
    
    all_actions_df = all_actions_df.sort_values("AVG", ascending=False)
    top_N_rows = all_actions_df.head(num_samples)
    for fp in files: 
        row_indices = list(top_N_rows[top_N_rows["fp"] == fp].index)
        row_indices.sort()
        ret[fp] = list(row_indices)
        print("{}: key-frame amount {}{:20}".format(fp, len(ret[fp]), " "))

    return ret


def fixedmpjpe_keyframe_sampling(files, w_perc, teacher, mpjpe_tresh=10):
    ret = {}
    
    all_actions_df = pd.DataFrame()
   
    file_teacher_map = {}
    for fp in files: 
        basename = os.path.basename(fp.replace(" ", ""))
        action = basename.split(".")[0]
        if any(c.isdigit() for c in action):
            actions = [action, "{} {}".format(action[:-1], action[-1])]
        else: 
            actions = [action]
        if "WalkingDog" in actions:
            actions.extend(["WalkDog"])
        if "WalkingDog1" in actions:
            actions.extend(["WalkDog1", "WalkDog 1"])
        if "TakingPhoto" in actions:
            actions.extend(["Photo"])
        if "TakingPhoto1" in actions:
            actions.extend(["Photo1", "Photo 1"])
        folder_sub_dirname = os.path.dirname(os.path.dirname(fp))
        exist_teacher_fp = False
        for a in actions: 
            teacher_fp = os.path.join(folder_sub_dirname, teacher, "{}.{}.csv".format(a, basename.split(".")[1]))
            if os.path.exists(teacher_fp):
                exist_teacher_fp = True
                break
        if exist_teacher_fp: 
            file_teacher_map[fp] = teacher_fp
        else: 
            print("Warning: unable to calculate MPJPE of file {} with {} teacher".format(fp, teacher))
    
    max_length = 0
    for fp in file_teacher_map: 
        ret[fp] = []
        df_teacher = pd.read_csv(file_teacher_map[fp])[h36m_kp_names].iloc[:IMAGE_ID_LIMIT]
        df = pd.read_csv(fp)[h36m_kp_names].iloc[:IMAGE_ID_LIMIT]

        common_length = min(len(df.index), len(df_teacher.index))
        max_length += max(len(df.index), len(df_teacher.index))
        df = df.iloc[:common_length]
        df_teacher = df_teacher.iloc[:common_length]

        df_reshape = df.values.reshape((len(df.index), -1, 2))
        df_teacher_reshape = df_teacher.values.reshape((len(df_teacher.index), -1, 2))

        mpjpe = np.linalg.norm(df_reshape - df_teacher_reshape, axis=2)
        mpjpe = np.where(np.isnan(mpjpe), 0, mpjpe)
        df["AVG"] = mpjpe.mean(axis=1)

        df["fp"] = fp
        all_actions_df = pd.concat([all_actions_df, df], axis=0)

    diff_length = max_length - len(all_actions_df.index)
    for w_i in range(math.ceil(len(all_actions_df.index) / WINDOW)):
        start_w = w_i*WINDOW
        end_w = min((w_i+1)*WINDOW, len(all_actions_df.index))
        window_size = end_w - start_w
        if end_w == len(all_actions_df.index):
            window_size += diff_length
        num_samples_for_window = int(w_perc * window_size)
        curr_window = all_actions_df.iloc[start_w:end_w]
        curr_window_sampling = curr_window.sort_values("AVG", ascending=False).head(num_samples_for_window)
        print("{}: fixed key-frame amount {} (window_size: {}) {:20}".format(w_i, len(curr_window_sampling.index), end_w-start_w, " "))
        for i, e in curr_window_sampling.iterrows(): 
            ret[e["fp"]].append(i)

    for fp in files: 
        ret[fp] = sorted(ret[fp])
        print("{}: key-frame amount {}{:20}".format(fp, len(ret[fp]), " "))

    return ret


def confidence_keyframe_sampling(files, num_samples):
    ret = {}
    all_actions_df = pd.DataFrame()
   
    for fp in files: 
        df = pd.read_csv(fp).iloc[:IMAGE_ID_LIMIT]
        fp = fp.replace(" ", "")
        df_acc = df.filter(like='ACC')
        min_val = df_acc.min(axis=1)
        df["minACC"] = min_val
        df["fp"] = fp
        all_actions_df = pd.concat([all_actions_df, df], axis=0)
    
    all_actions_df = all_actions_df.sort_values("minACC", ascending=True)
    top_N_rows = all_actions_df.head(num_samples)
    for fp in files: 
        row_indices = list(top_N_rows[top_N_rows["fp"] == fp].index)
        row_indices.sort()
        ret[fp] = list(row_indices)
        print("{}: key-frame amount {}{:20}".format(fp, len(ret[fp]), " "))

    return ret


def fixedconfidence_keyframe_sampling(files, w_perc):
    ret = {}
    all_actions_df = pd.DataFrame()

    for fp in files: 
        ret[fp] = []
        df = pd.read_csv(fp).iloc[:IMAGE_ID_LIMIT]
        fp = fp.replace(" ", "")
        df_acc = df.filter(like='ACC')
        min_val = df_acc.min(axis=1)
        df["minACC"] = min_val

        df["fp"] = fp
        all_actions_df = pd.concat([all_actions_df, df], axis=0)

    for w_i in range(math.ceil(len(all_actions_df.index) / WINDOW)):
        start_w = w_i*WINDOW
        end_w = min((w_i+1)*WINDOW, len(all_actions_df.index))
        window_size = end_w - start_w
        num_samples_for_window = int(w_perc * window_size)
        curr_window = all_actions_df.iloc[start_w:end_w]
        curr_window_sampling = curr_window.sort_values("minACC", ascending=True).head(num_samples_for_window)
        print("{}: fixed key-frame amount {} (window_size: {}) {:20}".format(w_i, len(curr_window_sampling.index), end_w-start_w, " "))
        for i, e in curr_window_sampling.iterrows(): 
            ret[e["fp"]].append(i)

    for fp in files: 
        ret[fp] = sorted(ret[fp])
        print("{}: key-frame amount {}{:20}".format(fp, len(ret[fp]), " "))

    return ret


def mean_confidence_keyframe_sampling(files, num_samples):
    ret = {}
    all_actions_df = pd.DataFrame()
    for fp in files: 
        df = pd.read_csv(fp).iloc[:IMAGE_ID_LIMIT]
        fp = fp.replace(" ", "")
        df_acc = df.filter(like='ACC')
        mean_val = df_acc.mean(axis=1)
        df["meanACC"] = mean_val
        df["fp"] = fp
        all_actions_df = pd.concat([all_actions_df, df], axis=0)
    
    all_actions_df = all_actions_df.sort_values("meanACC", ascending=True)
    top_N_rows = all_actions_df.head(num_samples)
    for fp in files: 
        row_indices = list(top_N_rows[top_N_rows["fp"] == fp].index)
        row_indices.sort()
        ret[fp] = list(row_indices)
        print("{}: key-frame amount {}{:20}".format(fp, len(ret[fp]), " "))

    return ret



def fixedmean_confidence_keyframe_sampling(files, w_perc):
    ret = {}
    
    all_actions_df = pd.DataFrame()
    for fp in files: 
        ret[fp] = []
        df = pd.read_csv(fp).iloc[:IMAGE_ID_LIMIT]
        fp = fp.replace(" ", "")
        df_acc = df.filter(like='ACC')
        mean_val = df_acc.mean(axis=1)
        df["meanACC"] = mean_val
        df["fp"] = fp
        all_actions_df = pd.concat([all_actions_df, df], axis=0)
    
    for w_i in range(math.ceil(len(all_actions_df.index) / WINDOW)):
        start_w = w_i*WINDOW
        end_w = min((w_i+1)*WINDOW, len(all_actions_df.index))
        window_size = end_w - start_w
        num_samples_for_window = int(w_perc * window_size)
        curr_window = all_actions_df.iloc[start_w:end_w]
        curr_window_sampling = curr_window.sort_values("meanACC", ascending=True).head(num_samples_for_window)
        print("{}: fixed key-frame amount {} (window_size: {}) {:20}".format(w_i, len(curr_window_sampling.index), end_w-start_w, " "))
        for i, e in curr_window_sampling.iterrows(): 
            ret[e["fp"]].append(i)

    for fp in files: 
        ret[fp] = sorted(ret[fp])
        print("{}: key-frame amount {}{:20}".format(fp, len(ret[fp]), " "))

    return ret


def get_keyframes(subjects, h36m_folder, sampling, perc, starting_model="trtpose_PARCO", teacher="vicon"):
    if sampling is None or perc is None: 
        return None
    print("Key-frame extraction - {}".format(sampling))
    lenghts_dataset = get_dataset_amount(subjects, h36m_folder)
    print(" - len dataset: {}".format(lenghts_dataset))

    keyframes = {}
    for camera in cameras:
        for sub in subjects:
            base_path = os.path.join(h36m_folder, sub, starting_model)
            files = glob.glob(os.path.join(base_path, "*" + camera + "*csv"))
            
            files = [f for f in files if "ALL" not in f]
            
            num_samples = int(lenghts_dataset[camera][sub] * perc)
            if sampling == "parco":
                cam_sub_keyframes = parco_keyframe_sampling(files, num_samples)
            elif sampling == "uniform":
                cam_sub_keyframes = uniform_sampling(files, num_samples)
            elif sampling == "random":
                cam_sub_keyframes = random_sampling(files, num_samples)
            elif sampling == "fixedrandom":
                cam_sub_keyframes = fixedrandom_sampling(files, perc)
            elif sampling == "action":
                cam_sub_keyframes = action_sampling(files, num_samples)
            elif sampling == "mpjpe":
                cam_sub_keyframes = mpjpe_keyframe_sampling(files, num_samples, teacher)
            elif sampling == "confidence":
                cam_sub_keyframes = confidence_keyframe_sampling(files, num_samples)
            elif sampling == "mean_confidence":
                cam_sub_keyframes = mean_confidence_keyframe_sampling(files, num_samples)
            elif sampling == "fixedmpjpe":
                cam_sub_keyframes = fixedmpjpe_keyframe_sampling(files, perc, teacher)
            elif sampling == "fixedconfidence":
                cam_sub_keyframes = fixedconfidence_keyframe_sampling(files, perc)
            elif sampling == "fixedmean_confidence":
                cam_sub_keyframes = fixedmean_confidence_keyframe_sampling(files, perc)
            else: 
                print("Error: sampling {} not recognized".format(sampling))
                exit()

            # Check amount: 
            num_keyframes = sum([len(cam_sub_keyframes[k]) for k in cam_sub_keyframes])
            if num_samples != num_keyframes: 
                print("Error: num samples required is not equal to the number of keyframes extracted for sampling {} and perc {} ({} != {})".format(
                    sampling, perc, num_samples, num_keyframes))
                exit()

            cam_sub_keyframes = {
                "{}/{}/{}".format(camera, sub, os.path.basename(k).replace(" ", "")): cam_sub_keyframes[k] 
                for k in cam_sub_keyframes
            }
        
            keyframes = {**keyframes, **cam_sub_keyframes}

    return keyframes


            
def generate_on_subjects(subjects, h36m_folder, coco_annotation, output_file, teacher="vicon", sampling=None, perc=None, enable_continual=False):

    out = {}
    annotations = []
    images = []

    with open(coco_annotation) as f: # /home/shared/nas/dataset/COCO/annotations/person_keypoints_val2017.json
        coco_json = json.load(f)

    # Initialize INFO, LICENSES and CATEGORIES
    out["info"] = coco_json["info"]
    out["categories"] = coco_json["categories"]
    out["licenses"] = [{
        "url": "http://vision.imar.ro/human3.6m/description.php",
        "id": 1,
        "name": "Human3.6M"
    }]
    out["info"]["description"] += " (Human3.6M source)"

    if enable_continual:
        out_continual_train = out.copy()
        out_continual_train["images"] = []
        out_continual_train["annotations"] = []
        out_continual_val = out_continual_train.copy()
        out_continual_val["images"] = []
        out_continual_val["annotations"] = []

    keyframes = get_keyframes(subjects, h36m_folder, sampling, perc, teacher=teacher)
    # print(list(keyframes.keys()))

    # Initialize IMAGES and ANNOTATIONS
    s = time.time()
    for camera in cameras: 
        for sub in subjects:

            base_path = os.path.join(h36m_folder, sub, teacher) # "/home/shared/nas/KnowledgeDistillation/h36m/" +  sub + "/vicon/"
            files = glob.glob(os.path.join(base_path, "*" + camera + "*csv"))

            for filename in files:
                filename_replaced = filename.split("/")[-1].replace(" ","_")
                try:
                    filename_replaced = filename_replaced.replace("_", "")
                    take = ".".join(filename_replaced.split("/")[-1].split("_")[0].split(".")[0:-1])
                except:
                    take = ".".join(filename_replaced.split("/")[-1].split("_")[0].split(".")[0:-1])
                cam = take.split(".")[-1]
                action = take.split(".")[0]

                if "ALL" in action:
                    continue # skip

                images_folder = os.path.join(h36m_folder, sub, cam, take)
                if not os.path.exists(images_folder):
                    if action == "WalkDog":
                        action = "WalkingDog"
                    if action == "WalkDog1":
                        action = "WalkingDog1"
                    if action == "Photo":
                        action = "TakingPhoto"
                    if action == "Photo1":
                        action = "TakingPhoto1"
                    take = "{}.{}".format(action, cam)

                print(filename, take)

                images_folder = os.path.join(h36m_folder, sub, cam, take)
                if not os.path.exists(images_folder):
                    print("Warning: the path {} does not exist in the dataset folder".format(images_folder))

                file = pd.read_csv(filename)
                
                continualtrain_images_count = 0 
                continualval_images_count = 0 
                images_count = 0 
                tot_frames = IMAGE_ID_LIMIT if IMAGE_ID_LIMIT is not None else len(file.index)
                
                # Check for exceeding keyframes
                if keyframes is not None:
                    keyframes["{}/{}/{}.csv".format(camera, sub, take)] = [kf
                        for kf in keyframes["{}/{}/{}.csv".format(camera, sub, take)]
                        if kf < len(file.index)]

                for i, r in file.iterrows(): 
                    if IMAGE_ID_LIMIT is not None and i >= IMAGE_ID_LIMIT: 
                        break
                    
                    if keyframes is not None and i not in keyframes[
                        "{}/{}/{}.csv".format(camera, sub, take)]:
                        continue

                    image_id = "{}/{}/{}/{}".format(sub, cam, take, i)
                    # print(image_id, end="\r")
                    image = get_image(image_id, H, W)
                    images.append(image)
                    annotation = get_annotation(i, r, image_id)
                    annotations.append(annotation)
                    if enable_continual:
                        if i <= int(tot_frames * CONTINUAL_TRAIN_PERC): 
                            out_continual_train["annotations"].append(annotation)
                            out_continual_train["images"].append(image)
                            continualtrain_images_count += 1
                        else: 
                            out_continual_val["annotations"].append(annotation)
                            out_continual_val["images"].append(image)
                            continualval_images_count += 1
                    images_count += 1
                
                if keyframes is not None and len(keyframes["{}/{}/{}.csv".format(camera, sub, take)]) != images_count:
                    print("Error: images count is not equal to the number of keyframes extracted for sampling {} and perc {} ({} != {})".format(
                        sampling, perc, len(keyframes["{}/{}/{}.csv".format(camera, sub, take)]), images_count))
                    print(filename)
                    print(keyframes["{}/{}/{}.csv".format(camera, sub, take)])
                    exit()

    e = time.time()
    print("elapsed run: {:.3f} ms {:.1f} fps".format((e - s) * 1000,  1 / (e - s)))
    print("images amount: {}; annotations amount: {}".format(len(images), len(annotations)))

    out["images"] = images
    out["annotations"] = annotations

    new_folderpath = os.path.join(h36m_folder, "..", "annotations")
    if not os.path.exists(new_folderpath):
        os.mkdir(new_folderpath)

    # Directly from dictionary
    if enable_continual:
        with open(os.path.join(new_folderpath, "continualtrain_" + output_file), "w") as outfile:
            json.dump(out_continual_train, outfile)
        with open(os.path.join(new_folderpath, "continualval_" + output_file), "w") as outfile:
            json.dump(out_continual_val, outfile)
        with open(os.path.join(new_folderpath, "continual_" + output_file), "w") as outfile:
            json.dump(out, outfile)
    else: 
        with open(os.path.join(new_folderpath, output_file), "w") as outfile:
            json.dump(out, outfile)


def main(h36m_folder, coco_annotation):
    # generate_on_subjects(["S1"], h36m_folder, coco_annotation, "person_keypoints_trainh36m_vicon.json", teacher="vicon")
    # generate_on_subjects(["S1"], h36m_folder, coco_annotation, "person_keypoints_trainh36m_openpose1.json", teacher="openpose1")
    # generate_on_subjects(["S9"], h36m_folder, coco_annotation, "person_keypoints_valh36m_vicon.json", teacher="vicon")
    # generate_on_subjects(["S9"], h36m_folder, coco_annotation, "person_keypoints_valh36m_openpose1.json", teacher="openpose1")

    generate_on_subjects(["S1", "S5", "S6", "S7", "S8"], h36m_folder, coco_annotation, "person_keypoints_trainh36m_vicon.json", teacher="vicon")
    generate_on_subjects(["S1", "S5", "S6", "S7", "S8"], h36m_folder, coco_annotation, "person_keypoints_trainh36m_openpose1.json", teacher="openpose1")
    generate_on_subjects(["S1", "S5", "S6", "S7", "S8"], h36m_folder, coco_annotation, "person_keypoints_trainh36m_CPN.json", teacher="CPN")
    generate_on_subjects(["S9", "S11"], h36m_folder, coco_annotation, "person_keypoints_valh36m_vicon.json", teacher="vicon")
    generate_on_subjects(["S9", "S11"], h36m_folder, coco_annotation, "person_keypoints_valh36m_openpose1.json", teacher="openpose1")
    generate_on_subjects(["S9", "S11"], h36m_folder, coco_annotation, "person_keypoints_valh36m_CPN.json", teacher="CPN")
    for teacher in TEACHERS: 
        for sampling in SAMPLINGS: 
            for perc in PERCENTAGES: 
                generate_on_subjects(["S1", "S5", "S6", "S7", "S8"], h36m_folder, coco_annotation, 
                                     "person_keypoints_trainh36m_{}sampling{}_{}.json".format(sampling, int(perc * 100), teacher), 
                                     teacher=teacher, sampling=sampling, perc=perc)
                generate_on_subjects(["S9", "S11"], h36m_folder, coco_annotation, 
                                    "person_keypoints_valh36m_{}sampling{}_{}.json".format(sampling, int(perc * 100), teacher), 
                                    teacher=teacher, sampling=sampling, perc=perc)
                
    # generate_on_subjects(["S1"], h36m_folder, coco_annotation, "person_keypoints_s1_vicon.json", teacher="vicon", enable_continual=True)
    # generate_on_subjects(["S1"], h36m_folder, coco_annotation, "person_keypoints_s1_openpose1.json", teacher="openpose1", enable_continual=True)
    # generate_on_subjects(["S1"], h36m_folder, coco_annotation, "person_keypoints_s1_CPN.json", teacher="CPN", enable_continual=True)

    # for teacher in TEACHERS: 
    #     for sampling in SAMPLINGS: 
    #         for perc in PERCENTAGES: 
    #             generate_on_subjects(["S1"], h36m_folder, coco_annotation, 
    #                                  "person_keypoints_s1_{}sampling{}_{}.json".format(sampling, int(perc * 100), teacher), 
    #                                  teacher=teacher, sampling=sampling, perc=perc, enable_continual=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human3.6M to COCO format", epilog="PARCO")
    parser.add_argument("--h36m-folder", 
                        "-hf", 
                        dest="h36m_folder", 
                        required=True, 
                        help="Folder with H36M")
    parser.add_argument("--coco-annotation", 
                        "-cf", 
                        dest="coco_annotation", 
                        required=True, 
                        help="File with COCO annotation")
    args = parser.parse_args()
    main(args.h36m_folder, args.coco_annotation)