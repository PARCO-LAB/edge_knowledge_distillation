import argparse
import json
import os
import pandas as pd
import numpy as np
import glob
import time
import random

SEED=213445
random.seed(SEED)

from error import calculate_svd, calculate_svd_reconstruction, calculate_mpjpe, h36m_kp_names


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
                if os.path.exists(os.path.join(f, "{}.png".format(int(IMAGE_ID_LIMIT - 1)))):
                    amount += IMAGE_ID_LIMIT
                    continue

                for i in range(IMAGE_ID_LIMIT - 1, -1, -1):
                    if os.path.exists(os.path.join(f, "{}.png".format(int(i)))):
                        amount += i
                    else: 
                        break

            amount_dict[camera][sub] = amount
    return amount_dict


def random_sampling(files, num_samples):
    ret = {}
    num_files = len(files)
    frame_amount_per_file = int(num_samples / num_files)
    
    for fp in files: 
        ret[fp] = list(sorted(random.sample(range(IMAGE_ID_LIMIT), frame_amount_per_file)))
        print("{}: key-frame amount {}{:20}".format(fp, len(ret[fp]), " "))

    return ret


def uniform_sampling(files, num_samples):
    ret = {}
    num_files = len(files)
    
    frame_amount_per_file = int(num_samples / num_files)
    
    for fp in files: 
        ret[fp] = list(range(0, IMAGE_ID_LIMIT, IMAGE_ID_LIMIT // frame_amount_per_file))[:frame_amount_per_file]
        print("{}: key-frame amount {}{:20}".format(fp, len(ret[fp]), " "))

    return ret


def action_sampling(files, num_samples):
    ret = {}

    def check_action(f):
        for a in ["Sitting", "SittingDown", "Eating", "Purchases", "Greeting", "Phoning", "Posing"]:
            if a in f: 
                return True
        return False
    
    for fp in files: 
        ret[fp] = {}

    files = [f for f in files if check_action(f)]
    num_files = len(files)
    frame_amount_per_file = int(num_samples / num_files)
    
    for fp in files: 
        ret[fp] = list(range(0, IMAGE_ID_LIMIT, IMAGE_ID_LIMIT // frame_amount_per_file))[:frame_amount_per_file]
        print("{}: key-frame amount {}{:20}".format(fp, len(ret[fp]), " "))

    return ret


def parco_keyframe_sampling(files, num_samples, num_eigen_vector=15, batch_size=30, continual=False):
    ret = {}
    if batch_size is None: 
        batch_size = num_eigen_vector
    
    num_files = len(files)
    frame_amount_per_file = int(num_samples / num_files)
    if continual: 
        first_good_eigen_vector_for_data_index = []
    for fp in files: 
        print(fp, end="\r")
        df = pd.read_csv(fp)

        THRESHOLD_PREC_SVD = 99.8
        if not continual: 
            first_good_eigen_vector_for_data_index = []
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
            if continual: 
                first_good_eigen_vector_for_data_index.append({
                    "fp": fp,
                    "data_idx": int(data_window.iloc[0]["frame"]),
                    "eigen_vector_idx": first_good_eigen_vector
                })
            else: 
                first_good_eigen_vector_for_data_index.append({
                    "data_idx": int(data_window.iloc[0]["frame"]),
                    "eigen_vector_idx": first_good_eigen_vector
                })
        if not continual:
            first_good_eigen_vector_for_data_index = sorted(
                first_good_eigen_vector_for_data_index, key=lambda x: x["eigen_vector_idx"], reverse=True)
        
            ret[fp] = [e["data_idx"] for e in first_good_eigen_vector_for_data_index[:frame_amount_per_file]]
    if continual:
        first_good_eigen_vector_for_data_index = sorted(
            first_good_eigen_vector_for_data_index, key=lambda x: x["eigen_vector_idx"], reverse=True)
        for fp in files: 
            ret[fp] = [e["data_idx"] for e in first_good_eigen_vector_for_data_index[:num_samples] if e["fp"] == fp]

    for fp in files: 
        ret[fp].sort()
        print("{}: key-frame amount {}{:20}".format(fp, len(ret[fp]), " "))
    return ret



def mpjpe_keyframe_sampling(files, num_samples, teacher, mpjpe_tresh=10, continual=False):
    ret = {}
    
    if continual: 
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

    num_files = len(files)
    frame_amount_per_file = int(num_samples / num_files)
    diff_num_files = len(files) - len(file_teacher_map)
    frame_amount_to_add = diff_num_files * frame_amount_per_file
    frame_amount_to_add_for_file = int(np.ceil(frame_amount_to_add / len(file_teacher_map)))
    
    for fp in file_teacher_map: 
        df_teacher = pd.read_csv(file_teacher_map[fp])[h36m_kp_names].iloc[:IMAGE_ID_LIMIT]
        df = pd.read_csv(fp)[h36m_kp_names].iloc[:IMAGE_ID_LIMIT]

        df_reshape = df.values.reshape((len(df.index), -1, 2))
        df_teacher_reshape = df_teacher.values.reshape((len(df_teacher.index), -1, 2))

        mpjpe = np.linalg.norm(df_reshape - df_teacher_reshape, axis=2)
        mpjpe = np.where(np.isnan(mpjpe), 0, mpjpe)
        df["AVG"] = mpjpe.mean(axis=1)

        if continual: 
            df["fp"] = fp
            all_actions_df = pd.concat([all_actions_df, df], axis=0)
        else: 
            df_sorted = df.sort_values("AVG", ascending=False)
            top_N_rows = df_sorted.head(frame_amount_per_file + min(frame_amount_to_add_for_file, frame_amount_to_add))
            row_indices = list(top_N_rows.index)
            row_indices.sort()
            ret[fp] = list(row_indices)
            print("{}: key-frame amount {}{:20}".format(fp, len(ret[fp]), " "))
        if (frame_amount_to_add - frame_amount_to_add_for_file) > 0:
            frame_amount_to_add -= frame_amount_to_add_for_file
        else: 
            frame_amount_to_add = 0
    
    if continual: 
        all_actions_df = all_actions_df.sort_values("AVG", ascending=False)
        top_N_rows = all_actions_df.head(num_samples)
        for fp in files: 
            row_indices = list(top_N_rows[top_N_rows["fp"] == fp].index)
            row_indices.sort()
            ret[fp] = list(row_indices)
            print("{}: key-frame amount {}{:20}".format(fp, len(ret[fp]), " "))

    return ret


def confidence_keyframe_sampling(files, num_samples, continual=False):

    ret = {}
    
    num_files = len(files)
    frame_amount_per_file = int(num_samples / num_files)

    if continual: 
        all_actions_df = pd.DataFrame()
   
    for fp in files: 
        df = pd.read_csv(fp).iloc[:IMAGE_ID_LIMIT]
        fp = fp.replace(" ", "")
        df_acc = df.filter(like='ACC')
        min_val = df_acc.min(axis=1)
        df["minACC"] = min_val

        if continual: 
            df["fp"] = fp
            all_actions_df = pd.concat([all_actions_df, df], axis=0)
        else: 
            df_sorted = df.sort_values("minACC", ascending=True)
            top_N_rows = df_sorted.head(frame_amount_per_file)
            row_indices = list(top_N_rows.index)
            row_indices.sort()
            ret[fp] = list(row_indices)
            print("{}: key-frame amount {}{:20}".format(fp, len(ret[fp]), " "), end="\r")
    
    if continual: 
        all_actions_df = all_actions_df.sort_values("minACC", ascending=True)
        top_N_rows = all_actions_df.head(num_samples)
        for fp in files: 
            row_indices = list(top_N_rows[top_N_rows["fp"] == fp].index)
            row_indices.sort()
            ret[fp] = list(row_indices)
            print("{}: key-frame amount {}{:20}".format(fp, len(ret[fp]), " "))

    return ret


def mean_confidence_keyframe_sampling(files, num_samples, continual=False):

    ret = {}
    
    num_files = len(files)
    frame_amount_per_file = int(num_samples / num_files)

    if continual: 
        all_actions_df = pd.DataFrame()
   
    for fp in files: 
        df = pd.read_csv(fp).iloc[:IMAGE_ID_LIMIT]
        fp = fp.replace(" ", "")
        df_acc = df.filter(like='ACC')
        mean_val = df_acc.mean(axis=1)
        df["meanACC"] = mean_val

        if continual: 
            df["fp"] = fp
            all_actions_df = pd.concat([all_actions_df, df], axis=0)
        else: 
            df_sorted = df.sort_values("meanACC", ascending=True)
            top_N_rows = df_sorted.head(frame_amount_per_file)
            row_indices = list(top_N_rows.index)
            row_indices.sort()
            ret[fp] = list(row_indices)
            print("{}: key-frame amount {}{:20}".format(fp, len(ret[fp]), " "), end="\r")
    
    if continual: 
        all_actions_df = all_actions_df.sort_values("meanACC", ascending=True)
        top_N_rows = all_actions_df.head(num_samples)
        for fp in files: 
            row_indices = list(top_N_rows[top_N_rows["fp"] == fp].index)
            row_indices.sort()
            ret[fp] = list(row_indices)
            print("{}: key-frame amount {}{:20}".format(fp, len(ret[fp]), " "))

    return ret


def get_keyframes(subjects, h36m_folder, sampling, perc, starting_model="trtpose_PARCO", teacher="vicon", continual=False):
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
                cam_sub_keyframes = parco_keyframe_sampling(files, num_samples, continual=continual)
            elif sampling == "uniform":
                cam_sub_keyframes = uniform_sampling(files, num_samples)
            elif sampling == "random":
                cam_sub_keyframes = random_sampling(files, num_samples)
            elif sampling == "action":
                cam_sub_keyframes = action_sampling(files, num_samples)
            elif sampling == "mpjpe":
                cam_sub_keyframes = mpjpe_keyframe_sampling(files, num_samples, teacher, continual=continual)
            elif sampling == "confidence":
                cam_sub_keyframes = confidence_keyframe_sampling(files, num_samples, continual=continual)
            elif sampling == "mean_confidence":
                cam_sub_keyframes = mean_confidence_keyframe_sampling(files, num_samples, continual=continual)
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

    keyframes = get_keyframes(subjects, h36m_folder, sampling, perc, teacher=teacher, continual=enable_continual)
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
                        if i <= int(IMAGE_ID_LIMIT * CONTINUAL_TRAIN_PERC): 
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
    else: 
        with open(os.path.join(new_folderpath, output_file), "w") as outfile:
            json.dump(out, outfile)


def main(h36m_folder, coco_annotation):
    # generate_on_subjects(["S1"], h36m_folder, coco_annotation, "person_keypoints_trainh36m_vicon.json", teacher="vicon")
    # generate_on_subjects(["S1"], h36m_folder, coco_annotation, "person_keypoints_trainh36m_openpose.json", teacher="openpose")
    # generate_on_subjects(["S9"], h36m_folder, coco_annotation, "person_keypoints_valh36m_vicon.json", teacher="vicon")
    # generate_on_subjects(["S9"], h36m_folder, coco_annotation, "person_keypoints_valh36m_openpose.json", teacher="openpose")

    generate_on_subjects(["S1", "S5", "S6", "S7", "S8"], h36m_folder, coco_annotation, "person_keypoints_trainh36m_vicon.json", teacher="vicon")
    generate_on_subjects(["S1", "S5", "S6", "S7", "S8"], h36m_folder, coco_annotation, "person_keypoints_trainh36m_openpose.json", teacher="openpose1")
    generate_on_subjects(["S1", "S5", "S6", "S7", "S8"], h36m_folder, coco_annotation, "person_keypoints_trainh36m_CPN.json", teacher="CPN")
    generate_on_subjects(["S9", "S11"], h36m_folder, coco_annotation, "person_keypoints_valh36m_vicon.json", teacher="vicon")
    generate_on_subjects(["S9", "S11"], h36m_folder, coco_annotation, "person_keypoints_valh36m_openpose.json", teacher="openpose1")
    generate_on_subjects(["S9", "S11"], h36m_folder, coco_annotation, "person_keypoints_valh36m_CPN.json", teacher="CPN")
    
    generate_on_subjects(["S1"], h36m_folder, coco_annotation, "person_keypoints_s1_vicon.json", teacher="vicon", enable_continual=True)
    generate_on_subjects(["S1"], h36m_folder, coco_annotation, "person_keypoints_s1_openpose.json", teacher="openpose1", enable_continual=True)
    generate_on_subjects(["S1"], h36m_folder, coco_annotation, "person_keypoints_s1_CPN.json", teacher="CPN", enable_continual=True)

    for teacher in ["vicon", "openpose1", "CPN"]: # ["vicon", "openpose", "CPN"]
        for sampling in ["mean_confidence", "confidence", "mpjpe", "uniform", "random", "action", "parco"]: # ["mean_confidence", "confidence", "mpjpe", "uniform", "random", "action", "parco"]
            for perc in [0.01, 0.05, 0.1, 0.2, 0.4]: # [0.01, 0.05, 0.1, 0.2, 0.4]
                generate_on_subjects(["S1", "S5", "S6", "S7", "S8"], h36m_folder, coco_annotation, 
                                     "person_keypoints_trainh36m_{}sampling{}_{}.json".format(sampling, int(perc * 100), teacher), 
                                     teacher=teacher, sampling=sampling, perc=perc)
                generate_on_subjects(["S9", "S11"], h36m_folder, coco_annotation, 
                                    "person_keypoints_valh36m_{}sampling{}_{}.json".format(sampling, int(perc * 100), teacher), 
                                    teacher=teacher, sampling=sampling, perc=perc)
                generate_on_subjects(["S1"], h36m_folder, coco_annotation, 
                                     "person_keypoints_s1_{}sampling{}_{}.json".format(sampling, int(perc * 100), teacher), 
                                     teacher=teacher, sampling=sampling, perc=perc, enable_continual=True)


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