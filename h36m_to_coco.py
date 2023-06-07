import argparse
import json
import os
import pandas as pd
import glob
import time
import random

from error import calculate_svd, calculate_svd_reconstruction


IMAGE_ID_LIMIT = 700
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
        ret[fp] = list(range(0, IMAGE_ID_LIMIT, IMAGE_ID_LIMIT // frame_amount_per_file))
        print("{}: key-frame amount {}{:20}".format(fp, len(ret[fp]), " "))

    return ret


def action_sampling(files, num_samples):
    ret = {}

    def check_action(f):
        for a in ["Sitting", "SittingDown", "Eating", "Purchases", "Greeting"]:
            if a in f: 
                return True
        return False

    files = [f for f in files if check_action(f)]
    num_files = len(files)
    frame_amount_per_file = int(num_samples / num_files)
    
    for fp in files: 
        ret[fp] = list(range(0, IMAGE_ID_LIMIT, IMAGE_ID_LIMIT // frame_amount_per_file))
        print("{}: key-frame amount {}{:20}".format(fp, len(ret[fp]), " "))

    return ret


def parco_keyframe_sampling(files, num_samples, num_eigen_vector=15, batch_size=30):
    ret = {}
    if batch_size is None: 
        batch_size = num_eigen_vector

    START_THRESHOLD_EIGEN_AMOUNT_SVD = num_eigen_vector - 1
    for fp in files: 
        ret[fp] = []
    samples = 0
    for thr_eigen_amount_svd in range(START_THRESHOLD_EIGEN_AMOUNT_SVD, 0, -1):
        for fp in files: 
            df = pd.read_csv(fp)
            ret[fp] = []

            THRESHOLD_PREC_SVD = 99.8
            for i, data_window in enumerate(df.rolling(batch_size)): 
                if IMAGE_ID_LIMIT is not None and i >= IMAGE_ID_LIMIT: 
                    break
                svd_info = calculate_svd(data_window.values)
                first_good_eigen_vector = None
                for e_i in range(1, num_eigen_vector + 1):
                    perc = calculate_svd_reconstruction(data_window.values, svd_info, e_i)
                    if first_good_eigen_vector is None and perc > THRESHOLD_PREC_SVD:
                        first_good_eigen_vector = e_i
                if first_good_eigen_vector is None:
                    first_good_eigen_vector = num_eigen_vector
                if first_good_eigen_vector > thr_eigen_amount_svd:
                    if int(data_window.iloc[0]["frame"]) not in ret[fp]:
                        ret[fp].append(int(data_window.iloc[0]["frame"]))
                        samples += 1
                    if samples >= num_samples: 
                        print("key-frame with {} amount of eigen vector: {}{:60}".format(thr_eigen_amount_svd, samples, ""))
                        return ret
                    
            print("{}: key-frame amount {}{:20}".format(fp, len(ret[fp]), " "), end="\r")
        print("key-frame with {} amount of eigen vector: {}{:60}".format(thr_eigen_amount_svd, samples, ""))

    return ret


def get_keyframes(subjects, h36m_folder, sampling, perc, starting_model="trtpose_PARCO"):
    if sampling is None or perc is None: 
        return None
    print("Key-frame extraction")
    lenghts_dataset = get_dataset_amount(subjects, h36m_folder)
    print(" - len dataset: {}".format(lenghts_dataset))

    keyframes = {}
    for camera in cameras:
        for sub in subjects:
            base_path = os.path.join(h36m_folder, sub, starting_model)
            files = glob.glob(os.path.join(base_path, "*" + camera + "*"))
            files = [f for f in files if "ALL" not in f]
            
            num_samples = int(lenghts_dataset[camera][sub] * perc)
            if sampling == "parco":
                cam_sub_keyframes = parco_keyframe_sampling(files, num_samples)
            elif sampling == "uniform":
                cam_sub_keyframes = uniform_sampling(files, num_samples)
            elif sampling == "random":
                cam_sub_keyframes = random_sampling(files, num_samples)
            elif sampling == "action":
                cam_sub_keyframes = action_sampling(files, num_samples)
            else: 
                print("Error: sampling {} not recognized".format(sampling))
                exit()

            cam_sub_keyframes = {
                "{}/{}/{}".format(camera, sub, os.path.basename(k).replace(" ", "")): cam_sub_keyframes[k] 
                for k in cam_sub_keyframes
            }
        
            keyframes = {**keyframes, **cam_sub_keyframes}

    return keyframes


            
def generate_on_subjects(subjects, h36m_folder, coco_annotation, output_file, teacher="vicon", sampling=None, perc=None):

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

    keyframes = get_keyframes(subjects, h36m_folder, sampling, perc)
    # print(list(keyframes.keys()))

    # Initialize IMAGES and ANNOTATIONS
    s = time.time()
    for camera in cameras: 
        for sub in subjects:

            base_path = os.path.join(h36m_folder, sub, teacher) # "/home/shared/nas/KnowledgeDistillation/h36m/" +  sub + "/vicon/"
            files = glob.glob(os.path.join(base_path, "*" + camera + "*"))

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
                
                for i, r in file.iterrows(): 
                    if IMAGE_ID_LIMIT is not None and i >= IMAGE_ID_LIMIT: 
                        break
                    
                    if keyframes is not None and i not in keyframes[
                        "{}/{}/{}.csv".format(camera, sub, take)]:
                        continue

                    image_id = "{}/{}/{}/{}".format(sub, cam, take, i)
                    print(image_id, end="\r")
                    image = get_image(image_id, H, W)
                    images.append(image)
                    annotation = get_annotation(i, r, image_id)
                    annotations.append(annotation)

    e = time.time()
    print("elapsed run: {:.3f} ms {:.1f} fps".format((e - s) * 1000,  1 / (e - s)))
    print("images amount: {}; annotations amount: {}".format(len(images), len(annotations)))

    out["images"] = images
    out["annotations"] = annotations

    new_folderpath = os.path.join(h36m_folder, "..", "annotations")
    if not os.path.exists(new_folderpath):
        os.mkdir(new_folderpath)

    # Directly from dictionary
    with open(os.path.join(new_folderpath, output_file), "w") as outfile:
        json.dump(out, outfile)


def main(h36m_folder, coco_annotation):
    # generate_on_subjects(["S1"], h36m_folder, coco_annotation, "person_keypoints_trainh36m_vicon.json", teacher="vicon")
    # generate_on_subjects(["S1"], h36m_folder, coco_annotation, "person_keypoints_trainh36m_openpose.json", teacher="openpose")
    # generate_on_subjects(["S9"], h36m_folder, coco_annotation, "person_keypoints_valh36m_vicon.json", teacher="vicon")
    # generate_on_subjects(["S9"], h36m_folder, coco_annotation, "person_keypoints_valh36m_openpose.json", teacher="openpose")

    # generate_on_subjects(["S1", "S5", "S6", "S7", "S8"], h36m_folder, coco_annotation, "person_keypoints_trainh36m_vicon.json", teacher="vicon")
    # generate_on_subjects(["S1", "S5", "S6", "S7", "S8"], h36m_folder, coco_annotation, "person_keypoints_trainh36m_openpose.json", teacher="openpose")
    # generate_on_subjects(["S1", "S5", "S6", "S7", "S8"], h36m_folder, coco_annotation, "person_keypoints_trainh36m_CPN.json", teacher="CPN")
    # generate_on_subjects(["S9", "S11"], h36m_folder, coco_annotation, "person_keypoints_valh36m_vicon.json", teacher="vicon")
    # generate_on_subjects(["S9", "S11"], h36m_folder, coco_annotation, "person_keypoints_valh36m_openpose.json", teacher="openpose")
    # generate_on_subjects(["S9", "S11"], h36m_folder, coco_annotation, "person_keypoints_valh36m_CPN.json", teacher="CPN")

    for teacher in ["vicon", "openpose", "CPN"]:
        for sampling in ["uniform", "random", "action", "parco"]:
            for perc in [0.1, 0.2, 0.4]:
                generate_on_subjects(["S1", "S5", "S6", "S7", "S8"], h36m_folder, coco_annotation, 
                                     "person_keypoints_trainh36m_{}sampling{}_{}.json".format(sampling, int(perc * 100), teacher), 
                                     teacher=teacher, sampling=sampling, perc=perc)
                generate_on_subjects(["S9", "S11"], h36m_folder, coco_annotation, 
                                     "person_keypoints_valh36m_{}sampling{}_{}.json".format(sampling, int(perc * 100), teacher), 
                                     teacher=teacher, sampling=sampling, perc=perc)


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