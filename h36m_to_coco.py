import argparse
import json
import os
import pandas as pd
import glob
import time


IMAGE_ID_LIMIT = 700
W, H = 1000, 1000


def get_annotation(id, keypoints, image_id):
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


def generate_on_subjects(subjects, h36m_folder, coco_annotation, output_file, teacher="vicon"):
    cameras = ["55011271"]

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
                print(filename, take)
                cam = take.split(".")[-1]

                file = pd.read_csv(filename)
                
                for i, r in file.iterrows(): 
                    if IMAGE_ID_LIMIT is not None and i >= IMAGE_ID_LIMIT: 
                        break
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
    generate_on_subjects(["S1"], h36m_folder, coco_annotation, "person_keypoints_trainh36m_vicon.json", teacher="vicon")
    generate_on_subjects(["S1"], h36m_folder, coco_annotation, "person_keypoints_trainh36m_openpose.json", teacher="openpose")
    generate_on_subjects(["S9"], h36m_folder, coco_annotation, "person_keypoints_valh36m_vicon.json", teacher="vicon")
    generate_on_subjects(["S9"], h36m_folder, coco_annotation, "person_keypoints_valh36m_openpose.json", teacher="openpose")

    # generate_on_subjects(["S1", "S5", "S6", "S7", "S8"], h36m_folder, coco_annotation, "person_keypoints_trainh36m_vicon.json", teacher="vicon")
    # generate_on_subjects(["S1", "S5", "S6", "S7", "S8"], h36m_folder, coco_annotation, "person_keypoints_trainh36m_openpose.json", teacher="openpose")
    # generate_on_subjects(["S9", "S11"], h36m_folder, coco_annotation, "person_keypoints_valh36m_vicon.json", teacher="vicon")
    # generate_on_subjects(["S9", "S11"], h36m_folder, coco_annotation, "person_keypoints_valh36m_openpose.json", teacher="openpose")


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