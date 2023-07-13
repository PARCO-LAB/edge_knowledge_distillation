import json
import pandas as pd
import glob
import argparse
import numpy as np

def main(order_file, index_file, output_file, subject):

    W, H = 1000, 1000

    def get_annotation(id, keypoints, image_id):
        trt_kp_names = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 
                        'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        vicon_kp_names = ['LShoulder:V', 'LShoulder:U', 'RShoulder:V', 'RShoulder:U','LElbow:V', 'LElbow:U','RElbow:V',
                           'RElbow:U', 'LWrist:V', 'LWrist:U', 'RWrist:V', 'RWrist:U', 'LHip:V', 'LHip:U', 'RHip:V',
                             'RHip:U', 'LKnee:V', 'LKnee:U','RKnee:V', 'RKnee:U', 
                'LAnkle:V', 'LAnkle:U', 'RAnkle:V', 'RAnkle:U']
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
            "license": 4,
            "file_name": id,
            "coco_url": str(url),
            "height": int(height),
            "width": int(width),
            "date_captured": str(date),
            "flickr_url": str(url),
            "id": id
        }
        return ret

    def fix_category_excluding_face(cat):
        return cat
        cat["keypoints"] = cat["keypoints"][5:]
        new_cat_skeleton = []
        for idx_pair in cat["skeleton"]:
            mask_drop = [0 for v in idx_pair if v < 5] 
            if len(mask_drop) == 0:
                idx_pair[0] -= 5
                idx_pair[1] -= 5
                new_cat_skeleton.append(idx_pair)
        cat["skeleton"] = new_cat_skeleton
        return cat


    camera = '55011271'

    # not important the content, just to fix categories
    with open('/home/shared/nas/KnowledgeDistillation/annotations/continual_person_keypoints_s1_actionsampling1_CPN.json') as f:
        coco_json = json.load(f)

    train_images = []
    import csv
    order_indexes = {}
    order_indexes_arr = []
    file_indexes = {}
    file_indexes_arr = []
    with open(order_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            order_indexes[row[1]] = row[0]
            order_indexes_arr.append(row[1])
    
    n = 0
    with open(index_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            id_in_order = int(row[0])
            fname = order_indexes_arr[id_in_order]
            # set what is the expected id
            file_indexes[fname] = n
            n = n + 1
    # preallocate size
    annotations = [None for i in file_indexes]
    images = [None for i in file_indexes]
    
    base_path = '/home/shared/nas//KnowledgeDistillation/h36m/' +  subject + '/vicon/'

    files = glob.glob(base_path + '*' + camera + '*')

    for filename in files:
        filename_replaced = filename.split('/')[-1].replace(" ","_")

        try:
            filename_replaced = filename_replaced.replace('_','.')
            take = '.'.join(filename_replaced.split('/')[-1].split('_')[0].split('.')[0:-1])
            print(take)
        except:
            take = '.'.join(filename_replaced.split('/')[-1].split('_')[0].split('.')[0:-1])
            print(take)
        
        file = pd.read_csv(filename)
        
        
        for i, r in file.iterrows(): 
            cam = take.split('.')[-1]
            action = ''.join(take.split('.')[:-1])
            dirname =  f'{cam}/{action}.{cam}'
            image_id = "{}/{}/{}.png".format(subject, dirname, i)
            
            current_idx = None
            try:
                # generates error if not existing, so behave like HashMap
                current_idx = file_indexes[image_id]
            except KeyError:
                continue
            
            # set to the correct id    
            image = get_image(image_id, H, W)
            images[current_idx] = image
            annotation = get_annotation(i, r, image_id)
            annotations[current_idx] = annotation

    out = {}
    out['info'] = coco_json['info']
    out['categories'] = [fix_category_excluding_face(coco_json['categories'][0])]
    # out['categories'] = [coco_json['categories'][0]]
        
    out["images"] = images
    out["annotations"] = annotations

    # Directly from dictionary
    with open(output_file, 'w') as outfile:
        json.dump(out, outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate COCO from order", epilog="PARCO")
    parser.add_argument("--order") 
    parser.add_argument("--index")
    parser.add_argument("--output", default="validation_data.json")
    parser.add_argument("--subject", default="S1")
    args = parser.parse_args()
    main(args.order, args.index, args.output, args.subject)


