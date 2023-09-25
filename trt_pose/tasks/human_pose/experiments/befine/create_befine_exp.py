

import json
import os
with open('blueprint.json') as f:
    d = json.load(f)

BASE_PATH='/home/saldegheri/nas/MAEVE/BEFINE/ice_demo_7cam'

for i in range(3,10):
    os.makedirs(f'jetsonzed0{i}', exist_ok=True)

    # images dir is the same for everyone, id is embedded in annotations
    # d['train_dataset']['images_dir']
    # d['test_dataset']['images_dir']

    d['train_dataset']['images_dir'] = BASE_PATH
    d['test_dataset']['images_dir'] = BASE_PATH
    
    d['train_dataset']['annotations_file'] = f'{BASE_PATH}/jetsonzed0{i}/single_person/openpose_annotations.json'
    d['test_dataset']['annotations_file'] = f'{BASE_PATH}/jetsonzed0{i}/single_person/openpose_annotations.json'
    
    d['train_dataset']['keep_aspect_ratio'] = True
    d['test_dataset']['keep_aspect_ratio'] = True
    with open(f'jetsonzed0{i}/densenet_PP3.json', 'w') as f:
        json.dump(d, f, indent=4)
    # d['train_dataset']['keep_aspect_ratio'] = True
    # d['test_dataset']['keep_aspect_ratio'] = True
    # with open(f'jetsonzed0{i}/densenet_kuka_crop.json', 'w') as f:
    #     json.dump(d, f, indent=4)


    d['train_dataset']['annotations_file'] = f'{BASE_PATH}/jetsonzed0{i}/annotations.json'
    d['test_dataset']['annotations_file'] = f'{BASE_PATH}/jetsonzed0{i}/annotations.json'
    
    d['train_dataset']['keep_aspect_ratio'] = True
    d['test_dataset']['keep_aspect_ratio'] = True
    with open(f'jetsonzed0{i}/densenet_PP4.json', 'w') as f:
        json.dump(d, f, indent=4)
    # d['train_dataset']['keep_aspect_ratio'] = True
    # d['test_dataset']['keep_aspect_ratio'] = True
    # with open(f'jetsonzed0{i}/densenet_full_crop.json', 'w') as f:
    #     json.dump(d, f, indent=4)