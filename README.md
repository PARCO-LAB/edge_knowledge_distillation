# Edge Knowledge Distillation

Knowledge Distillation at the Edge. 

## Setup

Download the library and initialize submodules in a single command:
```
git clone --recurse-submodules -j8 https://github.com/PARCO-LAB/edge_knowledge_distillation.git
```

Initialize submodules after library download:
```
git submodule update --init --recursive
```

Dataset: 
```
sudo mount -t cifs //157.27.95.61/MAEVE/ /home/shared/nas -o uid=$(id -u),gid=shared,file_mode=0664,dir_mode=0775,username=server_parco,password=Bagigi123.

wget http://images.cocodataset.org/zips/train2017.zip && unzip train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip && unzip val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip annotations_trainval2017.zip

python3 tasks/human_pose/preprocess_coco_person.py /home/shared/dataset/COCO/annotations/person_keypoints_train2017.json /home/shared/dataset/COCO/annotations/person_keypoints_train2017_modified.json
python3 tasks/human_pose/preprocess_coco_person.py /home/shared/dataset/COCO/annotations/person_keypoints_val2017.json /home/shared/dataset/COCO/annotations/person_keypoints_val2017_modified.json
```

## Dependencies

Apex:
```
git clone https://github.com/NVIDIA/apex apex && cd apex
git checkout 22.04-dev
pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Usage

```
cd trt_pose
python3 trt_pose/train.py tasks/human_pose/experiments/nohead_densenet121_baseline_att_256x256_B.json

nohup python3 trt_pose/train.py tasks/human_pose/experiments/nohead_densenet121_baseline_att_256x256_B.json &
```