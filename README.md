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


From Human3.6M dataset to COCO format: 
```
python3 h36m_to_coco.py -hf /home/shared/nas/KnowledgeDistillation/h36m -cf /home/shared/nas/dataset/COCO/annotations/person_keypoints_val2017.json
python3 h36m_to_coco.py -hf /home/shared/dataset/h36m -cf /home/shared/nas/dataset/COCO/annotations/person_keypoints_val2017.json
python3 h36m_to_coco.py -hf /home/shared/nas/KnowledgeDistillation/mini_h36m -cf /home/shared/nas/dataset/COCO/annotations/person_keypoints_val2017.json

# Jetson
python3 h36m_to_coco.py -hf /home/nvidia/dataset/h36m -cf /home/nvidia/nas/dataset/COCO/annotations/person_keypoints_val2017.json
```

Train:
```
export CUDA_VISIBLE_DEVICES=1
cd trt_pose
python3 trt_pose/train.py tasks/human_pose/experiments/nohead_densenet121_baseline_att_256x256_B.json

nohup bash scripts/copy_mini_h36m.bash &

nohup python3 trt_pose/train.py tasks/human_pose/experiments/nohead_densenet121_baseline_att_256x256_B.json &
nohup python3 trt_pose/train.py tasks/human_pose/experiments/h36m_nohead_densenet121_baseline_att_256x256_B.json &
nohup python3 trt_pose/train.py tasks/human_pose/experiments/uniform10_h36m_nohead_densenet121_baseline_att_256x256_B.json &
nohup python3 trt_pose/train.py tasks/human_pose/experiments/random10_h36m_nohead_densenet121_baseline_att_256x256_B.json &
nohup python3 trt_pose/train.py tasks/human_pose/experiments/action10_h36m_nohead_densenet121_baseline_att_256x256_B.json &
nohup python3 trt_pose/train.py tasks/human_pose/experiments/parco10_h36m_nohead_densenet121_baseline_att_256x256_B.json &

nohup python3 trt_pose/continual_train.py tasks/human_pose/experiments/continual_h36m_nohead_densenet121_baseline_att_256x256_B.json &

cp trt_pose/tasks/human_pose/experiments/h36m_nohead_densenet121_baseline_att_256x256_B.json.checkpoints/epoch_0.pth submodule/lib_maeve_py/maeve/nn/trtpose/models/
```

ParcoPose: 
```
nohup bash scripts/run_parcopose_h36m_vicon.bash 1> log/run_parcopose_h36m_vicon.log 2> log/run_parcopose_h36m_vicon_err.log &
nohup bash scripts/run_parcopose_h36m_openpose.bash 1> log/run_parcopose_h36m_openpose.log 2> log/run_parcopose_h36m_openpose_err.log &
nohup bash scripts/run_parcopose_h36m_CPN.bash 1> log/run_parcopose_h36m_CPN.log 2> log/run_parcopose_h36m_CPN_err.log &
nohup bash scripts/run_openpose.bash 1> log/run_openpose.log 2> log/run_openpose_err.log &

nohup bash scripts/run_parcopose_h36m_uniformsampling.bash 1> log/run_parcopose_h36m_uniformsampling.log 2> log/run_parcopose_h36m_uniformsampling_err.log &
nohup bash scripts/run_parcopose_h36m_randomsampling.bash 1> log/run_parcopose_h36m_randomsampling.log 2> log/run_parcopose_h36m_randomsampling_err.log &
nohup bash scripts/run_parcopose_h36m_actionsampling.bash 1> log/run_parcopose_h36m_actionsampling.log 2> log/run_parcopose_h36m_actionsampling_err.log &
nohup bash scripts/run_parcopose_h36m_parcosampling.bash 1> log/run_parcopose_h36m_parcosampling.log 2> log/run_parcopose_h36m_parcosampling_err.log &

nohup bash scripts/run_draw_parcopose_h36m.bash &

python3 parcopose_from_folder.py -f <folder>
python3 draw_parcopose_from_folder_jpg.py -f /home/shared/befine/lib_maeve_py/mirco_walking -n trtpose
DNN=openpose python3 draw_parcopose_from_folder_jpg.py -f /home/shared/befine/lib_maeve_py/mirco_walking -n openpose
python3 draw_parcopose_from_folder_jpg.py -f /home/shared/befine/lib_maeve_py/mirco_walking -n parcopose
python3 draw_parcopose_from_folder_jpg.py -f /home/shared/befine/lib_maeve_py/mirco_walking -n parcopose_h36m
python3 draw_parcopose_from_folder_jpg.py -f /home/shared/befine/lib_maeve_py/mirco_walking -n parcopose_h36m_openpose
python3 draw_parcopose_from_folder_jpg.py -f /home/shared/befine/lib_maeve_py/mirco_walking -n parcopose_h36m_CPN
mv /home/shared/befine/lib_maeve_py/*.mp4 .
python3 concat_video.py -n mirco_walking

ffmpeg -y -framerate 15 -i /home/shared/befine/lib_maeve_py/mirco_walking_trtpose/frame_%d.jpg -tag:v hvc1 -c:v libx265 -pix_fmt yuv420p mirco_walking_trtpose.mp4
ffmpeg -y -framerate 15 -i /home/shared/befine/lib_maeve_py/mirco_walking_parcopose/frame_%d.jpg -tag:v hvc1 -c:v libx265 -pix_fmt yuv420p mirco_walking_parcopose.mp4
ffmpeg -y -framerate 15 -i /home/shared/befine/lib_maeve_py/mirco_walking_parcoposeh36m/frame_%d.jpg -tag:v hvc1 -c:v libx265 -pix_fmt yuv420p mirco_walking_parcoposeh36m.mp4
ffmpeg -i mirco_walking_trtpose.mp4 -i mirco_walking_parcopose.mp4 -i mirco_walking_parcoposeh36m.mp4 -filter_complex "[0:v]crop=in_h:in_h:in_w/4:0[v0];[1:v]crop=in_h:in_h:in_w/4:0[v1];[2:v]crop=in_h:in_h:in_w/4:0[v2];[v0][v1][v2]hstack=inputs=3" mirco_walking_comparison.mp4
```

Validation: 
```
nohup python3 error.py -f /home/shared/nas/KnowledgeDistillation/h36m/ -r vicon -s openpose CPN trtpose trtpose_PARCO trtpose_retrained trtpose_retrained_op trtpose_retrained_CPN 1> log/validation.log 2> log/validation_err.log &
```
