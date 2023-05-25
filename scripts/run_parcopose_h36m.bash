#!/bin/bash

SUBJECTS=(
    "S9"
    "S11"
    "S1"
    "S5"
    "S6"
    "S7"
    "S8"
)

CAMERAS=(
    "55011271"
)

parcopose() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parcopose
            done
            mv /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*.csv /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_PARCO/
        done
    done
}

parcoposeh36m() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parcoposeh36m
            done
            mv /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*.csv /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained/
        done
    done
}

# parcopose
parcoposeh36m

# Train
# cd /home/shared/befine/edge_knowledge_distillation/trt_pose
# python3 trt_pose/train.py tasks/human_pose/experiments/h36m_nohead_densenet121_baseline_att_256x256_B.json

echo "Done"
