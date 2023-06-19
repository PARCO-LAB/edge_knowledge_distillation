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

parcopose_h36m_parcosampling10_vicon() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_parcosampling10/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_parcosampling10_vicon -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_parcosampling10/
            done
        done
    done
}

parcopose_h36m_parcosampling10_openpose() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_parcosampling10_op/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_parcosampling10_openpose -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_parcosampling10_op/
            done
        done
    done
}

parcopose_h36m_parcosampling10_CPN() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_parcosampling10_CPN/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_parcosampling10_CPN -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_parcosampling10_CPN/
            done
        done
    done
}

parcopose_h36m_parcosampling20_vicon() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_parcosampling20/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_parcosampling20_vicon -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_parcosampling20/
            done
        done
    done
}

parcopose_h36m_parcosampling20_openpose() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_parcosampling20_op/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_parcosampling20_openpose -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_parcosampling20_op/
            done
        done
    done
}

parcopose_h36m_parcosampling20_CPN() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_parcosampling20_CPN/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_parcosampling20_CPN -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_parcosampling20_CPN/
            done
        done
    done
}

parcopose_h36m_parcosampling40_vicon() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_parcosampling40/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_parcosampling40_vicon -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_parcosampling40/
            done
        done
    done
}

parcopose_h36m_parcosampling40_openpose() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_parcosampling40_op/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_parcosampling40_openpose -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_parcosampling40_op/
            done
        done
    done
}

parcopose_h36m_parcosampling40_CPN() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_parcosampling40_CPN/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_parcosampling40_CPN -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_parcosampling40_CPN/
            done
        done
    done
}

# parcopose_h36m_parcosampling10_vicon
parcopose_h36m_parcosampling10_CPN
parcopose_h36m_parcosampling10_openpose

# parcopose_h36m_parcosampling20_vicon
# parcopose_h36m_parcosampling20_openpose
# parcopose_h36m_parcosampling20_CPN

# parcopose_h36m_parcosampling40_vicon
# parcopose_h36m_parcosampling40_openpose
# parcopose_h36m_parcosampling40_CPN

# Train
# cd /home/shared/befine/edge_knowledge_distillation/trt_pose
# python3 trt_pose/train.py tasks/human_pose/experiments/h36m_nohead_densenet121_baseline_att_256x256_B.json

echo "Done"
