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

parcopose_h36m_randomsampling10_vicon() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_randomsampling10/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_randomsampling10_vicon -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_randomsampling10/
            done
        done
    done
}

parcopose_h36m_randomsampling10_openpose() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_randomsampling10_op/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_randomsampling10_openpose -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_randomsampling10_op/
            done
        done
    done
}

parcopose_h36m_randomsampling10_CPN() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_randomsampling10_CPN/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_randomsampling10_CPN -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_randomsampling10_CPN/
            done
        done
    done
}


parcopose_h36m_randomsampling20_vicon() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_randomsampling20/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_randomsampling20_vicon -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_randomsampling20/
            done
        done
    done
}

parcopose_h36m_randomsampling20_openpose() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_randomsampling20_op/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_randomsampling20_openpose -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_randomsampling20_op/
            done
        done
    done
}

parcopose_h36m_randomsampling20_CPN() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_randomsampling20_CPN/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_randomsampling20_CPN -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_randomsampling20_CPN/
            done
        done
    done
}


parcopose_h36m_randomsampling40_vicon() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_randomsampling40/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_randomsampling40_vicon -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_randomsampling40/
            done
        done
    done
}

parcopose_h36m_randomsampling40_openpose() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_randomsampling40_op/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_randomsampling40_openpose -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_randomsampling40_op/
            done
        done
    done
}

parcopose_h36m_randomsampling40_CPN() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_randomsampling40_CPN/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_randomsampling40_CPN -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_randomsampling40_CPN/
            done
        done
    done
}


# parcopose_h36m_randomsampling10_vicon
parcopose_h36m_randomsampling10_CPN
parcopose_h36m_randomsampling10_openpose

# parcopose_h36m_randomsampling20_vicon
# parcopose_h36m_randomsampling20_openpose
# parcopose_h36m_randomsampling20_CPN

# parcopose_h36m_randomsampling40_vicon
# parcopose_h36m_randomsampling40_openpose
# parcopose_h36m_randomsampling40_CPN

# Train
# cd /home/shared/befine/edge_knowledge_distillation/trt_pose
# python3 trt_pose/train.py tasks/human_pose/experiments/h36m_nohead_densenet121_baseline_att_256x256_B.json

echo "Done"
