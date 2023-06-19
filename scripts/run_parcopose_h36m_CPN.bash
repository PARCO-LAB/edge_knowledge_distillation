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

parcopose_h36m_CPN() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_CPN/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_CPN -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_retrained_CPN/
            done
        done
    done
}

parcopose_h36m_CPN

# Train
# cd /home/shared/befine/edge_knowledge_distillation/trt_pose
# python3 trt_pose/train.py tasks/human_pose/experiments/h36m_nohead_densenet121_baseline_att_256x256_B.json

echo "Done"
