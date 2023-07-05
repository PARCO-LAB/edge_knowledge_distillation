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

parco_h36m_randomsampling1_vicon() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling1_vicon/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_randomsampling1_vicon -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling1_vicon/
            done
        done
    done
}

parco_h36m_randomsampling5_vicon() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling5_vicon/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_randomsampling5_vicon -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling5_vicon/
            done
        done
    done
}

parco_h36m_randomsampling10_vicon() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling10_vicon/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_randomsampling10_vicon -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling10_vicon/
            done
        done
    done
}

parco_h36m_randomsampling20_vicon() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling20_vicon/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_randomsampling20_vicon -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling20_vicon/
            done
        done
    done
}

parco_h36m_randomsampling40_vicon() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling40_vicon/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_randomsampling40_vicon -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling40_vicon/
            done
        done
    done
}

parco_h36m_randomsampling1_openpose1() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling1_openpose1/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_randomsampling1_openpose1 -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling1_openpose1/
            done
        done
    done
}

parco_h36m_randomsampling5_openpose1() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling5_openpose1/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_randomsampling5_openpose1 -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling5_openpose1/
            done
        done
    done
}

parco_h36m_randomsampling10_openpose1() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling10_openpose1/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_randomsampling10_openpose1 -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling10_openpose1/
            done
        done
    done
}

parco_h36m_randomsampling20_openpose1() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling20_openpose1/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_randomsampling20_openpose1 -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling20_openpose1/
            done
        done
    done
}

parco_h36m_randomsampling40_openpose1() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling40_openpose1/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_randomsampling40_openpose1 -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling40_openpose1/
            done
        done
    done
}

parco_h36m_randomsampling1_CPN() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling1_CPN/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_randomsampling1_CPN -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling1_CPN/
            done
        done
    done
}

parco_h36m_randomsampling5_CPN() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling5_CPN/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_randomsampling5_CPN -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling5_CPN/
            done
        done
    done
}

parco_h36m_randomsampling10_CPN() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling10_CPN/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_randomsampling10_CPN -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling10_CPN/
            done
        done
    done
}

parco_h36m_randomsampling20_CPN() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling20_CPN/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_randomsampling20_CPN -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling20_CPN/
            done
        done
    done
}

parco_h36m_randomsampling40_CPN() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling40_CPN/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_randomsampling40_CPN -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_randomsampling40_CPN/
            done
        done
    done
}

parco_h36m_randomsampling1_vicon
parco_h36m_randomsampling5_vicon
parco_h36m_randomsampling10_vicon
parco_h36m_randomsampling20_vicon
parco_h36m_randomsampling40_vicon
parco_h36m_randomsampling1_openpose1
parco_h36m_randomsampling5_openpose1
parco_h36m_randomsampling10_openpose1
parco_h36m_randomsampling20_openpose1
parco_h36m_randomsampling40_openpose1
parco_h36m_randomsampling1_CPN
parco_h36m_randomsampling5_CPN
parco_h36m_randomsampling10_CPN
parco_h36m_randomsampling20_CPN
parco_h36m_randomsampling40_CPN

echo "Done"
