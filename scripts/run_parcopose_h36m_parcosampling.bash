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

parco_h36m_parcosampling1_vicon() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling1_vicon/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_parcosampling1_vicon -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling1_vicon/
            done
        done
    done
}

parco_h36m_parcosampling5_vicon() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling5_vicon/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_parcosampling5_vicon -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling5_vicon/
            done
        done
    done
}

parco_h36m_parcosampling10_vicon() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling10_vicon/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_parcosampling10_vicon -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling10_vicon/
            done
        done
    done
}

parco_h36m_parcosampling20_vicon() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling20_vicon/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_parcosampling20_vicon -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling20_vicon/
            done
        done
    done
}

parco_h36m_parcosampling40_vicon() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling40_vicon/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_parcosampling40_vicon -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling40_vicon/
            done
        done
    done
}

parco_h36m_parcosampling1_openpose1() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling1_openpose1/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_parcosampling1_openpose1 -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling1_openpose1/
            done
        done
    done
}

parco_h36m_parcosampling5_openpose1() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling5_openpose1/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_parcosampling5_openpose1 -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling5_openpose1/
            done
        done
    done
}

parco_h36m_parcosampling10_openpose1() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling10_openpose1/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_parcosampling10_openpose1 -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling10_openpose1/
            done
        done
    done
}

parco_h36m_parcosampling20_openpose1() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling20_openpose1/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_parcosampling20_openpose1 -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling20_openpose1/
            done
        done
    done
}

parco_h36m_parcosampling40_openpose1() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling40_openpose1/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_parcosampling40_openpose1 -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling40_openpose1/
            done
        done
    done
}

parco_h36m_parcosampling1_CPN() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling1_CPN/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_parcosampling1_CPN -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling1_CPN/
            done
        done
    done
}

parco_h36m_parcosampling5_CPN() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling5_CPN/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_parcosampling5_CPN -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling5_CPN/
            done
        done
    done
}

parco_h36m_parcosampling10_CPN() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling10_CPN/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_parcosampling10_CPN -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling10_CPN/
            done
        done
    done
}

parco_h36m_parcosampling20_CPN() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling20_CPN/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_parcosampling20_CPN -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling20_CPN/
            done
        done
    done
}

parco_h36m_parcosampling40_CPN() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling40_CPN/
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
                python3 parcopose_from_folder.py -f ${action} -n parco_h36m_parcosampling40_CPN -o /home/shared/nas/KnowledgeDistillation/h36m/${sub}/parco_h36m_parcosampling40_CPN/
            done
        done
    done
}

parco_h36m_parcosampling1_vicon
parco_h36m_parcosampling5_vicon
parco_h36m_parcosampling10_vicon
parco_h36m_parcosampling20_vicon
parco_h36m_parcosampling40_vicon
parco_h36m_parcosampling1_openpose1
parco_h36m_parcosampling5_openpose1
parco_h36m_parcosampling10_openpose1
parco_h36m_parcosampling20_openpose1
parco_h36m_parcosampling40_openpose1
parco_h36m_parcosampling1_CPN
parco_h36m_parcosampling5_CPN
parco_h36m_parcosampling10_CPN
parco_h36m_parcosampling20_CPN
parco_h36m_parcosampling40_CPN

echo "Done"
