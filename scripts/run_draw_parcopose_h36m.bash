#!/bin/bash

SUBJECTS=(
    # "S1"
    # "S5"
    # "S6"
    # "S7"
    # "S8"
    "S9"
    # "S11"
)

CAMERAS=(
    "55011271"
)

for cam in ${CAMERAS[*]}; do
    for sub in ${SUBJECTS[*]}; do
        echo "CAMERA ${cam} - SUBJECT ${sub}"
        for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
            echo ${action%/}
            python3 draw_parcopose_from_folder.py -f ${action%/} -n trtpose
            DNN=openpose python3 draw_parcopose_from_folder.py -f ${action%/} -n openpose
            python3 draw_parcopose_from_folder.py -f ${action%/} -n parcopose
            python3 draw_parcopose_from_folder.py -f ${action%/} -n parcopose_h36m
            python3 draw_parcopose_from_folder.py -f ${action%/} -n parcopose_h36m_openpose
            python3 draw_parcopose_from_folder.py -f ${action%/} -n parcopose_h36m_CPN
            python3 concat_video.py -n ${action%/}
        done
    done
done

echo "Done"
