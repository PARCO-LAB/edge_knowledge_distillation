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
            python3 draw_parcopose_from_folder.py -f ${action} -n trtpose
            python3 draw_parcopose_from_folder.py -f ${action} -n parcopose
            ffmpeg -i ${action%/}_trtpose.mp4 -i ${action%/}_parcopose.mp4 -filter_complex "[0:v]crop=in_h:in_h:in_w/4:0[v0];[1:v]crop=in_h:in_h:in_w/4:0[v1];[v0][v1]hstack=inputs=2" ${action}_comparison.mp4
        done
    done
done

echo "Done"
