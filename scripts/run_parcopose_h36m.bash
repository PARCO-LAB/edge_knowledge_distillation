#!/bin/bash

SUBJECTS=(
    "S1"
    "S5"
    "S6"
    "S7"
    "S8"
    "S9"
    "S11"
)

CAMERAS=(
    "55011271"
)

for cam in ${CAMERAS[*]}; do
    for sub in ${SUBJECTS[*]}; do
        echo "CAMERA ${cam} - SUBJECT ${sub}"
        for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*); do
            echo ${action}
            python3 parcopose_from_folder.py -f ${action}
        done
        mv /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*.csv /home/shared/nas/KnowledgeDistillation/h36m/${sub}/trtpose_PARCO/
    done
done

echo "Done"
