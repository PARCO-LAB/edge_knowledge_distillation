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

listdir() {
    for cam in ${CAMERAS[*]}; do
        for sub in ${SUBJECTS[*]}; do
            echo "CAMERA ${cam} - SUBJECT ${sub}"
            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
                echo ${action}
            done
        done
    done
}

listdir

echo "Done"
