#!/bin/bash

SUBJECTS=(
    "S1"
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

FRAME_ID_LIMIT=700

DST_DATASET="/home/shared/dataset/h36m"
# DST_DATASET="/home/shared/nas/KnowledgeDistillation/mini_h36m"

copy() {
    for (( i=0; i<$FRAME_ID_LIMIT; i++ )); do
        cp $1/$i.png $2
    done
}

for cam in ${CAMERAS[*]}; do
    for sub in ${SUBJECTS[*]}; do
        echo "CAMERA ${cam} - SUBJECT ${sub}"
        a_i=0
        for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${sub}/${cam}/*/); do
            echo "${action} -> ${DST_DATASET}/${sub}/${cam}/$(echo ${action} | rev | cut -d/ -f2 | rev)"
            mkdir -p ${DST_DATASET}/${sub}/${cam}/$(echo ${action} | rev | cut -d/ -f2 | rev)
            copy ${action} ${DST_DATASET}/${sub}/${cam}/$(echo ${action} | rev | cut -d/ -f2 | rev) &
            pids[${a_i}]=$!
            a_i=$(($a_i + 1))
        done
        # wait for all pids
        for pid in ${pids[*]}; do
            wait $pid
        done
    done
done

# Copy Vicon folder
for cam in ${CAMERAS[*]}; do
    for sub in ${SUBJECTS[*]}; do
        echo "CAMERA ${cam} - SUBJECT ${sub}"
        mkdir -p ${DST_DATASET}/${sub}/vicon
        cp -r /home/shared/nas/KnowledgeDistillation/h36m/${sub}/vicon ${DST_DATASET}/${sub}
    done
done

# Copy OpenPose folder
for cam in ${CAMERAS[*]}; do
    for sub in ${SUBJECTS[*]}; do
        echo "CAMERA ${cam} - SUBJECT ${sub}"
        mkdir -p ${DST_DATASET}/${sub}/openpose
        cp -r /home/shared/nas/KnowledgeDistillation/h36m/${sub}/openpose ${DST_DATASET}/${sub}
    done
done

echo "Done"
