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

FRAME_ID_LIMIT=700

# SRC_DATASET="/home/shared/nas/KnowledgeDistillation/h36m"
SRC_DATASET="/home/accounts/personale/ldgsfn95/nas/KnowledgeDistillation/h36m"

# DST_DATASET="/home/shared/dataset/h36m"
# DST_DATASET="/home/shared/nas/KnowledgeDistillation/mini_h36m"
DST_DATASET="/home/accounts/personale/ldgsfn95/KnowledgeDistillation/KnowledgeDistillation/h36m"

copy() {
    for (( i=0; i<$FRAME_ID_LIMIT; i++ )); do
        cp $1/$i.png $2
    done
}

# Copy images
# for cam in ${CAMERAS[*]}; do
#     for sub in ${SUBJECTS[*]}; do
#         echo "CAMERA ${cam} - SUBJECT ${sub}"
#         a_i=0
#         for action in $(ls -d ${SRC_DATASET}/${sub}/${cam}/*/); do
#             echo "${action} -> ${DST_DATASET}/${sub}/${cam}/$(echo ${action} | rev | cut -d/ -f2 | rev)"
#             mkdir -p ${DST_DATASET}/${sub}/${cam}/$(echo ${action} | rev | cut -d/ -f2 | rev)
#             copy ${action} ${DST_DATASET}/${sub}/${cam}/$(echo ${action} | rev | cut -d/ -f2 | rev) &
#             pids[${a_i}]=$!
#             a_i=$(($a_i + 1))
#         done
#         # wait for all pids
#         for pid in ${pids[*]}; do
#             wait $pid
#         done
#     done
# done

# # Remove folders
# for cam in ${CAMERAS[*]}; do
#     for sub in ${SUBJECTS[*]}; do
#         echo "CAMERA ${cam} - SUBJECT ${sub}"
#         rm -rf ${DST_DATASET}/${sub}/vicon ${DST_DATASET}/${sub}/trtpose* ${DST_DATASET}/${sub}/CPN ${DST_DATASET}/${sub}/openpose*
#     done
# done

# # Copy Vicon folder
# for cam in ${CAMERAS[*]}; do
#     for sub in ${SUBJECTS[*]}; do
#         echo "CAMERA ${cam} - SUBJECT ${sub}"
#         mkdir -p ${DST_DATASET}/${sub}/vicon
#         cp -r ${SRC_DATASET}/${sub}/vicon ${DST_DATASET}/${sub}
#     done
# done

# # Copy OpenPose folder
# for cam in ${CAMERAS[*]}; do
#     for sub in ${SUBJECTS[*]}; do
#         echo "CAMERA ${cam} - SUBJECT ${sub}"
#         mkdir -p ${DST_DATASET}/${sub}/openpose
#         mkdir -p ${DST_DATASET}/${sub}/openpose1
#         cp -r ${SRC_DATASET}/${sub}/openpose ${DST_DATASET}/${sub}
#         cp -r ${SRC_DATASET}/${sub}/openpose1 ${DST_DATASET}/${sub}
#     done
# done

# # Copy CPN folder
# for cam in ${CAMERAS[*]}; do
#     for sub in ${SUBJECTS[*]}; do
#         echo "CAMERA ${cam} - SUBJECT ${sub}"
#         mkdir -p ${DST_DATASET}/${sub}/CPN
#         cp -r ${SRC_DATASET}/${sub}/CPN ${DST_DATASET}/${sub}
#     done
# done

# # Copy trtpose_PARCO folder
# for cam in ${CAMERAS[*]}; do
#     for sub in ${SUBJECTS[*]}; do
#         echo "CAMERA ${cam} - SUBJECT ${sub}"
#         mkdir -p ${DST_DATASET}/${sub}/trtpose_PARCO
#         cp -r ${SRC_DATASET}/${sub}/trtpose_PARCO ${DST_DATASET}/${sub}
#     done
# done

cp ${SRC_DATASET}/../models/* ${DST_DATASET}/../models
cp ${SRC_DATASET}/../annotations/*.json ${DST_DATASET}/../annotations
cp ${SRC_DATASET}/../results/* ${DST_DATASET}/../results
cp ${SRC_DATASET}/../validation/* ${DST_DATASET}/../validation

echo "Done"
