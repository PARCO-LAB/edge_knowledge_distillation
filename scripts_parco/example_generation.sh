#!/bin/bash

WINDOW_SIZE=1500
CHUNK_SIZE=320
BATCH_SIZE=32
SAMPLING=uniform

TEACHER=openpose1
EVALUATOR=vicon


TRAIN_NAME=${SAMPLING}_${WINDOW_SIZE}_${CHUNK_SIZE}_${BATCH_SIZE}_${TEACHER}
# TEST_NAME=${SAMPLING}_${WINDOW_SIZE}_${CHUNK_SIZE}_${BATCH_SIZE}_${EVALUATOR}
TEST_NAME=window_${WINDOW_SIZE}_${EVALUATOR}


# our root folder
mkdir annotations/
mkdir adeguate
# generate order from subject 1 with little action set
python3 ./h36m_to_order.py \
    --subject S1 \
    --teacher vicon \
    --percentage 0.8 \
    --output-gt adeguate/vicon.csv \
    --output-adeguate-folder adeguate \
    --adeguate openpose1 trtpose_PARCO \
    --output annotations/order_small.csv \
    --actions "Directions 1" "Directions" "Discussion 1" "Discussion" "Eating 2" "Eating" "Greeting 1" "Greeting" "Phoning 1" "Phoning" "Posing 1" "Posing" "Purchases 1" "Purchases" "Sitting 1" "Sitting 2" "SittingDown 2" "SittingDown" "Smoking 1" "Smoking" "TakingPhoto 1" "TakingPhoto" "Waiting 1" "Waiting" "Walking 1" "Walking" "WalkingDog 1" "WalkingDog" "WalkTogether 1" "WalkTogether"
#     --actions "Directions 1" "Directions"
    
    # "Discussion 1" "Discussion" "Eating 2" "Eating" "Greeting 1" "Greeting" "Phoning 1" "Phoning" "Posing 1" "Posing" "Purchases 1" "Purchases" "Sitting 1" "Sitting 2" "SittingDown 2" "SittingDown" "Smoking 1" "Smoking" "TakingPhoto 1" "TakingPhoto" "Waiting 1" "Waiting" "Walking 1" "Walking" "WalkingDog 1" "WalkingDog" "WalkTogether 1" "WalkTogether"
#    --actions "Directions" "Phoning"
    # "Smoking" "Discussion 1"

# perform the sampling using that data size. Only random supported!
mkdir -p annotations/${TRAIN_NAME}/train_idx
mkdir -p annotations/${TEST_NAME}/test_idx
# Inside a window of 1500 frame, select 128 random sampling, and do
# that for all the frames advancing a 
python3 ./generate_sampling.py \
    --total-size $(cat annotations/order_small.csv | wc -l) \
    --window-size ${WINDOW_SIZE} \
    --chunk-size ${CHUNK_SIZE} \
    --sampling ${SAMPLING} \
    --output-folder-train annotations/${TRAIN_NAME}/train_idx \
    --output-folder-test  annotations/${TEST_NAME}/test_idx

echo "Generating train"
mkdir -p annotations/${TRAIN_NAME}/train
python3 ./h36m_to_coco_indexing.py \
    --order annotations/order_small.csv \
    --index  annotations/${TRAIN_NAME}/train_idx/ \
    --output annotations/${TRAIN_NAME}/train/ \
    --subject S1 \
    --source-folder dataset/ \
    --teacher ${TEACHER}

echo "Generating test"
mkdir -p annotations/${TEST_NAME}/test
python3 ./h36m_to_coco_indexing.py \
    --order annotations/order_small.csv \
    --index  annotations/${TEST_NAME}/test_idx/ \
    --output annotations/${TEST_NAME}/test/ \
    --subject S1 \
    --source-folder dataset/ \
    --teacher vicon

# full dataset
mkdir -p annotations/full_dataset_${WINDOW_SIZE}/
cat annotations/${TEST_NAME}/test_idx/*.txt > annotations/full_dataset_${WINDOW_SIZE}/chunk-000.txt
echo "Generating test for the full dataset"
python3 ./h36m_to_coco_indexing.py \
    --order annotations/order_small.csv \
    --index  annotations/full_dataset_${WINDOW_SIZE}/ \
    --output annotations/full_dataset_${WINDOW_SIZE}/ \
    --subject S1 \
    --source-folder dataset/ \
    --teacher vicon

# do the blueprint for the baseline (only testing)
mkdir -p experiments/full_dataset_${WINDOW_SIZE}
python3 blueprint.py \
    --input blueprint.json \
    --output $(pwd)/experiments/full_dataset_${WINDOW_SIZE}/ \
    --test-annotations $(pwd)/annotations/full_dataset_${WINDOW_SIZE}/ \
    --checkpoint-dir checkpoints/${TRAIN_NAME} \
    --num-chunks 1

##### Important part
# generate all the JSON file for training/testing offline, but with the previous step
# it is simulating the online!
mkdir -p experiments/${TRAIN_NAME}
python3 blueprint.py \
    --input blueprint.json \
    --output $(pwd)/experiments/${TRAIN_NAME}/ \
    --train-annotations $(pwd)/annotations/${TRAIN_NAME}/train/ \
    --test-annotations $(pwd)/annotations/${TEST_NAME}/test/ \
    --checkpoint-dir checkpoints/${TRAIN_NAME} \
    --chunk-size ${CHUNK_SIZE} \
    --batch-size ${BATCH_SIZE} \
    --learning-rate 1e-3 \
    --epochs 10 \
    --num-chunks $(ls $(pwd)/annotations/${TEST_NAME}/test/*.json | wc -l)

# finally, launch the continual learning!
# first, the train of everything (it will process folder/*.json)
python3 ../trt_pose/trt_pose/continual_train_with_indexing.py \
     experiments/${TRAIN_NAME}/

# then, evaluate everything!
# python3 ../trt_pose/trt_pose/continual_val_indexing.py \
#   experiments/continual_random_30s_128_32_vicon/
python3 ../trt_pose/trt_pose/continual_val_with_indexing.py \
   experiments/${TRAIN_NAME}/ \
   experiments/${TRAIN_NAME} \
   annotations/order_small.csv

python3 combine_csv.py \
    experiments/${TRAIN_NAME}/ \
    experiments/${TRAIN_NAME}.csv