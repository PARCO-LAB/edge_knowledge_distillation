#!/bin/bash

# our root folder
mkdir annotations/

# generate order from subject 1 with little action set
python3 ./h36m_to_order.py \
    --subject S1 \
    --output annotations/order_small.csv \
    --actions "Directions" "Discussion 1" "Phoning" "Smoking"

# perform the sampling using that data size. Only random supported!
mkdir -p annotations/continual_random_30s_128_32_vicon/train_idx
mkdir -p annotations/window-1500-chunk-128/test_idx
# Inside a window of 1500 frame, select 128 random sampling, and do
# that for all the frames advancing a 
python3 ./generate_random_sampling.py \
    --total-size $(cat annotations/order_small.csv | wc -l) \
    --window-size 1500 \
    --chunk-size 128 \
    --output-folder-train annotations/continual_random_30s_128_32_vicon/train_idx \
    --output-folder-test annotations/window-1500-chunk-128/test_idx

echo "Generating train"
python3 ./h36m_to_coco_indexing.py \
    --order annotations/order_small.csv \
    --index annotations/continual_random_30s_128_32_vicon/train_idx/ \
    --output annotations/continual_random_30s_128_32_vicon/train/ \
    --subject S1 \
    --source-folder /home/shared/nas//KnowledgeDistillation/h36m/ \
    --teacher vicon

echo "Generating test"
python3 ./h36m_to_coco_indexing.py \
    --order annotations/order_small.csv \
    --index annotations/window-1500-chunk-128/test_idx/ \
    --output annotations/window-1500-chunk-128/test/ \
    --subject S1 \
    --source-folder /home/shared/nas//KnowledgeDistillation/h36m/ \
    --teacher vicon


# full dataset
mkdir -p annotations/full_dataset/
cat window-1500-chunk-128/test_idx/*.txt > full_dataset/chunk-000.txt
echo "Generating test for the full dataset"
python3 ./h36m_to_coco_indexing.py \
    --order annotations/order_small.csv \
    --index annotations/full_dataset/ \
    --output annotations/full_dataset/ \
    --subject S1 \
    --source-folder /home/shared/nas//KnowledgeDistillation/h36m/ \
    --teacher vicon

mkdir -p experiments/full_dataset
python3 blueprint.py \
    --input blueprint.json \
    --output $(pwd)/experiments/full_dataset/ \
    --test-annotations $(pwd)/annotations/full_dataset/ \
    --checkpoint-dir /home/shared/nas/KnowledgeDistillation/continual/continual_random_30s_128_32_vicon \
    --chunk-size 128 \
    --batch-size 32 \
    --learning-rate 1e-5 \
    --epochs 10 \
    --num-chunks 1

##### Important part
# generate all the JSON file for training/testing offline, but with the previous step
# it is simulating the online!
mkdir -p experiments/continual_random_30s_128_32_vicon
python3 blueprint.py \
    --input blueprint.json \
    --output $(pwd)/experiments/continual_random_30s_128_32_vicon/ \
    --train-annotations $(pwd)/annotations/continual_random_30s_128_32_vicon/train/ \
    --test-annotations $(pwd)/annotations/window-1500-chunk-128/test/ \
    --checkpoint-dir /home/shared/nas/KnowledgeDistillation/continual/continual_random_30s_128_32_vicon \
    --chunk-size 128 \
    --batch-size 32 \
    --learning-rate 1e-5 \
    --epochs 10 \
    --num-chunks $(ls $(pwd)/annotations/window-1500-chunk-128/test/ | wc -l)


# finally, launch the continual learning!
# first, the train of everything (it will process folder/*.json)
python3 ../trt_pose/trt_pose/continual_train_with_indexing.py \
     experiments/continual_random_30s_128_32_vicon/

# then, evaluate everything!
# python3 ../trt_pose/trt_pose/continual_val_indexing.py \
#   experiments/continual_random_30s_128_32_vicon/
python3 ../trt_pose/trt_pose/continual_val_with_indexing.py \
   experiments/continual_random_30s_128_32_vicon/ experiments/continual_random_30s_128_32_vicon