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

# REALLY slow generating, since EACH time it generates a single chunk whilst 
# look for everything. But at least it's working!
mkdir -p annotations/continual_random_30s_128_32_vicon/train
for train in $(ls annotations/continual_random_30s_128_32_vicon/train_idx/); do
    echo "Generating train "$train
    python3 ./h36m_to_coco_indexing.py \
        --order annotations/order_small.csv \
        --index annotations/continual_random_30s_128_32_vicon/train_idx/$train \
        --output annotations/continual_random_30s_128_32_vicon/train/$(basename -s .txt $train).json \
        --subject S1
done

mkdir -p annotations/window-1500-chunk-128/test
for test in $(ls annotations/window-1500-chunk-128/test_idx/); do
    echo "Generating test "$test
    python3 ./h36m_to_coco_indexing.py \
        --order annotations/order_small.csv \
        --index annotations/window-1500-chunk-128/test_idx/$test \
        --output annotations/window-1500-chunk-128/test/$(basename -s .txt $test).json \
        --subject S1
done


##### Important part
# generate all the JSON file for training/testing offline, but with the previous step
# it is simulating the online!
mkdir -p experiments/continual_random_30s_128_32_vicon
python3 blueprint.py \
    --input blueprint.json \
    --output $(pwd)/experiments/continual_random_30s_128_32_vicon/ \
    --train-annotations $(pwd)/annotations/continual_random_30s_128_32_vicon/train/ \
    --test-annotations $(pwd)/annotations/window-1500-chunk-128/test/ \
    --chunk-size 128 \
    --batch-size 32 \
    --epochs 10 \
    --num-chunks $(ls $(pwd)/annotations/window-1500-chunk-128/test/ | wc -l)


# finally, launch the continual learning!
# first, the train of everything (it will process folder/*.json)
python3 ../trt_pose/trt_pose/continual_train_with_indexing.py \
     experiments/continual_random_30s_128_32_vicon/

# then, evaluate everything!
# python3 ../trt_pose/trt_pose/continual_val_indexing.py \
#   experiments/continual_random_30s_128_32_vicon/