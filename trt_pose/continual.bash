#!/bin/bash

task=$1
chunk_amount=$(python3 trt_pose/continual_chunk_amount.py $task)
echo "chunk_amount: $chunk_amount"

echo "================ CHUNK -1 =============================="
python3 trt_pose/continual_val.py -i -1 $task
# DNN=openpose python3 trt_pose/continual_val.py -i -1 $task
# python3 trt_pose/continual_val.py --baseline -i -1 $task
chunk_amount=4
for ((i=0; i<chunk_amount; i++)); do
    echo "================ CHUNK $i =============================="
    python3 trt_pose/continual_train.py -i $i $task
    # python3 trt_pose/continual_test.py -i $i $task
    python3 trt_pose/continual_val.py -i $i $task
done
