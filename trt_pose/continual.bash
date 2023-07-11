#!/bin/bash

task=$1
chunk_amount=$(python3 trt_pose/continual_chunk_amount.py $task)
echo "chunk_amount: $chunk_amount"

for ((i=0; i<chunk_amount; i++)); do
    echo "================ CHUNK $i =============================="
    python3 trt_pose/continual_train.py -i $i $task
    # python3 trt_pose/continual_test.py -i $i $task
    python3 trt_pose/continual_val.py -i $i $task
    python3 trt_pose/continual_val.py --baseline -i $i $task
done