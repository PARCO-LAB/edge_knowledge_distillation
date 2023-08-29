#!/bin/bash

task=$1
chunk_amount=$(python3 trt_pose/continual_chunk_amount.py $task)
echo "chunk_amount: $chunk_amount"

echo "================ CHUNK -1 =============================="
time python3 trt_pose/continual_val_smart.py -i -1 $task --selector svd
# DNN=openpose python3 trt_pose/continual_val.py -i -1 $task
# python3 trt_pose/continual_val.py --baseline -i -1 $task
#chunk_amount=4
for ((i=0; i<chunk_amount; i++)); do
    echo "================ CHUNK $i =============================="
    time python3 trt_pose/continual_train_smart.py -i $i $task --selector svd
    # python3 trt_pose/continual_test.py -i $i $task
    time python3 trt_pose/continual_val_smart.py -i $i $task --selector svd
done

