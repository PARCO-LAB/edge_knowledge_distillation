#!/bin/bash
# continual_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B
# continual_h36m_vicon_nohead_densenet121_baseline_att_256x256_B"
#

tasks=( 
    "continual_fixedmax_confidence1_h36m_vicon_nohead_densenet121_baseline_att_256x256_B" 
    "continual_fixedmax_confidence20_h36m_vicon_nohead_densenet121_baseline_att_256x256_B" 
    "continual_fixedmax_confidence1_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B" 
    "continual_fixedmax_confidence20_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B" 
)

for task in "${tasks[@]}"
do
    bash continual.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/$task.json
    cp -r /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/$task.json.checkpoints /home/shared/nas/KnowledgeDistillation/experiments/
    rm -rf /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/$task.json.checkpoints
    cp /home/shared/nas/KnowledgeDistillation/experiments/$task.json.checkpoints/resval_dist.csv /home/shared/nas/KnowledgeDistillation/results/$task.csv
done
