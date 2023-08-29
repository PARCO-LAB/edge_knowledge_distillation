#!/bin/bash

# module load Python/3.9.16
# module load cuda11.1
module restore pytorch

source ~/.venv_39/bin/activate

bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmax_confidence1_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmax_confidence5_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmax_confidence10_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmax_confidence20_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmax_confidence40_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmean_confidence1_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmean_confidence5_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmean_confidence10_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmean_confidence20_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmean_confidence40_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedconfidence1_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedconfidence5_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedconfidence10_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedconfidence20_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedconfidence40_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmpjpe1_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmpjpe5_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmpjpe10_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmpjpe20_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmpjpe40_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mean_confidence1_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mean_confidence5_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mean_confidence10_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mean_confidence20_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mean_confidence40_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_max_confidence1_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_max_confidence5_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_max_confidence10_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_max_confidence20_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_max_confidence40_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_confidence1_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_confidence5_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_confidence10_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_confidence20_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_confidence40_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mpjpe1_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mpjpe5_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mpjpe10_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mpjpe20_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mpjpe40_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_uniform1_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_uniform5_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_uniform10_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_uniform20_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_uniform40_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedrandom1_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedrandom5_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedrandom10_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedrandom20_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedrandom40_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_random1_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_random5_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_random10_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_random20_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_random40_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_action1_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_action5_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_action10_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_action20_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_action40_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_parco1_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_parco5_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_parco10_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_parco20_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_parco40_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmax_confidence1_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmax_confidence5_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmax_confidence10_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmax_confidence20_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmax_confidence40_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmean_confidence1_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmean_confidence5_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmean_confidence10_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmean_confidence20_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmean_confidence40_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedconfidence1_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedconfidence5_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedconfidence10_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedconfidence20_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedconfidence40_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmpjpe1_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmpjpe5_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmpjpe10_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmpjpe20_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmpjpe40_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mean_confidence1_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mean_confidence5_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mean_confidence10_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mean_confidence20_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mean_confidence40_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_max_confidence1_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_max_confidence5_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_max_confidence10_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_max_confidence20_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_max_confidence40_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_confidence1_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_confidence5_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_confidence10_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_confidence20_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_confidence40_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mpjpe1_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mpjpe5_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mpjpe10_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mpjpe20_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mpjpe40_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_uniform1_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_uniform5_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_uniform10_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_uniform20_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_uniform40_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedrandom1_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedrandom5_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedrandom10_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedrandom20_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedrandom40_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_random1_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_random5_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_random10_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_random20_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_random40_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_action1_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_action5_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_action10_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_action20_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_action40_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_parco1_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_parco5_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_parco10_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_parco20_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_parco40_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmax_confidence1_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmax_confidence5_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmax_confidence10_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmax_confidence20_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmax_confidence40_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmean_confidence1_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmean_confidence5_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmean_confidence10_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmean_confidence20_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmean_confidence40_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedconfidence1_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedconfidence5_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedconfidence10_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedconfidence20_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedconfidence40_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmpjpe1_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmpjpe5_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmpjpe10_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmpjpe20_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedmpjpe40_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mean_confidence1_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mean_confidence5_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mean_confidence10_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mean_confidence20_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mean_confidence40_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_max_confidence1_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_max_confidence5_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_max_confidence10_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_max_confidence20_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_max_confidence40_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_confidence1_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_confidence5_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_confidence10_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_confidence20_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_confidence40_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mpjpe1_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mpjpe5_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mpjpe10_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mpjpe20_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_mpjpe40_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_uniform1_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_uniform5_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_uniform10_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_uniform20_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_uniform40_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedrandom1_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedrandom5_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedrandom10_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedrandom20_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_fixedrandom40_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_random1_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_random5_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_random10_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_random20_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_random40_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_action1_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_action5_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_action10_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_action20_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_action40_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_parco1_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_parco5_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_parco10_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_parco20_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
bash continual_smart.bash /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/continual_parco40_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json

