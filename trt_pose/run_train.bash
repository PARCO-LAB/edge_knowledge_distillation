#!/bin/bash

# module load Python/3.9.16
# module load cuda11.1
# module restore pytorch

# source ~/.venv_39/bin/activate

python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mean_confidence1_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mean_confidence5_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mean_confidence10_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mean_confidence20_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mean_confidence40_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/confidence1_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/confidence5_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/confidence10_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/confidence20_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/confidence40_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mpjpe1_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mpjpe5_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mpjpe10_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mpjpe20_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mpjpe40_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/uniform1_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/uniform5_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/uniform10_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/uniform20_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/uniform40_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/random1_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/random5_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/random10_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/random20_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/random40_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/action1_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/action5_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/action10_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/action20_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/action40_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/parco1_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/parco5_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/parco10_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/parco20_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/parco40_h36m_vicon_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mean_confidence1_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mean_confidence5_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mean_confidence10_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mean_confidence20_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mean_confidence40_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/confidence1_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/confidence5_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/confidence10_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/confidence20_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/confidence40_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mpjpe1_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mpjpe5_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mpjpe10_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mpjpe20_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mpjpe40_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/uniform1_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/uniform5_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/uniform10_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/uniform20_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/uniform40_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/random1_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/random5_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/random10_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/random20_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/random40_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/action1_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/action5_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/action10_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/action20_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/action40_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/parco1_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/parco5_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/parco10_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/parco20_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/parco40_h36m_openpose1_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mean_confidence1_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mean_confidence5_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mean_confidence10_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mean_confidence20_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mean_confidence40_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/confidence1_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/confidence5_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/confidence10_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/confidence20_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/confidence40_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mpjpe1_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mpjpe5_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mpjpe10_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mpjpe20_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/mpjpe40_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/uniform1_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/uniform5_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/uniform10_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/uniform20_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/uniform40_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/random1_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/random5_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/random10_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/random20_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/random40_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/action1_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/action5_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/action10_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/action20_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/action40_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/parco1_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/parco5_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/parco10_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/parco20_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json
python3 trt_pose/train.py /home/shared/befine/edge_knowledge_distillation/trt_pose/tasks/human_pose/experiments/parco40_h36m_CPN_nohead_densenet121_baseline_att_256x256_B.json

