#!/usr/bin/bash

ln -s /home/shared/nas//KnowledgeDistillation/h36m/ dataset
ln -s /home/shared/nas/KnowledgeDistillation/continual checkpoints
ln -s /home/shared/befine/edge_knowledge_distillation/submodule/lib_maeve_py/maeve/nn/trtpose/models/densenet121_baseline_att_256x256_B_epoch_160_parco.pth .

# ln -s /home/saldegheri/nas/MAEVE/KnowledgeDistillation/h36m dataset
# ln -s /home/saldegheri/nas/MAEVE/KnowledgeDistillation/continual checkpoints