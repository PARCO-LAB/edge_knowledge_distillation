# How to run

`example_generation.sh` has a full pipeline example for a random sampling
with a window of 1500 and chunk of 128 with 4 actions. Adjust code/parameters
accordingly as needed.

IMPORTANT NOTE:
need to create folder in nas: MAEVE/KnowledgeDistillation/continual/<name>
and edit the blueprint.json

IMPORTANT NOTE PARCO04
Torch is installed system-wide. Use the line below to use it
python3 -m venv  --system-site-packages .venv

TODO: fix documentation for the next part

## Order

Everything has an order. The `h36m_to_order.py` generate this order.
For each image it is assigned a number, from 0 to N-1, where N is the total number of images.
This is needed to eventually change order action without changing the code.
Actually, using a lexycographic one (i.e.: the one use up to now).

## Sampling

`generate_random_sampling.py`

This file generates the corresponding sample with parameters
as window size, chunk size, window offset. It may takes the order for sample-free
metrics as Oracle, confidence-rate, etc.
It generates N files, where N is the chunk numbers (total / window), each one with the list of the
corresponding images.

`python generate_random_sampling.py --output-folder folder ...`

## Generate file for the train

`h36m_to_coco_indexing.py`

Given a file with index and an order, it generates the COCO file annotations.

## Generate experiments

`blueprint.py`

This file use a template JSON and generate the experiments correcting initial and final state_dict

## Run experiments
`../trt_pose/continal_train_with_indexing.py experiments/<folder_with_chunks>/` 

The execution will execute all the *.json file inside the folder in lexicographically fashion