import argparse
import json
import os
import shutil
import glob

from h36m_to_coco import TEACHERS, SAMPLINGS, PERCENTAGES


REF_HUMAN_POSE = "human_pose_parco.json"
REF_TASK = "nohead_densenet121_baseline_att_256x256_B.json"
INITIAL_MODEL = "densenet121_baseline_att_256x256_B_epoch_160_parco.pth"

SHUFFLE = False
MAX_NUM_WORKERS = 4
MEMORY_BATCH_SIZE = 32
REAL_BATCH_SIZE = 64
EPOCHS = 10


CONTINUAL_LEARNING_SUBJECT = "S1"
CONTINUAL_NUM_WORKERS = 1
CONTINUAL_WINDOW = int(50 * 30) # 1500 [0.01: 15]
CONTINUAL_BATCH_SIZE = 32
CONTINUAL_EPOCHS = 10



def save_task_info(task_json, initial_state_dict, images_dir, annotation_dir, tasks_dir, train_file, test_file, name):
    task_json["model"]["initial_state_dict"] = initial_state_dict
    task_json["train_dataset"]["images_dir"] = images_dir
    task_json["train_dataset"]["annotations_file"] = os.path.join(annotation_dir, train_file)
    task_json["train_dataset"]["image_extension"] = "png"
    task_json["train_loader"]["batch_size"] = MEMORY_BATCH_SIZE
    task_json["train_loader"]["shuffle"] = SHUFFLE
    task_json["train_loader"]["num_workers"] = MAX_NUM_WORKERS
    task_json["test_dataset"]["images_dir"] = images_dir
    task_json["test_dataset"]["annotations_file"] = os.path.join(annotation_dir, test_file)
    task_json["test_dataset"]["image_extension"] = "png"
    task_json["test_loader"]["batch_size"] = MEMORY_BATCH_SIZE
    task_json["test_loader"]["shuffle"] = SHUFFLE
    task_json["test_loader"]["num_workers"] = MAX_NUM_WORKERS
    task_json["batch_size"] = REAL_BATCH_SIZE
    task_json["epochs"] = EPOCHS

    json_data = json.dumps(task_json, indent=4)

    # Write the JSON data to a file
    with open(os.path.join(tasks_dir, name), "w") as f:
        f.write(json_data)


def save_continual_task_info(task_json, initial_state_dict, images_dir, annotation_dir, tasks_dir, train_file, ref_train_file, test_file, name, perc=None):
    task_json["model"]["initial_state_dict"] = initial_state_dict
    task_json["train_dataset"]["images_dir"] = images_dir
    task_json["train_dataset"]["annotations_file"] = os.path.join(annotation_dir, train_file)
    task_json["train_dataset"]["image_extension"] = "png"
    task_json["train_loader"]["batch_size"] = CONTINUAL_WINDOW if perc is None else int(CONTINUAL_WINDOW * perc)
    task_json["train_loader"]["shuffle"] = SHUFFLE
    task_json["train_loader"]["num_workers"] = CONTINUAL_NUM_WORKERS
    task_json["test_dataset"]["images_dir"] = images_dir
    task_json["test_dataset"]["annotations_file"] = os.path.join(annotation_dir, test_file)
    task_json["test_dataset"]["image_extension"] = "png"
    task_json["test_loader"]["batch_size"] = CONTINUAL_WINDOW if perc is None else int(CONTINUAL_WINDOW * perc)
    task_json["test_loader"]["shuffle"] = SHUFFLE
    task_json["test_loader"]["num_workers"] = CONTINUAL_NUM_WORKERS
    task_json["ref_annotations_file"] = os.path.join(annotation_dir, ref_train_file)
    task_json["batch_size"] = CONTINUAL_BATCH_SIZE
    task_json["window"] = CONTINUAL_WINDOW
    task_json["epochs"] = CONTINUAL_EPOCHS
    task_json["ground_truth"] = {
        "folder": os.path.join(os.path.dirname(images_dir), "h36m"),
        "model": "vicon",
        "subjects": ["S9", "S11"],
        "cameras": ["55011271"],
    }

    json_data = json.dumps(task_json, indent=4)

    # Write the JSON data to a file
    with open(os.path.join(tasks_dir, name), "w") as f:
        f.write(json_data)


def gen_bash_script(list_of_tasks, name, is_continual):
    script =  "#!/bin/bash\n"
    script += "\n"
    script += "# module load Python/3.9.16\n"
    script += "# module load cuda11.1\n"
    script += "module restore pytorch\n"
    script += "\n"
    script += "source ~/.venv_39/bin/activate\n"
    script += "\n"

    if is_continual: 
        command = "bash continual.bash"
    else: 
        command = "python3 trt_pose/train.py"
    
    for e in list_of_tasks:
        script += "{} {}\n".format(command, e) 
    script += "\n"
    with open(name, "w") as file:
        file.write(script)

    print("Bash script {} generated successfully".format(name))


def create_human_pose(human_pose_json, human_pose_folder, name):
    # Write the JSON data to a file
    prefix = os.path.splitext(REF_HUMAN_POSE)[0]
    json_data = json.dumps(human_pose_json, indent=4)
    with open(os.path.join(human_pose_folder, "{}_{}.json".format(prefix, name)), "w") as f:
        f.write(json_data)


def gen_inference(repo_folder, teachers, samplings, percentages, name):
    script_folder = os.path.join(repo_folder, "scripts")
    script  = "#!/bin/bash\n"
    script += "\n"
    script += "SUBJECTS=(\n"
    script += "    \"S9\"\n"
    script += "    \"S11\"\n"
    script += "    \"S1\"\n"
    script += "    \"S5\"\n"
    script += "    \"S6\"\n"
    script += "    \"S7\"\n"
    script += "    \"S8\"\n"
    script += ")\n"
    script += "\n"
    script += "CAMERAS=(\n"
    script += "    \"55011271\"\n"
    script += ")\n"
    script += "\n"

    template_run = ""
    template_run += "{0}() {{\n"
    template_run += "    i=0\n"
    template_run += "    for cam in ${{CAMERAS[*]}}; do\n"
    template_run += "        for sub in ${{SUBJECTS[*]}}; do\n"
    template_run += "            echo \"CAMERA ${{cam}} - SUBJECT ${{sub}}\"\n"
    template_run += "            mkdir -p /home/shared/nas/KnowledgeDistillation/h36m/${{sub}}/{0}/\n"
    template_run += "            for action in $(ls -d /home/shared/nas/KnowledgeDistillation/h36m/${{sub}}/${{cam}}/*/); do\n"
    template_run += "                echo ${{action}}\n"
    template_run += "                TESTS[${{i}}]=${{action}}\n"
    template_run += "                i=$(($i + 1))\n"
    template_run += "            done\n"
    template_run += "        done\n"
    template_run += "    done\n"
    template_run += "    {1}python3 parcopose_from_folder.py -f ${{action}} -n {0} -o /home/shared/nas/KnowledgeDistillation/h36m/${{sub}}/{0}/\n"
    template_run += "    unset TESTS\n"
    template_run += "}}\n"
    template_run += "\n"


    if type(teachers) is str:
        if teachers == "openpose" or teachers == "openpose1":
            script += template_run.format(teachers, "DNN=openpose ")
        else: 
            script += template_run.format(teachers, "")
        script += "{}\n".format(teachers)
    elif samplings is None or percentages is None: 
        for teacher in teachers: 
            script += template_run.format("parco_h36m_{}".format(teacher), "")
        
        for teacher in teachers: 
            script += "parco_h36m_{}\n".format(teacher)
        
    else: 
        for teacher in teachers: 
            for sampling in samplings: 
                for perc in percentages:
                    script += template_run.format("parco_h36m_{}sampling{}_{}".format(sampling, perc, teacher), "")
        
        for teacher in teachers: 
            for sampling in samplings: 
                for perc in percentages:
                    script += "parco_h36m_{}sampling{}_{}\n".format(sampling, perc, teacher)
    
    script += "\necho \"Done\"\n"

    with open(os.path.join(script_folder, name), "w") as file:
        file.write(script)


def copy_checkpoints(teachers, samplings, percentages, tasks_dir, data_folder, human_pose_folder):
    model_dir = os.path.join(data_folder, "models")
    model_prefix = os.path.splitext(INITIAL_MODEL)[0]

    for teacher in teachers: 
        for sampling in samplings: 
            for perc in percentages:
                perc_str = "{}".format(int(perc * 100))
                task = "{}{}_h36m_{}_{}".format(sampling, perc_str, teacher, REF_TASK)
                task_dir = os.path.join(tasks_dir, "{}.checkpoints".format(task))
                model_file = os.path.join(task_dir, "epoch_0.pth")

                target_model_name = "{}_h36m_{}sampling{}_{}.pth".format(model_prefix, sampling, perc_str, teacher)
                target_model_path = os.path.join(model_dir, target_model_name)

                if os.path.exists(model_file):
                    shutil.copy(model_file, target_model_path)
    
    # Copy all files from model dir
    model_files = glob.glob(os.path.join(model_dir, "*.pth"))
    for model_file in model_files:
        model_basename = os.path.basename(model_file)
        target_model_path = os.path.join(human_pose_folder, model_basename)
        shutil.copy(model_file, target_model_path)




def main(repo_folder, data_folder):
    
    initial_state_dict = os.path.join(
        repo_folder, "submodule", "lib_maeve_py", "maeve", "nn", "trtpose", "models", INITIAL_MODEL)
    human_pose_folder = os.path.join(
        repo_folder, "submodule", "lib_maeve_py", "maeve", "nn", "trtpose", "models")
    images_dir = os.path.join(data_folder, "h36m")
    annotations_dir = os.path.join(data_folder, "annotations")
    tasks_dir = os.path.join(repo_folder, "trt_pose", "tasks", "human_pose", "experiments")

    with open(os.path.join(tasks_dir, REF_TASK)) as f:
        ref_task_json = json.load(f)

    with open(os.path.join(human_pose_folder, REF_HUMAN_POSE)) as f:
        ref_human_pose_json = json.load(f)

    tasks = []
    continual_tasks = []
    for teacher in TEACHERS:
        save_task_info(
            ref_task_json, initial_state_dict, images_dir, annotations_dir, tasks_dir, 
            "person_keypoints_trainh36m_{}.json".format(teacher), 
            "person_keypoints_valh36m_uniformsampling10_{}.json".format(teacher),
            "h36m_{}_{}".format(teacher, REF_TASK))
        save_continual_task_info(
            ref_task_json, initial_state_dict, images_dir, annotations_dir, tasks_dir, 
            "continualtrain_person_keypoints_{}_{}.json".format(CONTINUAL_LEARNING_SUBJECT.lower(), teacher), 
            "continualtrain_person_keypoints_{}_{}.json".format(CONTINUAL_LEARNING_SUBJECT.lower(), teacher), 
            "continualval_person_keypoints_{}_uniformsampling10_{}.json".format(CONTINUAL_LEARNING_SUBJECT.lower(), teacher), 
            "continual_h36m_{}_{}".format(teacher, REF_TASK))
        
        tasks.append(os.path.join(tasks_dir, "h36m_{}_{}".format(teacher, REF_TASK)))
        continual_tasks.append(os.path.join(tasks_dir, "continual_h36m_{}_{}".format(teacher, REF_TASK)))

        create_human_pose(ref_human_pose_json, human_pose_folder, "h36m_{}".format(teacher))

    for teacher in TEACHERS: 
        for sampling in SAMPLINGS: 
            for perc in PERCENTAGES:
                perc_str = "{}".format(int(perc * 100))
                save_task_info(
                    ref_task_json, initial_state_dict, images_dir, annotations_dir, tasks_dir, 
                   "person_keypoints_trainh36m_{}sampling{}_{}.json".format(sampling, perc_str, teacher), 
                   "person_keypoints_valh36m_uniformsampling10_{}.json".format(teacher), 
                   "{}{}_h36m_{}_{}".format(sampling, perc_str, teacher, REF_TASK))
                save_continual_task_info(
                    ref_task_json, initial_state_dict, images_dir, annotations_dir, tasks_dir, 
                   "continualtrain_person_keypoints_{}_{}sampling{}_{}.json".format(CONTINUAL_LEARNING_SUBJECT.lower(), sampling, perc_str, teacher), 
                   "continualtrain_person_keypoints_{}_{}.json".format(CONTINUAL_LEARNING_SUBJECT.lower(), teacher), 
                   "continualval_person_keypoints_{}_uniformsampling10_{}.json".format(CONTINUAL_LEARNING_SUBJECT.lower(), teacher), 
                   "continual_{}{}_h36m_{}_{}".format(sampling, perc_str, teacher, REF_TASK), perc)
                tasks.append(os.path.join(tasks_dir, "{}{}_h36m_{}_{}".format(sampling, perc_str, teacher, REF_TASK)))
                continual_tasks.append(os.path.join(tasks_dir, "continual_{}{}_h36m_{}_{}".format(sampling, perc_str, teacher, REF_TASK)))

                create_human_pose(ref_human_pose_json, human_pose_folder, "h36m_{}sampling{}_{}".format(sampling, perc_str, teacher))
    
    gen_bash_script(tasks, os.path.join(repo_folder, "trt_pose", "run_train.bash"), is_continual=False)
    gen_bash_script(continual_tasks, os.path.join(repo_folder, "trt_pose", "run_train_continual.bash"), is_continual=True)

    percentages_string = ["{}".format(int(perc * 100)) for perc in PERCENTAGES]
    for sampling in SAMPLINGS: 
        gen_inference(repo_folder, TEACHERS, [sampling], percentages_string, "run_parcopose_h36m_{}sampling.bash".format(sampling))
    gen_inference(repo_folder, TEACHERS, None, None, "run_parcopose_h36m.bash")
    gen_inference(repo_folder, "trtpose", None, None, "run_trtpose.bash")
    gen_inference(repo_folder, "openpose1", None, None, "run_openpose.bash")
    gen_inference(repo_folder, "parco", None, None, "run_parcopose.bash")

    copy_checkpoints(TEACHERS, SAMPLINGS, PERCENTAGES, tasks_dir, data_folder, human_pose_folder)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automatic generation of tasks", epilog="PARCO")
    parser.add_argument("--repo-folder", 
                        "-rf", 
                        dest="repo_folder", 
                        required=True, 
                        help="Folder with EdgeKnowledgeDistillation repository")
    parser.add_argument("--data-folder", 
                        "-df", 
                        dest="data_folder", 
                        required=True, 
                        help="Folder with data (annotations and images)")
    args = parser.parse_args()
    main(args.repo_folder, args.data_folder)