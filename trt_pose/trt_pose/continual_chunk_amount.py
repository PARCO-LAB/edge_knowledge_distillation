import argparse
import json
import math



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)

    chunk_size = config["train_loader"]["batch_size"]

    train_annotations_fp = config["train_dataset"]["annotations_file"]
    with open(train_annotations_fp, 'r') as f:
        train_annotations_json = json.load(f)

    train_annotations = train_annotations_json["annotations"]
    # print(len(train_annotations))
    chunk_amount = math.ceil(len(train_annotations_json["annotations"]) / chunk_size)

    print(chunk_amount)