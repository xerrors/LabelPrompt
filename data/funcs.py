import json
import os
import numpy as np
from collections import defaultdict

def add_tag(sentence, tags:list, name:str):

    if len(tags) == 1:
        tags = tags * 2

    sentence = sentence.replace(name, f"{tags[0]} {name} {tags[1]}")

    return sentence

def find_train_instances(instance, label=None):
    """ Find training samples from the training data that are closer to the example.

    @instance:
    @
    """
    pass




def sample_few_shot_data(dataset_dir, seed, ratio_or_num):
    """ Sample few-shot items from all data.

    k-shot(8,16,32) and ratio-shot(0.01,0.05,0.1,0.25,0.5)

    1. get all examples and shuffle them.
    2. count sample size and split it.
    3. export to out_dir

    """

    output_dir = os.path.join(dataset_dir, "few-shot", f"{seed}-{ratio_or_num}")
    if os.path.exists(output_dir):
        print(f"FewShot dir : {output_dir} already existed! Skip it.")
        return output_dir

    os.makedirs(output_dir)
    print(f"Creating new dir: {output_dir}.")

    data_file = ["train.txt", "val.txt"]
    for df in data_file:
        with open(os.path.join(dataset_dir, df), "r") as file_in, open(os.path.join(output_dir, df), "w+") as file_out:
            lines = file_in.readlines()
            np.random.seed(seed)
            np.random.shuffle(lines)
            total_len = len(lines)

            # * (>1) means k-shot and ensure it is int
            if ratio_or_num < 1:
                file_out.write("".join(lines[:int(ratio_or_num*total_len)]))
            elif ratio_or_num > 1 and np.floor(ratio_or_num) == ratio_or_num:
                label_list = defaultdict(list)
                for line in lines:
                    line = eval(line)
                    label = line["relation"]
                    if len(label_list[label]) < ratio_or_num:
                        label_list[label].append(line)

                for label_lines in label_list.values():
                    for line in label_lines:
                        file_out.writelines(json.dumps(line) + "\n")

    for file in os.listdir(dataset_dir):
        if file not in os.listdir(output_dir):
            path_from = os.path.join(dataset_dir, file)
            path_to = os.path.join(output_dir, file)
            os.system(f"cp {path_from} {path_to}") # if your os is windows, show use `copy` rather than `cp`

    return output_dir


    # output_dir = os.path.join(dataset_dir, name)
    # retrun output_dir

if __name__ == "__main__":
    sample_few_shot_data("/home/zwj/nlp/RelationPrompt/dataset/tacred", 0, 0.05)