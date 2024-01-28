import os
import time
from matplotlib import pyplot as plt
from matplotlib import colors
from numpy import mat
import yaml
import json
import importlib

class CustomArgs(dict):

    def __getattr__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        self[key] = value


def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'"""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_

def save_args(args):
    config = vars(args)
    dir_name = time.strftime("%Y-%m-%d")
    file_name = "{}-{}.yaml".format(time.strftime("%H:%M:%S", time.localtime()), args.special_mark)

    if not os.path.exists("config"):
        os.mkdir("config")
    if not os.path.exists(os.path.join("config", dir_name)):
        os.mkdir(os.path.join("config", time.strftime("%Y-%m-%d")))

    with open(os.path.join("config", dir_name, file_name), "w") as f:
        f.write(yaml.dump(config))

def load_args(config_path, gpu=0):
    with open(config_path, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
        args = CustomArgs(args)
    
    args.test_from_checkpoint = config_path
    args.wandb = False
    args.gpu = gpu

    return args


def handle_args_by_local_env(args):
    # * test
    if args.test_from_checkpoint:
        args = load_args(args.test_from_checkpoint, args.gpu)

    temp = args.data_dir.split("/")
    dataset_name = temp[-1] if not args.few_shot or "few-shot" not in temp else temp[temp.index("few-shot")-1]
    args.dataset_name = dataset_name

    with open(os.path.join(args.data_dir, "rel2id.json"), "r") as file:
        t = json.load(file)
        label_list = sorted(list(t.keys()), key=lambda x: t[x])
    args.rel_num = len(label_list)

    # * arg entity labels
    # sub_types = []
    # obj_types = []
    # split_symbol = "-" if dataset_name == "semeval" else ":"

    # rel_ent_types = {}
    # for label in label_list:
    #     label_idx = t[label]
    #     if label in ["no_relation", "NA", "Other"]:
    #         sub_type, obj_type = "no", "no"
    #     else:
    #         label = label.lower().replace("per", "preson").replace("org", "organization")
    #         types = label.split("(")[0].split(split_symbol)

    #         if "e2,e1" in label:
    #             obj_type, sub_type = types
    #         else:
    #             sub_type, obj_type = types

    #     if sub_type not in sub_types: sub_types.append(sub_type)
    #     if obj_type not in obj_types: obj_types.append(obj_type)

    #     rel_ent_types[label_idx] = [sub_types.index(sub_type), obj_types.index(obj_type)]

    # args.sub_types = sub_types
    # args.obj_types = obj_types
    # args.rel_ent_types = rel_ent_types
    
    ##### ? config area start
    # args.padding_layers = 3

    ##### ? config end

    # * handle gpu
    if args.gpu == "not specified" or args.test_from_checkpoint:
        args.gpu = get_gpu_by_user_input()

    # * handle server env
    with open("./.vscode/env") as f:
        env = f.readline()

    if not env: return args
    if env == "1080Ti" or env == "3090":
        bert_path = "/home/zwj/nlp/Bert"
    elif env == "2080Ti":
        bert_path = "/data/zwj/bert"
        args.output_dirpath = "/data/zwj/RelationPromptOutput"

    if env and len(args.model_name_or_path.split("/")) == 1:
        args.model_name_or_path = os.path.join(bert_path, args.model_name_or_path)

    if "base" in args.model_name_or_path and args.few_shot:
        print("\033[33m[WARNING]\033[0m", "Few-Shot will not perform well on BASE model.")

    args.model_dir = os.path.join(args.output_dirpath, dataset_name, args.special_mark)
    if not os.path.exists(args.model_dir): os.mkdir(args.model_dir)

    # * handle info
    cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    runing_info = {
        "Dataset Name": dataset_name,
        "Max Epoch": args.max_epochs,
        "Batch Size": args.batch_size,
        "GPU": args.gpu,
        "Note": args.note,
    }
    print(f"\nRunning at {cur_time}\n- " + "\n- ".join([f"{key}: {value}" for key, value in runing_info.items()]) + "\n")

    return args


def get_gpu_by_user_input():
    os.system("gpustat")
    gpu = input("\nSelect GPU >>> ")
    assert gpu and int(gpu[0]) in [0, 1, 2, 3], "Can not run scripts on GPU: {}".format(gpu if gpu else "None")
    print("This scripts will use GPU {}".format(gpu))
    return gpu


def plt_mat_and_save_to_model_dir(mat, out_dir, title=None):
    plt.matshow(mat.cpu().detach().numpy())
    if title:
        plt.title(title)
    plt.savefig(os.path.join(out_dir, str(int(time.time())) + ".png"))
    plt.close()

def plt_mats_and_save_to_model_dir(mats, out_dir, title=None, Nr=0, Nc=0):
    assert len(mats) in [12, 24] or Nr * Nc != 0, f"size not enough. len(mats): {len(mats)}, Nr: {Nr}, Nc: {Nc}."
    assert len(mats) >= Nr * Nc, f"size not enough. len(mats): {len(mats)}, Nr: {Nr}, Nc: {Nc}."

    if len(mats) == 12:
        Nr, Nc = 3, 4
    elif len(mats) == 24:
        Nr, Nc = 4, 6

    fig, axs = plt.subplots(Nr, Nc)
    fig.suptitle(title)

    images = []
    for i in range(Nr):
        for j in range(Nc):
            # Generate data with a range that varies from one plot to the next.
            data = mats[i*Nc+j].cpu().detach().numpy()
            images.append(axs[i, j].imshow(data))
            axs[i, j].label_outer()

    # Find the min and max of all colors for use in setting the color scale.
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)

    fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)

    # Make images respond to changes in the norm of other images (e.g. via the
    # "edit axis, curves and images parameters" GUI on Qt), but be careful not to
    # recurse infinitely!
    def update(changed_image):
        for im in images:
            if (changed_image.get_cmap() != im.get_cmap()
                    or changed_image.get_clim() != im.get_clim()):
                im.set_cmap(changed_image.get_cmap())
                im.set_clim(changed_image.get_clim())

    for im in images:
        im.callbacks.connect('changed', update)

    plt.savefig(os.path.join(out_dir, str(int(time.time())) + ".png"))
    plt.close()