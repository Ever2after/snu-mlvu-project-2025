import re

# Define placeholders for dataset paths
DEMO1 = {
    "annotation_path": "demo/single_images.json",
    "data_path": "",
}

DEMO2 = {
    "annotation_path": "demo/video.json",
    "data_path": "demo/videos",
}

FREE_FALL = {
    "annotation_path": "../data/qwen2.5-vl/free_fall/train.json",
    "data_path": "../dataset_gen/scene",
}

OBJ_MOVING = {
    "annotation_path": "../data/qwen2.5-vl/obj_moving/train.json",
    "data_path": "../dataset_gen/scene",
}

OBJ_INTERACTION = {
    "annotation_path": "../data/qwen2.5-vl/obj_interaction/train.json",
    "data_path": "../dataset_gen/scene",
}

SLOPE = {
    "annotation_path": "../data/qwen2.5-vl/slope/train.json",
    "data_path": "../dataset_gen/scene",
}

RIPPLE = {
    "annotation_path": "../data/qwen2.5-vl/ripple/train.json",
    "data_path": "../dataset_gen/scene",
}

data_dict = {
    "demo1": DEMO1,
    "demo2": DEMO2,
    "free_fall": FREE_FALL,
    "obj_moving": OBJ_MOVING,
    "obj_interaction": OBJ_INTERACTION,
    "slope": SLOPE,
    "ripple": RIPPLE
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
