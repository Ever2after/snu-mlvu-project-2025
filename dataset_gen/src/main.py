import os
import sys
import subprocess
import shutil
from sample import *
import json
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))
#print(os.getcwd())

def process_value(val):
    if type(val).__name__ == "tuple":
        res = ""
        for i in val:
            res += f"{i:.4f}" + ","
        return f"v{res[:-1]}"
    else:
        return str(f"v{val}")

def gen_scene(script_path, sample):
    subproc_args = [
        BLENDER_PATH,
        "--background",
        "--python", script_path, 
        "--", 
    ]
    for var_name in sample:
        if sample[var_name] is not None:
            subproc_args.extend([f"--{var_name}", process_value(sample[var_name])])
    
    subprocess.run(subproc_args, check=True)

def execute_bake(blender_path, scene_aug_path, script_path, cache_path):
    subprocess.run([
        blender_path,
        "--background", scene_aug_path,
        "--python", script_path, 
        "--", 
        "--cache", os.path.abspath(cache_path)
    ], check=True)

def execute_render(blender_path, scene_aug_path, script_path, idx):
    subprocess.run([
        blender_path,
        "--background", scene_aug_path,
        "--python", script_path, 
        "--", 
        "--idx", str(idx)
    ], check=True)

def write_log(output_path, sample, cache_dir_name):
    # open summary file
    defaults = {}
    with open(os.path.join(os.path.join(output_path, "../../../"), "out_summary.txt")) as f:
        lines = f.readlines()
        is_param_region = False
        for line in lines:
            line = line.strip("\n")
            if len(line) < 2: continue

            if not is_param_region and line == "DEFAULTS":
                is_param_region = True
            elif is_param_region and line[0] == "\t" and line[1] != "\t":
                name, value = line[1:].split(": ")
                defaults[name] = value
            elif is_param_region and line[0] != "\t":
                break

    res = {}
    for var_name in sample:
        if sample[var_name] is None:
            res[var_name] = defaults[var_name]
        else:
            res[var_name] = sample[var_name]
    res["_cache_dir_name"] = cache_dir_name

    with open(output_path, "w") as json_file:
        json.dump(res, json_file)

##### CONFIG #####
BLENDER_PATH = "C:\\blender_src\\build_windows_x64_vc17_Release\\bin\\Release\\blender.exe" # for Windows
RENDER_BLENDER_PATH = "C:\\Program Files\\Blender Foundation\\Blender 4.4\\blender.exe"
# "/Applications/Blender.app/Contents/MacOS/Blender" # for macOS
# "/usr/bin/blender" # for Linux

GEN_SCRIPT = os.path.abspath("generate_scene.py")
BAKE_SCRIPT = os.path.abspath("bake_scene.py")
RENDER_SCRIPT = os.path.abspath("render_scene.py")

mode = "all" # ["scene_gen_only", "bake", "render", "all"]
# "scene_gen_only": generate scenes and do not bake/render
# "bake": generate scenes, bake but not render
# "render": render scene given .blend file and fluid cache
# "all": generate scenes, bake and render
assert mode in ["scene_gen_only", "bake", "render", "all"]

##################

with open(f"../scene/_ignore.txt") as f:
    ignore_list = f.readlines()
    for i in range(len(ignore_list)):
        ignore_list[i] = ignore_list[i].replace("\n", "")

for i in os.listdir("../scene"):
    if i == "_ignore.txt" or i in ignore_list: continue
    print(f"Current scene: {i}\n\n")

    # attempt to create and run the generation script
    script_path = f"../scene/{i}/out.py"
    sample_config_path = f"../scene/{i}/sampleconf.py"

    gen_script_path = os.path.abspath("generate_scene.py")
    subprocess.run([
        BLENDER_PATH,
        "--background", os.path.abspath(f"../scene/{i}/scene.blend"), 
        "--python", "generate_scene.py", 
        "--", 
        "--out_script_path", os.path.abspath(f"../scene/{i}/out.py"), 
    ], check=True)

    # 1. sample the parameters
    sampler = GridSample_joint(sample_config_path)

    for idx in range(len(sampler)):
        sample = sampler[idx]
        print(f"current: {idx + 1}/{len(sampler)}")
        for n in sample:
            print(f"\t{n}: {'Default' if sample[n] is None else sample[n]}")

        # 2. create augmented scene
        # scene_aug will be automatically created
        if mode in ["scene_gen_only", "bake", "all"]:
            gen_scene(script_path, sample)
        
            # archive scene and generation script
            os.makedirs(f"../scene/{i}/output/{idx}", exist_ok=True)
            shutil.copy(f"../scene/{i}/scene_aug.blend", f"../scene/{i}/output/{idx}/scene.blend")
            shutil.copy(f"../scene/{i}/out.py", f"../scene/{i}/output/{idx}/out.py")

            # write log file
            cache_dir_name = sampler.get_cache_folder_name(idx)
            write_log(f"../scene/{i}/output/{idx}/params.log", sample, cache_dir_name)

        # 3. bake scene
        scene_aug_path = f"../scene/{i}/output/{idx}/scene.blend"
        if mode in ["bake", "all"]:
            cache_dir = f"../scene/{i}/fluid_cache/{cache_dir_name}"
            if not os.path.exists(cache_dir):
                # bake scene only when cache does not exist
                execute_bake(BLENDER_PATH, scene_aug_path, BAKE_SCRIPT, cache_dir)
        
        if mode in ["bake", "all"]:
            # copy generated fluid cache
            output_fluid_cache_dir = f"../scene/{i}/output/{idx}/fluid_cache"
            os.makedirs(output_fluid_cache_dir, exist_ok=True)
            shutil.copytree(cache_dir, output_fluid_cache_dir, dirs_exist_ok=True)
        
        # 3. render scene
        if mode in ["render", "all"]:
            execute_render(RENDER_BLENDER_PATH, scene_aug_path, RENDER_SCRIPT, idx)
