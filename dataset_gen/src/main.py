import os
import sys
import subprocess
import shutil
from sample import *
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
        subproc_args.extend([f"--{var_name}", process_value(sample[var_name])])
    
    subprocess.run(subproc_args, check=True)

def execute_bake_or_render(scene_aug_path, script_path, idx):
    subprocess.run([
        BLENDER_PATH,
        "--background", scene_aug_path,
        "--python", script_path, 
        "--", 
        "--idx", str(idx)
    ], check=True)

##### CONFIG #####
BLENDER_PATH = "C:\\Program Files\\Blender Foundation\\Blender 4.4\\blender.exe"
GEN_SCRIPT = os.path.abspath("generate_scene.py")
BAKE_SCRIPT = os.path.abspath("bake_scene.py")
RENDER_SCRIPT = os.path.abspath("render_scene.py")

mode = "all" # ["scene_gen_only", "bake", "render", "both"]
# "scene_gen_only": generate scenes and do not bake/render
# "bake": generate scenes, bake but not render
# "render": render scene given .blend file and fluid cache
# "all": generate scenes, bake and render

##################

for i in os.listdir("../scene"):
    print(i)

    # attempt to create and run the generation script
    script_path = f"../scene/{i}/out.py"
    sample_config_path = f"../scene/{i}/sampleconf.py"

    gen_script_path = os.path.abspath("generate_scene.py")
    subprocess.run([
        BLENDER_PATH,
        "--background", os.path.abspath(f"../scene/{i}/scene.blend"), 
        "--python", "generate_scene.py", 
        "--", 
        "--out_dir", os.path.abspath(f"../scene/{i}"), 
    ], check=True)

    # 1. sample the parameters
    sampler = GridSample_joint(sample_config_path)

    for idx in range(len(sampler)):
        sample = sampler[idx]
        print(f"current: {idx + 1}/{len(sampler)}")
        for n in sample:
            print(f"\t{n}: {sample[n]}")

        # 2. create augmented scene
        # scene_aug will be automatically created
        if mode in ["scene_gen_only", "bake", "all"]:
            gen_scene(script_path, sample)
        
            # archive scene and generation script
            os.makedirs(f"../scene/{i}/output/{idx}", exist_ok=True)
            shutil.copy(f"../scene/{i}/scene_aug.blend", f"../scene/{i}/output/{idx}/scene.blend")
            shutil.copy(f"../scene/{i}/out.py", f"../scene/{i}/output/{idx}/out.py")

        # 3. bake scene
        scene_aug_path = f"../scene/{i}/output/{idx}/scene.blend"
        if mode in ["bake", "all"]:
            execute_bake_or_render(scene_aug_path, BAKE_SCRIPT, idx)
        
        if mode in ["bake", "all"]:
            # copy generated fluid cache
            os.makedirs(f"../scene/{i}/output/{idx}/fluid_cache", exist_ok=True)
            shutil.copytree(f"../scene/{i}/fluid_cache", f"../scene/{i}/output/{idx}/fluid_cache", dirs_exist_ok=True)
        
        # 3. render scene
        if mode in ["render", "all"]:
            execute_bake_or_render(scene_aug_path, RENDER_SCRIPT, idx)
