import os
import sys
import subprocess
import shutil
from sample import *
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))
BLENDER_PATH = "C:\\Program Files\\Blender Foundation\\Blender 4.4\\blender.exe"
GEN_SCRIPT = os.path.abspath("generate_scene.py")
BAKE_RENDER_SCRIPT = os.path.abspath("bake_render_scene.py")
#print(os.getcwd())

def process_value(val):
    if type(val).__name__ == "tuple":
        res = ""
        for i in val:
            res += f"{i:.4f}" + ","
        return res[:-1]
    else:
        return str(val)


for i in os.listdir("../scene"):
    print(i)

    if i != "splash": continue

    # attempt to create and run the generation script
    script_path = f"../scene/{i}/out.py"
    sample_config_path = f"../scene/{i}/sampleconf.py"

    gen_script_path = os.path.abspath("generate_scene.py")
    print(gen_script_path)
    subprocess.run([
        BLENDER_PATH,
        "--background", os.path.abspath(f"../scene/{i}/scene.blend"), 
        "--python", "generate_scene.py", 
        "--", 
        "--out_dir", os.path.abspath(f"../scene/{i}"), 
    ], check=True)

    # 1. sample the parameters
    sampler = GridSample(sample_config_path)

    for idx in range(len(sampler)):
        sample = sampler[idx]
        print(f"current: {idx + 1}/{len(sampler)}")
        for n in sample:
            print(f"\t{n}: {sample[n]}")

        # 2. create augmented scene
        # scene_aug will be automatically created
        subproc_args = [
            BLENDER_PATH,
            "--background",
            "--python", script_path, 
            "--", 
        ]
        for var_name in sample:
            subproc_args.extend([f"--{var_name}", process_value(sample[var_name])])
        
        subprocess.run(subproc_args, check=True)

        # 3. bake and render scene
        scene_aug_path = f"../scene/{i}/scene_aug.blend"
        subprocess.run([
            BLENDER_PATH,
            "--background", scene_aug_path,
            "--python", BAKE_RENDER_SCRIPT, 
            "--", 
            "--idx", str(idx)
        ], check=True)

        # 4. copy fluid cache
        os.makedirs(f"../scene/{i}/out/output_{idx}/fluid_cache", exist_ok=True)
        shutil.copytree(f"../scene/{i}/fluid_cache", f"../scene/{i}/out/output_{idx}/fluid_cache", dirs_exist_ok=True)
