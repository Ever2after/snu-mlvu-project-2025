import os
import sys
import importlib.util
import subprocess
from sample import SimpleSample
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))
BLENDER_PATH = "C:\\Program Files\\Blender Foundation\\Blender 4.4\\blender.exe"
GEN_SCRIPT = os.path.abspath("generate_scene.py")
BAKE_RENDER_SCRIPT = os.path.abspath("bake_render_scene.py")
#print(os.getcwd())


for i in os.listdir("../scene"):
    print(i)

    if i != "splash": continue

    # attempt to run the generation script
    script_path = f"../scene/{i}/out.py"
    sample_config_path = f"../scene/{i}/sampleconf.py"
    if not os.path.exists(script_path):
        # make the generation script if it does not exist
        print("not exist")

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
    sampler = SimpleSample(sample_config_path)

    for idx in range(1):
        sample = sampler[idx]

        # 2. create augmented scene
        # scene_aug will be automatically created
        subproc_args = [
            BLENDER_PATH,
            "--background",
            "--python", script_path, 
            "--", 
        ]
        for var_name in sample:
            subproc_args.extend([f"--{var_name}", str(sample[var_name])])
        
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
