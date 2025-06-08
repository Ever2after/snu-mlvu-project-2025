import os
import sys
import argparse
import subprocess
import shutil
from sample import *
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

BLENDER_PATH = "C:\\blender_src\\build_windows_x64_vc17_Release\\bin\\Release\\blender.exe"

"""
aug_scene.py -s ... [--init] [--clean]

Mandetory options: 
-s: Scene(s) to perform action, use "@All" to perform action on all scenes

modes
<no mode>: peform difference comparison, add difference to sampleconf.py
--init: Initialize, copy original scene to make scene_delta.blend
--clean: Remove scene_delta.blend
"""

def write_values(values):
    res = ""
    for var_set in values:
        res += f"    {var_set} : [\n"
        for value in values[var_set]:
            res += f"        {value},\n"
        res += "    ],\n"
    res += "}\n"

    return res
    

def main(args):
    scenes = [] # collect scenes to perform script
    if args.s == "@All":
        scenes = os.listdir("../scene")
        scenes.remove("_ignore.txt")
    else:
        scenes_all = os.listdir("../scene")
        for scene_name in args.s.split(","):
            scene_name_strip = scene_name.strip(" ")
            scenes.append(scene_name_strip)

            # check if the given scene name exists
            assert scene_name_strip in scenes_all
    
    for scene_name in scenes:
        print(f"current: {scene_name}")
        if args.init:
            # create scene_delta.blend
            shutil.copy(f"../scene/{scene_name}/scene.blend", f"../scene/{scene_name}/scene_delta.blend")

        elif args.clean:
            # remove scene_delta.blend, out_delta.py
            if os.path.exists(f"../scene/{scene_name}/scene_delta.blend"):
                os.remove(f"../scene/{scene_name}/scene_delta.blend")
            if os.path.exists(f"../scene/{scene_name}/out_delta.py"):
                os.remove(f"../scene/{scene_name}/out_delta.py")

        else:
            # generate out_delta.py
            subprocess.run([
                BLENDER_PATH,
                "--background", os.path.abspath(f"../scene/{scene_name}/scene_delta.blend"), 
                "--python", "generate_scene.py", 
                "--", 
                "--out_script_path", os.path.abspath(f"../scene/{scene_name}/out_delta.py"), 
                "--args_only", 
            ], check=True)

            # parse the variables
            var_script = ""
            with open(f"../scene/{scene_name}/out_delta.py", "rb") as f:
                lines = f.readlines()
                is_param_region = False
                for line in lines:
                    line = line.decode()
                    if not is_param_region and line.startswith("parser = argparse.ArgumentParser()"):
                        is_param_region = True
                    elif is_param_region:
                        if line.startswith("parser.add_argument("):
                            var_script += line
                        else:
                            break
            
            parser = argparse.ArgumentParser()
            exec(var_script)
            p_args = parser.parse_args('')
            var_delta = vars(p_args)

            # compare and add novel values
            module_name = "loaded_sample_conf"
            spec = importlib.util.spec_from_file_location(module_name, f"../scene/{scene_name}/sampleconf.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            cfg_values = getattr(module, "values", None)

            for var_set in cfg_values:
                # generate candidate value set
                to_append = []
                for var_name in var_set:
                    to_append.append(var_delta[var_name])

                to_append = tuple(to_append)
                if to_append not in cfg_values[var_set]:
                    print(to_append)
                    print(cfg_values[var_set])
                    cfg_values[var_set].append(to_append)
            
            # pretty-print modified values
            new_file = ""
            with open(f"../scene/{scene_name}/sampleconf.py", "r") as f:
                lines = f.readlines()
                is_values_region = False
                for line in lines:
                    if not is_values_region:
                        new_file += line
                        if line.startswith("values = "):
                            is_values_region = True
                            new_file += write_values(cfg_values)
                    if is_values_region and line.startswith("}"):
                        is_values_region = False
                f.close()
            
            with open(f"../scene/{scene_name}/sampleconf.py", "w") as f:
                f.write(new_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str)
    parser.add_argument('--init', action="store_true")
    parser.add_argument('--clean', action="store_true")
    p_args = parser.parse_args()

    assert not p_args.init or not p_args.clean # both cannot be true

    main(p_args)