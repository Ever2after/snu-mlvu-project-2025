import os
import sys
import argparse
import subprocess
import shutil
from sample import *
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

"""
aug_scenes.py -s ... [--init] [--clean]

Mandetory options: 
-s: Scene(s) to perform action, use "@All" to perform action on all scenes

modes
<no mode>: peform difference comparison, add difference to sampleconf.py
--init: Initialize, copy original scene to make scene_delta.blend
--clean: Remove scene_delta.blend
"""

def write_values(values):
    pass
    

def main(args):
    scenes = []
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
        if args.init:
            # create scene_delta.blend
            shutil.copy(f"../scene/{scene_name}/scene.blend", f"../scene/{scene_name}/scene_delta.blend")

        elif args.clean:
            # remove scene_delta.blend, out_delta.py

            pass
        else:
            pass
    print(scenes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', type=str)
    parser.add_argument('--init', action="store_true")
    parser.add_argument('--clean', action="store_true")
    p_args = parser.parse_args()

    assert not p_args.init or not p_args.clean # both cannot be true

    main(p_args)