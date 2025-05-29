import bpy
import os
import sys
import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=int)

args = sys.argv
if '--' in args:
    args = args[args.index('--') + 1:]
    p_args = parser.parse_args(args)
else:
    p_args = parser.parse_args('')

# reset fluid cache for all domains
for obj in bpy.context.scene.objects:
    if obj.type == "MESH":
        mesh_name = obj.name
        for mod_key in obj.modifiers.keys():
            if mod_key == "Fluid":
                mod = obj.modifiers[mod_key]
                if mod.domain_settings:
                    target_path = f"//fluid_cache/{mesh_name}"
                    mod.domain_settings.cache_directory = bpy.path.abspath(target_path)
                    print(mod.domain_settings.cache_directory)


scene = bpy.context.scene
if scene.render.ffmpeg.format == 'MKV':
    scene.render.filepath = bpy.path.abspath(f"//render.mkv")
if scene.render.ffmpeg.format == 'MP4':
    scene.render.filepath = bpy.path.abspath(f"//render.mp4")

bpy.ops.render.render(animation=True)