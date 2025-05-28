import bpy
import os
import sys
import argparse
import time

# -----------------------------------
# Bake all fluid simulations
# -----------------------------------
def bake_fluid_simulations():
    for obj in bpy.data.objects:
        for mod in obj.modifiers:
            if mod.type == 'FLUID' and mod.fluid_type == 'DOMAIN':
                print(f"Baking fluid for domain: {obj.name}")
                
                # Set the object as active
                bpy.context.view_layer.objects.active = obj

                bpy.ops.fluid.free_all()
                time.sleep(0.5)
                bpy.ops.fluid.bake_data()
                bpy.ops.fluid.bake_mesh()


parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=int)

args = sys.argv
if '--' in args:
    args = args[args.index('--') + 1:]
    p_args = parser.parse_args(args)
else:
    p_args = parser.parse_args('')

bake_fluid_simulations()
