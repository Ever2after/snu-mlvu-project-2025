import os
import sys
import bpy
import argparse

def generate_scene(
    sim_res, 
    ):
    # Delete all existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # 1. meshes
    base_path = 'c:\\Users\\jason\\바탕 화면\\lecture\\시각적\\final\\dataset_gen\\scene\\scene_old\\meshes'

    ###################
    # Mesh: cup
    bpy.ops.wm.obj_import(filepath=os.path.join(base_path, 'cup.obj').replace('/', '\\'))
    imported_obj = bpy.context.selected_objects[0]
    imported_obj.name = 'cup'
    imported_obj.location = (0.0000, 0.0000, 1.0000)
    imported_obj.rotation_euler = (0.0000, 0.0000, 0.0000)
    imported_obj.scale = (1.0000, 1.0000, 1.0000)

    imported_obj.hide_render = False
    # Material: cup
    mat = bpy.data.materials.new(name='cup')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (0.8000, 0.8000, 0.8000, 1.0000)
    bsdf.inputs['Metallic'].default_value = 0.0000
    bsdf.inputs['Roughness'].default_value = 0.2559
    bsdf.inputs['IOR'].default_value = 1.4500
    bsdf.inputs['Alpha'].default_value = 1.0000
    bsdf.inputs['Transmission Weight'].default_value = 0.0000
    
    imported_obj.data.materials.append(bpy.data.materials['cup'])

    # effector settings
    mod = imported_obj.modifiers.new(name='Fluid', type='FLUID')
    mod.fluid_type = 'EFFECTOR'

    dst_settings = mod.effector_settings
    dst_settings.surface_distance = 0.1000
    dst_settings.use_plane_init = False

    ###################
    # Mesh: domain
    bpy.ops.wm.obj_import(filepath=os.path.join(base_path, 'domain.obj').replace('/', '\\'))
    imported_obj = bpy.context.selected_objects[0]
    imported_obj.name = 'domain'
    imported_obj.location = (0.0000, 0.0000, 2.4000)
    imported_obj.rotation_euler = (0.0000, 0.0000, 0.0000)
    imported_obj.scale = (1.5000, 1.5000, 2.4000)

    imported_obj.hide_render = False
    # Material: water
    mat = bpy.data.materials.new(name='water')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (0.6491, 0.8003, 0.8003, 1.0000)
    bsdf.inputs['Metallic'].default_value = 0.0000
    bsdf.inputs['Roughness'].default_value = 0.0000
    bsdf.inputs['IOR'].default_value = 1.3300
    bsdf.inputs['Alpha'].default_value = 1.0000
    bsdf.inputs['Transmission Weight'].default_value = 1.0000
    
    imported_obj.data.materials.append(bpy.data.materials['water'])

    # domain settings
    mod = imported_obj.modifiers.new(name='Fluid', type='FLUID')
    mod.fluid_type = 'DOMAIN'

    dst_settings = mod.domain_settings
    dst_settings.domain_type = 'LIQUID'
    dst_settings.cache_type = 'ALL'
    dst_settings.resolution_max = sim_res
    dst_settings.use_mesh = True
    dst_settings.cfl_condition = 4.0000
    dst_settings.particle_radius = 1.0000
    dst_settings.particle_band_width = 3.0000
    dst_settings.cache_frame_start = 1
    dst_settings.cache_frame_end = 90
    dst_settings.cache_directory = 'c:\\Users\\jason\\바탕 화면\\lecture\\시각적\\final\\dataset_gen\\scene\\scene_old\\fluid_cache'

    ###################
    # Mesh: source
    bpy.ops.wm.obj_import(filepath=os.path.join(base_path, 'source.obj').replace('/', '\\'))
    imported_obj = bpy.context.selected_objects[0]
    imported_obj.name = 'source'
    imported_obj.location = (0.4000, 0.5200, 3.7200)
    imported_obj.rotation_euler = (0.0000, 0.0000, 0.0000)
    imported_obj.scale = (0.3000, 0.3000, 0.3000)

    imported_obj.hide_render = False
    # flow settings
    mod = imported_obj.modifiers.new(name='Fluid', type='FLUID')
    mod.fluid_type = 'FLOW'

    dst_settings = mod.flow_settings
    dst_settings.flow_type = 'LIQUID'
    dst_settings.flow_behavior = 'INFLOW'
    dst_settings.flow_source = 'MESH'

    ###################
    # Mesh: cylinder
    bpy.ops.wm.obj_import(filepath=os.path.join(base_path, 'cylinder.obj').replace('/', '\\'))
    imported_obj = bpy.context.selected_objects[0]
    imported_obj.name = 'cylinder'
    imported_obj.location = (-0.3600, -0.3300, 1.0000)
    imported_obj.rotation_euler = (0.0000, 0.0000, 0.0000)
    imported_obj.scale = (0.2000, 0.2000, 1.0000)

    imported_obj.hide_render = False
    # Material: cyl_mtr
    mat = bpy.data.materials.new(name='cyl_mtr')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (0.8000, 0.0000, 0.0173, 1.0000)
    bsdf.inputs['Metallic'].default_value = 0.8976
    bsdf.inputs['Roughness'].default_value = 0.5000
    bsdf.inputs['IOR'].default_value = 1.4500
    bsdf.inputs['Alpha'].default_value = 1.0000
    bsdf.inputs['Transmission Weight'].default_value = 0.0000
    
    imported_obj.data.materials.append(bpy.data.materials['cyl_mtr'])

    # effector settings
    mod = imported_obj.modifiers.new(name='Fluid', type='FLUID')
    mod.fluid_type = 'EFFECTOR'

    dst_settings = mod.effector_settings
    dst_settings.surface_distance = 0.1000
    dst_settings.use_plane_init = False

    # 2. lights
    # Light: point
    light_data = bpy.data.lights.new(name='point', type='POINT')
    light_data.energy = 100.0000
    light_data.color = (1.0000, 1.0000, 1.0000)
    light_obj = bpy.data.objects.new(name='point', object_data=light_data)
    light_obj.location = (0.0000, 0.0000, 3.3100)
    light_obj.rotation_euler = (0.0000, 0.0000, 0.0000)
    bpy.context.collection.objects.link(light_obj)

    # 3. cameras
    # Camera: camera
    cam_data = bpy.data.cameras.new(name='camera')
    cam_data.lens = 50.0000
    cam_data.sensor_width = 36.0000
    cam_data.type = 'PERSP'
    cam_data.clip_start = 0.1000
    cam_data.clip_end = 1000.0000
    cam_obj = bpy.data.objects.new('camera', cam_data)
    cam_obj.location = (0.0000, 3.3900, 4.7800)
    cam_obj.rotation_euler = (0.8783, 0.0064, -3.0614)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 16
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.denoising_use_gpu = True

    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'PERC_LOSSLESS'
    scene.render.ffmpeg.ffmpeg_preset = 'BEST'
    scene.render.ffmpeg.video_bitrate = 6000
    scene.render.ffmpeg.gopsize = 18
    scene.frame_start = 1
    scene.frame_end = 90

    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    scene.render.filepath = '/tmp\\'

parser = argparse.ArgumentParser()
parser.add_argument('--sim_res', type=int, default=128)
args = sys.argv
if '--' in args:
    args = args[args.index('--') + 1:]
    p_args = parser.parse_args(args)
else:
    p_args = parser.parse_args('')

generate_scene(
sim_res=p_args.sim_res, 
)

bpy.ops.wm.save_mainfile(filepath='c:\\Users\\jason\\바탕 화면\\lecture\\시각적\\final\\dataset_gen\\scene\\scene_old\\scene_aug.blend')
exit()
