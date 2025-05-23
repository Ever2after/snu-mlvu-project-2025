import os
import sys
import bpy
import argparse

def generate_scene(
    sm, 
    ):
    # Delete all existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # 1. meshes
    base_path = 'c:\\Users\\jason\\바탕 화면\\lecture\\시각적\\final\\dataset_gen\\scene\\splash\\meshes'

    ###################
    # Mesh: cup
    bpy.ops.wm.obj_import(filepath=os.path.join(base_path, 'cup.obj').replace('/', '\\'))
    imported_obj = bpy.context.selected_objects[0]
    imported_obj.name = 'cup'
    imported_obj.location = (0.0000, 0.0000, 0.0000)
    imported_obj.rotation_euler = (0.0000, 0.0000, 0.0000)
    imported_obj.scale = (1.0000, 1.0000, 1.0000)

    imported_obj.hide_render = False
    # Material: Material_001
    mat = bpy.data.materials.new(name='Material_001')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (0.8000, 0.8000, 0.8000, 1.0000)
    bsdf.inputs['Metallic'].default_value = 0.0000
    bsdf.inputs['Roughness'].default_value = 0.5000
    bsdf.inputs['IOR'].default_value = 1.4500
    bsdf.inputs['Alpha'].default_value = 1.0000
    bsdf.inputs['Transmission Weight'].default_value = 0.0000
    
    imported_obj.data.materials.append(bpy.data.materials['Material_001'])

    # effector settings
    mod = imported_obj.modifiers.new(name='Fluid', type='FLUID')
    mod.fluid_type = 'EFFECTOR'

    dst_settings = mod.effector_settings
    dst_settings.surface_distance = 0.5000
    dst_settings.use_plane_init = False

    # rigid body settings
    bpy.context.view_layer.objects.active = imported_obj
    imported_obj.select_set(True)
    bpy.ops.rigidbody.object_add()

    imported_obj.rigid_body.type = 'PASSIVE'
    imported_obj.rigid_body.mass = 1.0000
    imported_obj.rigid_body.collision_shape = 'MESH'

    ###################
    # Mesh: domain
    bpy.ops.wm.obj_import(filepath=os.path.join(base_path, 'domain.obj').replace('/', '\\'))
    imported_obj = bpy.context.selected_objects[0]
    imported_obj.name = 'domain'
    imported_obj.location = (0.0000, 0.0000, 0.5000)
    imported_obj.rotation_euler = (0.0000, 0.0000, 0.0000)
    imported_obj.scale = (1.4000, 1.4000, 1.6000)

    imported_obj.hide_render = False
    # Material: water
    mat = bpy.data.materials.new(name='water')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (0.5633, 0.7354, 0.8002, 1.0000)
    bsdf.inputs['Metallic'].default_value = 0.0000
    bsdf.inputs['Roughness'].default_value = 0.0000
    bsdf.inputs['IOR'].default_value = 1.3330
    bsdf.inputs['Alpha'].default_value = 1.0000
    bsdf.inputs['Transmission Weight'].default_value = 1.0000
    
    imported_obj.data.materials.append(bpy.data.materials['water'])

    # domain settings
    mod = imported_obj.modifiers.new(name='Fluid', type='FLUID')
    mod.fluid_type = 'DOMAIN'

    dst_settings = mod.domain_settings
    dst_settings.domain_type = 'LIQUID'
    dst_settings.cache_type = 'ALL'
    dst_settings.resolution_max = 96
    dst_settings.use_mesh = True
    dst_settings.cfl_condition = 2.0000
    dst_settings.particle_radius = 0.8000
    dst_settings.particle_band_width = 3.0000
    dst_settings.cache_frame_start = 1
    dst_settings.cache_frame_end = 170
    dst_settings.cache_directory = 'c:\\Users\\jason\\바탕 화면\\lecture\\시각적\\final\\dataset_gen\\scene\\splash\\fluid_cache'

    ###################
    # Mesh: water
    bpy.ops.wm.obj_import(filepath=os.path.join(base_path, 'water.obj').replace('/', '\\'))
    imported_obj = bpy.context.selected_objects[0]
    imported_obj.name = 'water'
    imported_obj.location = (0.0000, 0.0000, 0.1500)
    imported_obj.rotation_euler = (0.0000, 0.0000, 0.0000)
    imported_obj.scale = (0.8500, 0.8500, 0.8500)

    imported_obj.hide_render = True
    # flow settings
    mod = imported_obj.modifiers.new(name='Fluid', type='FLUID')
    mod.fluid_type = 'FLOW'

    dst_settings = mod.flow_settings
    dst_settings.flow_type = 'LIQUID'
    dst_settings.flow_behavior = 'GEOMETRY'
    dst_settings.flow_source = 'MESH'

    ###################
    # Mesh: Suzanne
    bpy.ops.wm.obj_import(filepath=os.path.join(base_path, 'Suzanne.obj').replace('/', '\\'))
    imported_obj = bpy.context.selected_objects[0]
    imported_obj.name = 'Suzanne'
    imported_obj.location = (0.0000, 0.0000, 5.0000)
    imported_obj.rotation_euler = (0.0000, 0.0000, 0.0000)
    imported_obj.scale = (0.3000, 0.3000, 0.3000)

    imported_obj.hide_render = False
    # effector settings
    mod = imported_obj.modifiers.new(name='Fluid', type='FLUID')
    mod.fluid_type = 'EFFECTOR'

    dst_settings = mod.effector_settings
    dst_settings.surface_distance = 0.0000
    dst_settings.use_plane_init = True

    # rigid body settings
    bpy.context.view_layer.objects.active = imported_obj
    imported_obj.select_set(True)
    bpy.ops.rigidbody.object_add()

    imported_obj.rigid_body.type = 'ACTIVE'
    imported_obj.rigid_body.mass = 3.0000
    imported_obj.rigid_body.collision_shape = 'MESH'

    imported_obj.rigid_body.kinematic = True
    imported_obj.keyframe_insert(data_path='rigid_body.kinematic', frame=0)
    start_move = sm
    imported_obj.rigid_body.kinematic = False
    imported_obj.keyframe_insert(data_path='rigid_body.kinematic', frame=start_move)

    ###################
    # Mesh: Plane
    bpy.ops.wm.obj_import(filepath=os.path.join(base_path, 'Plane.obj').replace('/', '\\'))
    imported_obj = bpy.context.selected_objects[0]
    imported_obj.name = 'Plane'
    imported_obj.location = (0.0000, 0.0000, -1.6800)
    imported_obj.rotation_euler = (0.0000, 0.0000, 0.0000)
    imported_obj.scale = (10.0000, 10.0000, 1.0000)

    imported_obj.hide_render = False
    # Material: Material_002
    mat = bpy.data.materials.new(name='Material_002')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (0.0132, 0.0000, 0.8003, 1.0000)
    bsdf.inputs['Metallic'].default_value = 1.0000
    bsdf.inputs['Roughness'].default_value = 0.0000
    bsdf.inputs['IOR'].default_value = 1.5000
    bsdf.inputs['Alpha'].default_value = 1.0000
    bsdf.inputs['Transmission Weight'].default_value = 0.0000
    
    imported_obj.data.materials.append(bpy.data.materials['Material_002'])

    # 2. lights
    # Light: Light
    light_data = bpy.data.lights.new(name='Light', type='SUN')
    light_data.energy = 15.0000
    light_data.color = (1.0000, 0.9476, 0.6701)
    light_data.angle = 0.1993
    light_obj = bpy.data.objects.new(name='Light', object_data=light_data)
    light_obj.location = (4.0762, 1.0055, 5.9039)
    light_obj.rotation_euler = (0.6503, 0.0552, 1.8664)
    bpy.context.collection.objects.link(light_obj)

    # 3. cameras
    # Camera: Camera
    cam_data = bpy.data.cameras.new(name='Camera')
    cam_data.lens = 50.0000
    cam_data.sensor_width = 36.0000
    cam_data.type = 'PERSP'
    cam_data.clip_start = 0.1000
    cam_data.clip_end = 100.0000
    cam_obj = bpy.data.objects.new('Camera', cam_data)
    cam_obj.location = (4.6989, -2.9958, 7.4383)
    cam_obj.rotation_euler = (0.6381, 0.0000, 0.9895)
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
    scene.frame_end = 170

    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    scene.render.filepath = 'C:\\Users\\jason\\바탕 화면\\lecture\\시각적\\final\\dataset_gen\\scene\\splash\\res\\'

parser = argparse.ArgumentParser()
parser.add_argument('--sm', type=float, default=90.0000)
args = sys.argv
if '--' in args:
    args = args[args.index('--') + 1:]
    p_args = parser.parse_args(args)
else:
    p_args = parser.parse_args('')

generate_scene(
sm=p_args.sm, 
)

bpy.ops.wm.save_mainfile(filepath='c:\\Users\\jason\\바탕 화면\\lecture\\시각적\\final\\dataset_gen\\scene\\splash\\scene_aug.blend')
exit()
