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
    base_path = 'c:\\Users\\jason\\바탕 화면\\lecture\\시각적\\final\\dataset_gen\\scene\\water_slope\\meshes'

    ###################
    # Mesh: sink
    bpy.ops.wm.obj_import(filepath=os.path.join(base_path, 'sink.obj').replace('/', '\\'))
    imported_obj = bpy.context.selected_objects[0]
    imported_obj.name = 'sink'
    imported_obj.location = (0.0000, 0.0000, 0.0000)
    imported_obj.rotation_euler = (0.0000, 0.0000, 0.0000)
    imported_obj.scale = (1.0000, 1.0000, 1.0000)

    imported_obj.hide_render = False
    # Material: Sink
    mat = bpy.data.materials.new(name='Sink')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (0.2462, 0.4564, 0.7991, 1.0000)
    bsdf.inputs['Metallic'].default_value = 0.0000
    bsdf.inputs['Roughness'].default_value = 0.5000
    bsdf.inputs['IOR'].default_value = 1.5000
    bsdf.inputs['Alpha'].default_value = 1.0000
    bsdf.inputs['Transmission Weight'].default_value = 0.0000
    
    imported_obj.data.materials.append(bpy.data.materials['Sink'])

    # effector settings
    mod = imported_obj.modifiers.new(name='Fluid', type='FLUID')
    mod.fluid_type = 'EFFECTOR'

    dst_settings = mod.effector_settings
    dst_settings.surface_distance = 0.0000
    dst_settings.use_plane_init = False

    ###################
    # Mesh: slope
    bpy.ops.wm.obj_import(filepath=os.path.join(base_path, 'slope.obj').replace('/', '\\'))
    imported_obj = bpy.context.selected_objects[0]
    imported_obj.name = 'slope'
    imported_obj.location = (-1.5472, 0.0000, 0.4729)
    imported_obj.rotation_euler = (0.0000, 0.0000, 0.0000)
    imported_obj.scale = (1.0000, 1.2421, 1.0000)

    imported_obj.hide_render = False
    # Material: Sink2
    mat = bpy.data.materials.new(name='Sink2')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (0.2462, 0.4564, 0.7991, 1.0000)
    bsdf.inputs['Metallic'].default_value = 0.0000
    bsdf.inputs['Roughness'].default_value = 0.5000
    bsdf.inputs['IOR'].default_value = 1.5000
    bsdf.inputs['Alpha'].default_value = 1.0000
    bsdf.inputs['Transmission Weight'].default_value = 0.0000
    
    imported_obj.data.materials.append(bpy.data.materials['Sink2'])

    # effector settings
    mod = imported_obj.modifiers.new(name='Fluid', type='FLUID')
    mod.fluid_type = 'EFFECTOR'

    dst_settings = mod.effector_settings
    dst_settings.surface_distance = 0.0000
    dst_settings.use_plane_init = False

    ###################
    # Mesh: Plane
    bpy.ops.wm.obj_import(filepath=os.path.join(base_path, 'Plane.obj').replace('/', '\\'))
    imported_obj = bpy.context.selected_objects[0]
    imported_obj.name = 'Plane'
    imported_obj.location = (0.8983, 0.2741, -1.9290)
    imported_obj.rotation_euler = (0.0000, 0.0000, 0.0000)
    imported_obj.scale = (38.3832, 38.3832, 38.3832)

    imported_obj.hide_render = False
    # Material: Background
    mat = bpy.data.materials.new(name='Background')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (0.0000, 0.0000, 0.0000, 1.0000)
    bsdf.inputs['Metallic'].default_value = 0.0000
    bsdf.inputs['Roughness'].default_value = 0.5000
    bsdf.inputs['IOR'].default_value = 1.5000
    bsdf.inputs['Alpha'].default_value = 1.0000
    bsdf.inputs['Transmission Weight'].default_value = 0.0000
    
    imported_obj.data.materials.append(bpy.data.materials['Background'])

    ###################
    # Mesh: source
    bpy.ops.wm.obj_import(filepath=os.path.join(base_path, 'source.obj').replace('/', '\\'))
    imported_obj = bpy.context.selected_objects[0]
    imported_obj.name = 'source'
    imported_obj.location = (-2.3645, -0.0117, 1.9305)
    imported_obj.rotation_euler = (0.0000, 0.0000, 0.0000)
    imported_obj.scale = (-3.2947, -3.2947, -3.2947)

    imported_obj.hide_render = False
    # Material: Water
    mat = bpy.data.materials.new(name='Water')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (0.8000, 0.8000, 0.8000, 1.0000)
    bsdf.inputs['Metallic'].default_value = 0.0000
    bsdf.inputs['Roughness'].default_value = 0.0000
    bsdf.inputs['IOR'].default_value = 1.3330
    bsdf.inputs['Alpha'].default_value = 1.0000
    bsdf.inputs['Transmission Weight'].default_value = 1.0000
    
    imported_obj.data.materials.append(bpy.data.materials['Water'])

    # flow settings
    mod = imported_obj.modifiers.new(name='Fluid', type='FLUID')
    mod.fluid_type = 'FLOW'

    dst_settings = mod.flow_settings
    dst_settings.flow_type = 'LIQUID'
    dst_settings.flow_behavior = 'INFLOW'
    dst_settings.flow_source = 'MESH'

    ###################
    # Mesh: domain
    bpy.ops.wm.obj_import(filepath=os.path.join(base_path, 'domain.obj').replace('/', '\\'))
    imported_obj = bpy.context.selected_objects[0]
    imported_obj.name = 'domain'
    imported_obj.location = (-0.1822, -0.0101, 1.1576)
    imported_obj.rotation_euler = (0.0000, 0.0000, 0.0000)
    imported_obj.scale = (1.0000, 1.0000, 1.0000)

    imported_obj.hide_render = False
    # Material: Water
    mat = bpy.data.materials.new(name='Water')
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (0.8000, 0.8000, 0.8000, 1.0000)
    bsdf.inputs['Metallic'].default_value = 0.0000
    bsdf.inputs['Roughness'].default_value = 0.0000
    bsdf.inputs['IOR'].default_value = 1.3330
    bsdf.inputs['Alpha'].default_value = 1.0000
    bsdf.inputs['Transmission Weight'].default_value = 1.0000
    
    imported_obj.data.materials.append(bpy.data.materials['Water'])

    # domain settings
    mod = imported_obj.modifiers.new(name='Fluid', type='FLUID')
    mod.fluid_type = 'DOMAIN'

    dst_settings = mod.domain_settings
    dst_settings.domain_type = 'LIQUID'
    dst_settings.cache_type = 'ALL'
    dst_settings.resolution_max = sim_res
    dst_settings.use_mesh = True
    dst_settings.cfl_condition = 4.0000
    dst_settings.particle_radius = 0.8000
    dst_settings.particle_band_width = 2.2000
    dst_settings.cache_frame_start = 1
    dst_settings.cache_frame_end = 200
    dst_settings.cache_directory = 'c:\\Users\\jason\\바탕 화면\\lecture\\시각적\\final\\dataset_gen\\scene\\water_slope\\fluid_cache'

    # 2. lights
    # Light: Sun
    light_data = bpy.data.lights.new(name='Sun', type='SUN')
    light_data.energy = 5.0000
    light_data.color = (0.9123, 1.0000, 0.6875)
    light_data.angle = 0.0092
    light_obj = bpy.data.objects.new(name='Sun', object_data=light_data)
    light_obj.location = (1.0930, 1.7728, 0.8459)
    light_obj.rotation_euler = (0.9385, 0.7829, 0.6483)
    bpy.context.collection.objects.link(light_obj)

    # 3. cameras
    # Camera: Camera
    cam_data = bpy.data.cameras.new(name='Camera')
    cam_data.lens = 56.9998
    cam_data.sensor_width = 36.0000
    cam_data.type = 'PERSP'
    cam_data.clip_start = 0.1000
    cam_data.clip_end = 100.0000
    cam_obj = bpy.data.objects.new('Camera', cam_data)
    cam_obj.location = (2.7967, -3.1037, 2.4991)
    cam_obj.rotation_euler = (1.1093, 0.0000, 0.8149)
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
    scene.frame_end = 200

    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    env_tex_node = nodes.new(type='ShaderNodeTexEnvironment')
    bg_node = nodes.new(type='ShaderNodeBackground')
    output_node = nodes.new(type='ShaderNodeOutputWorld')

    texture_path = 'c:\\Users\\jason\\바탕 화면\\lecture\\시각적\\final\\dataset_gen\\scene\\water_slope\\texture\\birchwood_1k.exr'
    env_tex_node.image = bpy.data.images.load(texture_path)
    links.new(env_tex_node.outputs['Color'], bg_node.inputs['Color'])
    links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])

    scene.render.filepath = 'C:\\Users\\jason\\바탕 화면\\lecture\\시각적\\final\\water_slope\\'

parser = argparse.ArgumentParser()
parser.add_argument('--sim_res', type=int, default=384)
args = sys.argv
if '--' in args:
    args = args[args.index('--') + 1:]
    p_args = parser.parse_args(args)
else:
    p_args = parser.parse_args('')

generate_scene(
sim_res=p_args.sim_res, 
)

bpy.ops.wm.save_mainfile(filepath='c:\\Users\\jason\\바탕 화면\\lecture\\시각적\\final\\dataset_gen\\scene\\water_slope\\scene_aug.blend')
exit()
