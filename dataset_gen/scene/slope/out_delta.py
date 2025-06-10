import os
import sys
import bpy
from mathutils import Vector
import argparse

def generate_scene(
    vis, 
    cam_loc, 
    cam_rot, 
    ):
    # Delete all existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    _fps = 30
    _fps_scale = _fps / 30
    bpy.context.scene.render.fps = _fps

    # 1. meshes
    _base_path = 'C:\\Users\\jason\\바탕 화면\\lecture\\시각적\\final\\snu-mlvu-project-2025\\dataset_gen\\scene\\slope\\meshes'

    ###################
    # Mesh: sink
    bpy.ops.wm.obj_import(filepath=os.path.join(_base_path, 'sink.obj').replace('/', '\\'))
    _imported_obj = bpy.context.selected_objects[0]
    _imported_obj.name = 'sink'
    _imported_obj.location = (0.0000, 0.0000, 0.0000)
    _imported_obj.rotation_euler = (0.0000, 0.0000, 0.0000)
    _imported_obj.scale = (1.0000, 1.0000, 1.0000)

    _imported_obj.hide_render = False
    # Material: Sink
    _mat = bpy.data.materials.new(name='Sink')
    _mat.use_nodes = True
    _nodes = _mat.node_tree.nodes

    _bsdf = _nodes.get('Principled BSDF')
    if _bsdf is None:
        _bsdf = _nodes.new(type='ShaderNodeBsdfPrincipled')
        _output = _nodes.get('Material Output') or _nodes.new(type='ShaderNodeOutputMaterial')
        _mat.node_tree.links.new(_bsdf.outputs['BSDF'], _output.inputs['Surface'])
    _mat_base_color = (0.2462, 0.4564, 0.7991, 1.0000)
    _bsdf.inputs['Base Color'].default_value = _mat_base_color
    _mat_Metallic = 0.0000
    _mat_Roughness = 0.5000
    _mat_IOR = 1.5000
    _mat_Alpha = 1.0000
    _mat_Transmission_Weight = 0.0000
    _bsdf.inputs['Metallic'].default_value = _mat_Metallic
    _bsdf.inputs['Roughness'].default_value = _mat_Roughness
    _bsdf.inputs['IOR'].default_value = _mat_IOR
    _bsdf.inputs['Alpha'].default_value = _mat_Alpha
    _bsdf.inputs['Transmission Weight'].default_value = _mat_Transmission_Weight
    
    _imported_obj.data.materials.append(bpy.data.materials['Sink'])

    # effector settings
    _mod = _imported_obj.modifiers.new(name='Fluid', type='FLUID')
    _mod.fluid_type = 'EFFECTOR'

    _dst_settings = _mod.effector_settings
    _dst_settings.surface_distance = 0.0000
    _dst_settings.use_plane_init = False

    # animation keyframes
    ###################
    # Mesh: slope
    bpy.ops.wm.obj_import(filepath=os.path.join(_base_path, 'slope.obj').replace('/', '\\'))
    _imported_obj = bpy.context.selected_objects[0]
    _imported_obj.name = 'slope'
    _imported_obj.location = (0.0000, 0.0000, 0.0000)
    _imported_obj.rotation_euler = (0.0000, -0.0914, -0.0000)
    _imported_obj.scale = (0.8570, 1.0000, 0.9989)

    _imported_obj.hide_render = False
    # Material: Sink2
    _mat = bpy.data.materials.new(name='Sink2')
    _mat.use_nodes = True
    _nodes = _mat.node_tree.nodes

    _bsdf = _nodes.get('Principled BSDF')
    if _bsdf is None:
        _bsdf = _nodes.new(type='ShaderNodeBsdfPrincipled')
        _output = _nodes.get('Material Output') or _nodes.new(type='ShaderNodeOutputMaterial')
        _mat.node_tree.links.new(_bsdf.outputs['BSDF'], _output.inputs['Surface'])
    _mat_base_color = (0.2462, 0.4564, 0.7991, 1.0000)
    _bsdf.inputs['Base Color'].default_value = _mat_base_color
    _mat_Metallic = 0.0000
    _mat_Roughness = 0.5000
    _mat_IOR = 1.5000
    _mat_Alpha = 1.0000
    _mat_Transmission_Weight = 0.0000
    _bsdf.inputs['Metallic'].default_value = _mat_Metallic
    _bsdf.inputs['Roughness'].default_value = _mat_Roughness
    _bsdf.inputs['IOR'].default_value = _mat_IOR
    _bsdf.inputs['Alpha'].default_value = _mat_Alpha
    _bsdf.inputs['Transmission Weight'].default_value = _mat_Transmission_Weight
    
    _imported_obj.data.materials.append(bpy.data.materials['Sink2'])

    # effector settings
    _mod = _imported_obj.modifiers.new(name='Fluid', type='FLUID')
    _mod.fluid_type = 'EFFECTOR'

    _dst_settings = _mod.effector_settings
    _dst_settings.surface_distance = 0.4000
    _dst_settings.use_plane_init = False

    ###################
    # Mesh: icosphere
    bpy.ops.wm.obj_import(filepath=os.path.join(_base_path, 'icosphere.obj').replace('/', '\\'))
    _imported_obj = bpy.context.selected_objects[0]
    _imported_obj.name = 'icosphere'
    _imported_obj.location = (1.7578, 0.0312, 0.5758)
    _imported_obj.rotation_euler = (0.0000, 0.0000, 0.0000)
    _imported_obj.scale = (0.8602, 0.8602, 0.8602)

    _imported_obj.hide_render = False
    # Material: Water
    _mat = bpy.data.materials.new(name='Water')
    _mat.use_nodes = True
    _nodes = _mat.node_tree.nodes

    _bsdf = _nodes.get('Principled BSDF')
    if _bsdf is None:
        _bsdf = _nodes.new(type='ShaderNodeBsdfPrincipled')
        _output = _nodes.get('Material Output') or _nodes.new(type='ShaderNodeOutputMaterial')
        _mat.node_tree.links.new(_bsdf.outputs['BSDF'], _output.inputs['Surface'])
    _mat_base_color = (0.8000, 0.8000, 0.8000, 1.0000)
    _bsdf.inputs['Base Color'].default_value = _mat_base_color
    _mat_Metallic = 0.0000
    _mat_Roughness = 0.0000
    _mat_IOR = 1.3330
    _mat_Alpha = 1.0000
    _mat_Transmission_Weight = 0.7180
    _bsdf.inputs['Metallic'].default_value = _mat_Metallic
    _bsdf.inputs['Roughness'].default_value = _mat_Roughness
    _bsdf.inputs['IOR'].default_value = _mat_IOR
    _bsdf.inputs['Alpha'].default_value = _mat_Alpha
    _bsdf.inputs['Transmission Weight'].default_value = _mat_Transmission_Weight
    
    _imported_obj.data.materials.append(bpy.data.materials['Water'])

    # flow settings
    _mod = _imported_obj.modifiers.new(name='Fluid', type='FLUID')
    _mod.fluid_type = 'FLOW'

    _dst_settings = _mod.flow_settings
    _dst_settings.flow_type = 'LIQUID'
    _dst_settings.flow_behavior = 'INFLOW'
    _dst_settings.flow_source = 'MESH'
    _dst_settings.use_initial_velocity = True
    _dst_settings.velocity_coord = (0.0000, 0.0000, 0.0000)

    ###################
    # Mesh: domain
    bpy.ops.wm.obj_import(filepath=os.path.join(_base_path, 'domain.obj').replace('/', '\\'))
    _imported_obj = bpy.context.selected_objects[0]
    _imported_obj.name = 'domain'
    _imported_obj.location = (0.0000, 0.0000, 0.0000)
    _imported_obj.rotation_euler = (0.0000, 0.0000, 0.0000)
    _imported_obj.scale = (1.0000, 1.0000, 1.0000)

    _imported_obj.hide_render = False
    # Material: Water
    _mat = bpy.data.materials.new(name='Water')
    _mat.use_nodes = True
    _nodes = _mat.node_tree.nodes

    _bsdf = _nodes.get('Principled BSDF')
    if _bsdf is None:
        _bsdf = _nodes.new(type='ShaderNodeBsdfPrincipled')
        _output = _nodes.get('Material Output') or _nodes.new(type='ShaderNodeOutputMaterial')
        _mat.node_tree.links.new(_bsdf.outputs['BSDF'], _output.inputs['Surface'])
    _mat_base_color = (0.8000, 0.8000, 0.8000, 1.0000)
    _bsdf.inputs['Base Color'].default_value = _mat_base_color
    _mat_Metallic = 0.0000
    _mat_Roughness = 0.0000
    _mat_IOR = 1.3330
    _mat_Alpha = 1.0000
    _mat_Transmission_Weight = 0.7180
    _bsdf.inputs['Metallic'].default_value = _mat_Metallic
    _bsdf.inputs['Roughness'].default_value = _mat_Roughness
    _bsdf.inputs['IOR'].default_value = _mat_IOR
    _bsdf.inputs['Alpha'].default_value = _mat_Alpha
    _bsdf.inputs['Transmission Weight'].default_value = _mat_Transmission_Weight
    
    _imported_obj.data.materials.append(bpy.data.materials['Water'])

    # domain settings
    _mod = _imported_obj.modifiers.new(name='Fluid', type='FLUID')
    _mod.fluid_type = 'DOMAIN'

    _dst_settings = _mod.domain_settings
    _dst_settings.domain_type = 'LIQUID'
    _dst_settings.cache_type = 'MODULAR'
    _dst_settings.cache_mesh_format = 'OBJECT'
    _dst_settings.cache_resumable = True
    _dst_settings.resolution_max = 80
    _dst_settings.use_mesh = True
    _dst_settings.cfl_condition = 4.0000
    _dst_settings.particle_radius = 0.8000
    _dst_settings.particle_number = 2
    _dst_settings.particle_randomness = 0.1000
    _dst_settings.particle_band_width = 2.2000
    _dst_settings.cache_frame_start = 1
    _dst_settings.cache_frame_end = 200
    _dst_settings.cache_frame_end = round(_dst_settings.cache_frame_end * _fps_scale)
    _dst_settings.cache_directory = '//fluid_cache/domain/'

    _viscosity_given = vis
    if _viscosity_given > -1.0:
        _dst_settings.use_viscosity = True
        _dst_settings.viscosity_value = max(_viscosity_given, 0.0)

    # 2. lights
    # Light: sun
    _light_data = bpy.data.lights.new(name='sun', type='SUN')
    _light_data.energy = 0.1000
    _light_data.color = (0.9123, 1.0000, 0.6875)
    _light_data.angle = 0.0092
    _light_obj = bpy.data.objects.new(name='sun', object_data=_light_data)
    _light_obj.location = (0.4465, 2.9227, 1.5639)
    _light_obj.rotation_euler = (-1.0621, 0.1769, -2.2585)
    bpy.context.collection.objects.link(_light_obj)

    # 3. cameras
    # Camera: Camera
    _cam_data = bpy.data.cameras.new(name='Camera')
    _cam_data.lens = 56.9998
    _cam_data.sensor_width = 36.0000
    _cam_data.type = 'PERSP'
    _cam_data.clip_start = 0.1000
    _cam_data.clip_end = 100.0000
    _cam_obj = bpy.data.objects.new('Camera', _cam_data)
    _cam_obj.location = cam_loc
    _cam_obj.rotation_euler = cam_rot
    bpy.context.collection.objects.link(_cam_obj)
    bpy.context.scene.camera = _cam_obj

    bpy.context.scene.render.engine = 'CYCLES'
    _cycles_sample = 64
    bpy.context.scene.cycles.samples = _cycles_sample
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.denoising_use_gpu = True

    _scene = bpy.context.scene
    _scene.render.image_settings.file_format = 'FFMPEG'
    _scene.render.ffmpeg.format = 'MKV'
    _scene.render.ffmpeg.codec = 'H264'
    _scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    _scene.render.ffmpeg.ffmpeg_preset = 'GOOD'
    _scene.render.ffmpeg.video_bitrate = 6000
    _scene.render.ffmpeg.gopsize = 18
    _frame_start = 20
    _frame_end = 170
    _frame_step = 1
    _scene.frame_start = _frame_start
    _scene.frame_end = round(_frame_end * _fps_scale)
    _scene.frame_step = _frame_step

    _world = bpy.context.scene.world
    _world.use_nodes = True
    _nodes = _world.node_tree.nodes
    _links = _world.node_tree.links
    _nodes.clear()
    _env_tex_node = _nodes.new(type='ShaderNodeTexEnvironment')
    _bg_node = _nodes.new(type='ShaderNodeBackground')
    _output_node = _nodes.new(type='ShaderNodeOutputWorld')

    texture_path = 'C:\\Users\\jason\\바탕 화면\\lecture\\시각적\\final\\snu-mlvu-project-2025\\dataset_gen\\scene\\slope\\texture/birchwood_1k.exr'
    _env_tex_node.image = bpy.data.images.load(texture_path)
    _links.new(_env_tex_node.outputs['Color'], _bg_node.inputs['Color'])
    _links.new(_bg_node.outputs['Background'], _output_node.inputs['Surface'])

    _scene.render.filepath = '/home\\lee\\Desktop\\Blender\\Flow\\Slope\\Frames\\'

parser = argparse.ArgumentParser()
parser.add_argument('--vis', type=lambda x: float(x[1:]), default=-1.0000)
parser.add_argument('--cam_loc', type=lambda x: tuple(map(float, x[1:].split(','))) , default=(4.9520, 7.0234, 1.4715))
parser.add_argument('--cam_rot', type=lambda x: tuple(map(float, x[1:].split(','))) , default=(1.4451, 0.0000, 2.4295))
args = sys.argv
if '--' in args:
    args = args[args.index('--') + 1:]
    p_args = parser.parse_args(args)
else:
    p_args = parser.parse_args('')

generate_scene(
vis=p_args.vis, 
cam_loc=p_args.cam_loc, 
cam_rot=p_args.cam_rot, 
)

bpy.ops.wm.save_mainfile(filepath='C:\\Users\\jason\\바탕 화면\\lecture\\시각적\\final\\snu-mlvu-project-2025\\dataset_gen\\scene\\slope\\scene_aug.blend')
exit()
