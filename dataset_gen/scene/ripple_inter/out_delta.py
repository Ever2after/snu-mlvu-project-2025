import os
import sys
import bpy
from mathutils import Vector
import argparse

def generate_scene(
    cam_loc, 
    cam_rot, 
    viscosity_i, 
    part_rad_i, 
    part_num_i, 
    part_random_i, 
    ts_min_i, 
    flow_init_v, 
    viscosity_r, 
    part_rad_r, 
    part_num_r, 
    part_random_r, 
    ts_min_r, 
    light_angle, 
    sphere_loc, 
    cube_loc, 
    cube_rot, 
    ripple_size, 
    water_color_i, 
    water_color_r, 
    samples, 
    step, 
    ):
    # Delete all existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    _fps = 30
    _fps_scale = _fps / 30
    bpy.context.scene.render.fps = _fps

    # 1. meshes
    _base_path = 'C:\\Users\\jason\\바탕 화면\\lecture\\시각적\\final\\snu-mlvu-project-2025\\dataset_gen\\scene\\ripple_inter\\meshes'

    ###################
    # Mesh: Icosphere_r
    bpy.ops.wm.obj_import(filepath=os.path.join(_base_path, 'Icosphere_r.obj').replace('/', '\\'))
    _imported_obj = bpy.context.selected_objects[0]
    _imported_obj.name = 'Icosphere_r'
    _imported_obj.location = (2.3610, 0.2221, 0.7336)
    _imported_obj.rotation_euler = (0.0000, 0.0000, 0.0000)
    _imported_obj.scale = ripple_size

    _imported_obj.hide_render = False
    # Material: Water_r
    _mat = bpy.data.materials.new(name='Water_r')
    _mat.use_nodes = True
    _nodes = _mat.node_tree.nodes

    _bsdf = _nodes.get('Principled BSDF')
    if _bsdf is None:
        _bsdf = _nodes.new(type='ShaderNodeBsdfPrincipled')
        _output = _nodes.get('Material Output') or _nodes.new(type='ShaderNodeOutputMaterial')
        _mat.node_tree.links.new(_bsdf.outputs['BSDF'], _output.inputs['Surface'])
    _mat_base_color = water_color_r
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
    
    _imported_obj.data.materials.append(bpy.data.materials['Water_r'])

    # flow settings
    _mod = _imported_obj.modifiers.new(name='Fluid', type='FLUID')
    _mod.fluid_type = 'FLOW'

    _dst_settings = _mod.flow_settings
    _dst_settings.flow_type = 'LIQUID'
    _dst_settings.flow_behavior = 'INFLOW'
    _dst_settings.flow_source = 'MESH'
    _dst_settings.use_initial_velocity = True
    _dst_settings.velocity_coord = (0.0000, 0.0000, 0.0000)

    _stop_flow_fra = 66.0000

    if _stop_flow_fra > -1: 
        bpy.context.scene.frame_set(1)
        _dst_settings.use_inflow = 1
        _imported_obj.keyframe_insert(data_path='modifiers["Fluid"].flow_settings.use_inflow', frame=1)

        _kf_new_scene = round((_stop_flow_fra - 1) * _fps_scale)
        bpy.context.scene.frame_set(_kf_new_scene)
        _dst_settings.use_inflow = 1
        _imported_obj.keyframe_insert(data_path='modifiers["Fluid"].flow_settings.use_inflow', frame=_kf_new_scene)

        _kf_new_scene = round(_stop_flow_fra * _fps_scale)
        bpy.context.scene.frame_set(_kf_new_scene)
        _dst_settings.use_inflow = 0
        _imported_obj.keyframe_insert(data_path='modifiers["Fluid"].flow_settings.use_inflow', frame=_kf_new_scene)

    # animation keyframes
    ###################
    # Mesh: Domain_r
    bpy.ops.wm.obj_import(filepath=os.path.join(_base_path, 'Domain_r.obj').replace('/', '\\'))
    _imported_obj = bpy.context.selected_objects[0]
    _imported_obj.name = 'Domain_r'
    _imported_obj.location = (2.4034, 0.2277, 0.1958)
    _imported_obj.rotation_euler = (0.0000, 0.0000, 0.0000)
    _imported_obj.scale = (1.5123, 1.5123, 1.5123)

    _imported_obj.hide_render = False
    # Material: Water_r
    _mat = bpy.data.materials.new(name='Water_r')
    _mat.use_nodes = True
    _nodes = _mat.node_tree.nodes

    _bsdf = _nodes.get('Principled BSDF')
    if _bsdf is None:
        _bsdf = _nodes.new(type='ShaderNodeBsdfPrincipled')
        _output = _nodes.get('Material Output') or _nodes.new(type='ShaderNodeOutputMaterial')
        _mat.node_tree.links.new(_bsdf.outputs['BSDF'], _output.inputs['Surface'])
    _mat_base_color = water_color_r
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
    
    _imported_obj.data.materials.append(bpy.data.materials['Water_r'])

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
    _dst_settings.timesteps_min = ts_min_r
    _dst_settings.timesteps_max = 4
    _dst_settings.particle_radius = part_rad_r
    _dst_settings.particle_number = part_num_r
    _dst_settings.particle_randomness = part_random_r
    _dst_settings.particle_band_width = 2.2000
    _dst_settings.cache_frame_start = 1
    _dst_settings.cache_frame_end = 150
    _dst_settings.cache_frame_end = round(_dst_settings.cache_frame_end * _fps_scale)
    _dst_settings.cache_directory = '//fluid_cache/Domain_r/'

    _viscosity_given = viscosity_r
    if _viscosity_given > -1.0:
        _dst_settings.use_viscosity = True
        _dst_settings.viscosity_value = max(_viscosity_given, 0.0)

    ###################
    # Mesh: Plane_r
    bpy.ops.wm.obj_import(filepath=os.path.join(_base_path, 'Plane_r.obj').replace('/', '\\'))
    _imported_obj = bpy.context.selected_objects[0]
    _imported_obj.name = 'Plane_r'
    _imported_obj.location = (2.3556, 0.2277, 0.1643)
    _imported_obj.rotation_euler = (0.0000, 0.0000, 0.0000)
    _imported_obj.scale = (1.5123, 1.5123, 1.5123)

    _imported_obj.hide_render = False
    # Material: Water_r
    _mat = bpy.data.materials.new(name='Water_r')
    _mat.use_nodes = True
    _nodes = _mat.node_tree.nodes

    _bsdf = _nodes.get('Principled BSDF')
    if _bsdf is None:
        _bsdf = _nodes.new(type='ShaderNodeBsdfPrincipled')
        _output = _nodes.get('Material Output') or _nodes.new(type='ShaderNodeOutputMaterial')
        _mat.node_tree.links.new(_bsdf.outputs['BSDF'], _output.inputs['Surface'])
    _mat_base_color = water_color_r
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
    
    _imported_obj.data.materials.append(bpy.data.materials['Water_r'])

    # flow settings
    _mod = _imported_obj.modifiers.new(name='Fluid', type='FLUID')
    _mod.fluid_type = 'FLOW'

    _dst_settings = _mod.flow_settings
    _dst_settings.flow_type = 'LIQUID'
    _dst_settings.flow_behavior = 'GEOMETRY'
    _dst_settings.flow_source = 'MESH'
    _dst_settings.use_initial_velocity = True
    _dst_settings.velocity_coord = (0.0000, 0.0000, 0.0000)

    ###################
    # Mesh: Sphere_i
    bpy.ops.wm.obj_import(filepath=os.path.join(_base_path, 'Sphere_i.obj').replace('/', '\\'))
    _imported_obj = bpy.context.selected_objects[0]
    _imported_obj.name = 'Sphere_i'
    _imported_obj.location = sphere_loc
    _imported_obj.rotation_euler = (0.0000, 0.0000, 0.0000)
    _imported_obj.scale = (0.2010, 0.2010, 0.2010)

    _imported_obj.hide_render = False
    # effector settings
    _mod = _imported_obj.modifiers.new(name='Fluid', type='FLUID')
    _mod.fluid_type = 'EFFECTOR'

    _dst_settings = _mod.effector_settings
    _dst_settings.surface_distance = 0.2000
    _dst_settings.use_plane_init = False

    ###################
    # Mesh: Plane_i
    bpy.ops.wm.obj_import(filepath=os.path.join(_base_path, 'Plane_i.obj').replace('/', '\\'))
    _imported_obj = bpy.context.selected_objects[0]
    _imported_obj.name = 'Plane_i'
    _imported_obj.location = (0.0795, 0.2095, 0.1726)
    _imported_obj.rotation_euler = (0.0000, 0.0000, 0.0000)
    _imported_obj.scale = (0.2010, 0.2010, 0.2010)

    _imported_obj.hide_render = False
    # flow settings
    _mod = _imported_obj.modifiers.new(name='Fluid', type='FLUID')
    _mod.fluid_type = 'FLOW'

    _dst_settings = _mod.flow_settings
    _dst_settings.flow_type = 'LIQUID'
    _dst_settings.flow_behavior = 'INFLOW'
    _dst_settings.flow_source = 'MESH'
    _dst_settings.use_initial_velocity = True
    _dst_settings.velocity_coord = flow_init_v

    _stop_flow_fra = 61.0000

    if _stop_flow_fra > -1: 
        bpy.context.scene.frame_set(1)
        _dst_settings.use_inflow = 1
        _imported_obj.keyframe_insert(data_path='modifiers["Fluid"].flow_settings.use_inflow', frame=1)

        _kf_new_scene = round((_stop_flow_fra - 1) * _fps_scale)
        bpy.context.scene.frame_set(_kf_new_scene)
        _dst_settings.use_inflow = 1
        _imported_obj.keyframe_insert(data_path='modifiers["Fluid"].flow_settings.use_inflow', frame=_kf_new_scene)

        _kf_new_scene = round(_stop_flow_fra * _fps_scale)
        bpy.context.scene.frame_set(_kf_new_scene)
        _dst_settings.use_inflow = 0
        _imported_obj.keyframe_insert(data_path='modifiers["Fluid"].flow_settings.use_inflow', frame=_kf_new_scene)

    # animation keyframes
    ###################
    # Mesh: Domain_i
    bpy.ops.wm.obj_import(filepath=os.path.join(_base_path, 'Domain_i.obj').replace('/', '\\'))
    _imported_obj = bpy.context.selected_objects[0]
    _imported_obj.name = 'Domain_i'
    _imported_obj.location = (0.0795, 0.2095, 0.1726)
    _imported_obj.rotation_euler = (0.0000, 0.0000, 0.0000)
    _imported_obj.scale = (0.2010, 0.2010, 0.2010)

    _imported_obj.hide_render = False
    # Material: Water_i
    _mat = bpy.data.materials.new(name='Water_i')
    _mat.use_nodes = True
    _nodes = _mat.node_tree.nodes

    _bsdf = _nodes.get('Principled BSDF')
    if _bsdf is None:
        _bsdf = _nodes.new(type='ShaderNodeBsdfPrincipled')
        _output = _nodes.get('Material Output') or _nodes.new(type='ShaderNodeOutputMaterial')
        _mat.node_tree.links.new(_bsdf.outputs['BSDF'], _output.inputs['Surface'])
    _mat_base_color = water_color_i
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
    
    _imported_obj.data.materials.append(bpy.data.materials['Water_i'])

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
    _dst_settings.timesteps_min = ts_min_i
    _dst_settings.timesteps_max = 4
    _dst_settings.particle_radius = part_rad_i
    _dst_settings.particle_number = part_num_i
    _dst_settings.particle_randomness = part_random_i
    _dst_settings.particle_band_width = 2.2000
    _dst_settings.cache_frame_start = 1
    _dst_settings.cache_frame_end = 150
    _dst_settings.cache_frame_end = round(_dst_settings.cache_frame_end * _fps_scale)
    _dst_settings.cache_directory = '//fluid_cache/Domain_i/'

    _viscosity_given = viscosity_i
    if _viscosity_given > -1.0:
        _dst_settings.use_viscosity = True
        _dst_settings.viscosity_value = max(_viscosity_given, 0.0)

    ###################
    # Mesh: Cube_i
    bpy.ops.wm.obj_import(filepath=os.path.join(_base_path, 'Cube_i.obj').replace('/', '\\'))
    _imported_obj = bpy.context.selected_objects[0]
    _imported_obj.name = 'Cube_i'
    _imported_obj.location = cube_loc
    _imported_obj.rotation_euler = cube_rot
    _imported_obj.scale = (0.2010, 0.2010, 0.2010)

    _imported_obj.hide_render = False
    # effector settings
    _mod = _imported_obj.modifiers.new(name='Fluid', type='FLUID')
    _mod.fluid_type = 'EFFECTOR'

    _dst_settings = _mod.effector_settings
    _dst_settings.surface_distance = 0.2000
    _dst_settings.use_plane_init = False

    # 2. lights
    # Light: Sun
    _light_data = bpy.data.lights.new(name='Sun', type='SUN')
    _light_data.energy = 8.0000
    _light_data.color = (0.9123, 1.0000, 0.6875)
    _light_data.angle = 0.0092
    _light_obj = bpy.data.objects.new(name='Sun', object_data=_light_data)
    _light_obj.location = (0.1796, 0.4731, 0.3897)
    _light_obj.rotation_euler = light_angle
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
    _cycles_sample = samples
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
    _frame_start = 1
    _frame_end = 150
    _frame_step = step
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

    texture_path = 'C:\\Users\\jason\\바탕 화면\\lecture\\시각적\\final\\snu-mlvu-project-2025\\dataset_gen\\scene\\ripple_inter\\texture/birchwood_1k.exr'
    _env_tex_node.image = bpy.data.images.load(texture_path)
    _links.new(_env_tex_node.outputs['Color'], _bg_node.inputs['Color'])
    _links.new(_bg_node.outputs['Background'], _output_node.inputs['Surface'])

    _scene.render.filepath = '/home\\lee\\Desktop\\Blender\\Flow\\Slope\\Frames\\'

parser = argparse.ArgumentParser()
parser.add_argument('--ripple_size', type=lambda x: tuple(map(float, x[1:].split(','))) , default=(3.1000, 3.1000, 3.1000))
parser.add_argument('--water_color_r', type=lambda x: tuple(map(float, x[1:].split(','))) , default=(0.8000, 0.8000, 0.8000, 1.0000))
parser.add_argument('--ts_min_r', type=lambda x: int(x[1:]), default=1)
parser.add_argument('--part_rad_r', type=lambda x: float(x[1:]), default=0.8000)
parser.add_argument('--part_num_r', type=lambda x: int(x[1:]), default=2)
parser.add_argument('--part_random_r', type=lambda x: float(x[1:]), default=0.1000)
parser.add_argument('--viscosity_r', type=lambda x: float(x[1:]), default=-1.0000)
parser.add_argument('--sphere_loc', type=lambda x: tuple(map(float, x[1:].split(','))) , default=(0.0795, 0.2095, 0.1726))
parser.add_argument('--flow_init_v', type=lambda x: tuple(map(float, x[1:].split(','))) , default=(0.0000, 0.0000, 0.0000))
parser.add_argument('--water_color_i', type=lambda x: tuple(map(float, x[1:].split(','))) , default=(0.8000, 0.8000, 0.8000, 1.0000))
parser.add_argument('--ts_min_i', type=lambda x: int(x[1:]), default=1)
parser.add_argument('--part_rad_i', type=lambda x: float(x[1:]), default=0.8000)
parser.add_argument('--part_num_i', type=lambda x: int(x[1:]), default=2)
parser.add_argument('--part_random_i', type=lambda x: float(x[1:]), default=0.1000)
parser.add_argument('--viscosity_i', type=lambda x: float(x[1:]), default=-1.0000)
parser.add_argument('--cube_loc', type=lambda x: tuple(map(float, x[1:].split(','))) , default=(0.0795, 0.2095, 0.1726))
parser.add_argument('--cube_rot', type=lambda x: tuple(map(float, x[1:].split(','))) , default=(0.0000, 0.0000, 0.0000))
parser.add_argument('--light_angle', type=lambda x: tuple(map(float, x[1:].split(','))) , default=(1.8718, -3.1220, -4.1463))
parser.add_argument('--cam_loc', type=lambda x: tuple(map(float, x[1:].split(','))) , default=(1.4327, -6.5218, 1.9709))
parser.add_argument('--cam_rot', type=lambda x: tuple(map(float, x[1:].split(','))) , default=(1.3653, 0.0000, 0.0215))
parser.add_argument('--samples', type=lambda x: int(x[1:]), default=64)
parser.add_argument('--step', type=lambda x: int(x[1:]), default=1)
args = sys.argv
if '--' in args:
    args = args[args.index('--') + 1:]
    p_args = parser.parse_args(args)
else:
    p_args = parser.parse_args('')

generate_scene(
ripple_size=p_args.ripple_size, 
water_color_r=p_args.water_color_r, 
ts_min_r=p_args.ts_min_r, 
part_rad_r=p_args.part_rad_r, 
part_num_r=p_args.part_num_r, 
part_random_r=p_args.part_random_r, 
viscosity_r=p_args.viscosity_r, 
sphere_loc=p_args.sphere_loc, 
flow_init_v=p_args.flow_init_v, 
water_color_i=p_args.water_color_i, 
ts_min_i=p_args.ts_min_i, 
part_rad_i=p_args.part_rad_i, 
part_num_i=p_args.part_num_i, 
part_random_i=p_args.part_random_i, 
viscosity_i=p_args.viscosity_i, 
cube_loc=p_args.cube_loc, 
cube_rot=p_args.cube_rot, 
light_angle=p_args.light_angle, 
cam_loc=p_args.cam_loc, 
cam_rot=p_args.cam_rot, 
samples=p_args.samples, 
step=p_args.step, 
)

bpy.ops.wm.save_mainfile(filepath='C:\\Users\\jason\\바탕 화면\\lecture\\시각적\\final\\snu-mlvu-project-2025\\dataset_gen\\scene\\ripple_inter\\scene_aug.blend')
exit()
