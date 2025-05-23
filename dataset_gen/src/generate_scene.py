import bpy
import os
import sys
import io
import argparse
import importlib.util
from contextlib import redirect_stdout
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))
#sys.path.append("C:\\Users\\jason\\바탕 화면\\lecture\\시각적\\final")
#import generate_scene_conf as conf

def load_vars(conf_path):
    if not os.path.exists(conf_path):
        with open(conf_path, 'w') as f:
            f.write("#######################################################################################\n")
            f.write("##### for scene generation #####\n")
            f.write("what_to_change = {\n")
            f.write("}\n\n\n")
            f.write("#######################################################################################\n")
            f.write("##### for variable sampling #####\n")

    module_name = "loaded_conf_module"

    spec = importlib.util.spec_from_file_location(module_name, conf_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    what_to_change = getattr(module, "what_to_change", None)

    return what_to_change


def print_var(object):
    if isinstance(object, tuple):
        tuple_str = "("
        for v in object:
            tuple_str += f"{v:.4f}, "
        tuple_str = tuple_str[:-2] # remove last ", "
        tuple_str += ")"
        res = tuple_str
    elif isinstance(object, str):
        res = f"'{object}'"
    elif isinstance(object, int):
        res = f"{object}"
    else:
        res = f"{object:.4f}"
    
    return res


def chkvar(top_abbr, var, value):
    """
    checks whether the given variable is modifiable
    and outputs the resulting line accordingly
    
    ex)
    top_abbr = "sink"
    var = "imported_obj.location"
    value = (0.7, 1.1, 1.2) <tuple>
    """
    if "." in var:
        var_property = var[var.find("."):] # ".location"
    else:
        var_property = ""
    var_path = top_abbr + var_property # "sink.location"
    if var_path in what_to_change.keys():
        res = f"{var} = {what_to_change[var_path]}\n"
        default_values[what_to_change[var_path]] = value
    else:
        value_str = print_var(value)
        res = f"{var} = {value_str}\n"
        
    return res


def save_geometry(mesh):
    original_scene = bpy.context.scene
    mesh_name = mesh.name

    # Step 1: Create a new scene
    temp_scene = bpy.data.scenes.new(name="TEMP_EXPORT_SCENE")
    
    # Step 2: Duplicate the object and remove all other objects from temp scene
    obj_copy = mesh.copy()
    obj_copy.data = mesh.data.copy()  # Ensure a full duplicate of mesh data
    temp_scene.collection.objects.link(obj_copy)
    
    obj_copy.location = (0.0, 0.0, 0.0)
    obj_copy.rotation_euler = (0.0, 0.0, 0.0)
    obj_copy.scale = (1.0, 1.0, 1.0)

    # Step 3: Switch to the new scene
    bpy.context.window.scene = temp_scene

    # remove all modifiers
    for mod in obj_copy.modifiers:
        obj_copy.modifiers.remove(mod)

    # Step 4: Export the mesh
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        mesh_path = os.path.join(mesh_dir, f"{mesh_name}.obj").replace("/", "\\")
        bpy.ops.wm.obj_export(
            filepath=mesh_path,
            export_selected_objects=False,
            export_animation=False,
            export_materials=False,
            export_uv=False,
            path_mode='ABSOLUTE',
            forward_axis='Y',
            up_axis='Z'
        )

    # Step 5: Clean up
    bpy.data.scenes.remove(temp_scene)
    bpy.context.window.scene = original_scene


def output_mesh_mat(mats, indent=0):
    res = ""
    i_str = "    " * indent

    for mat in mats:
        mat = mat.material
        bsdf = None
        for node in mat.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                bsdf = node
                break

        if not bsdf:
            continue  # Skip non-principled materials
        
        mat_name = mat.name.replace(".", "_")

        res += f"{i_str}# Material: {mat_name}\n"
        res += f"{i_str}mat = bpy.data.materials.new(name='{mat_name}')\n"
        res += f"{i_str}mat.use_nodes = True\n"
        res += f"{i_str}nodes = mat.node_tree.nodes\n"
        res += f"{i_str}bsdf = nodes.get('Principled BSDF')\n"

        # Albedo
        base_color = bsdf.inputs['Base Color'].default_value
        top = f"M@{mat_name}"
        res += f"{i_str}" + chkvar(top, "bsdf.inputs['Base Color'].default_value", tuple(base_color))

        # Metallic, Roughness, IOR, Alpha, Transmission Weight
        for mat_attr in ['Metallic', 'Roughness', 'IOR', 'Alpha', 'Transmission Weight']:
            value = bsdf.inputs[mat_attr].default_value
            res += f"{i_str}" + chkvar(top, f"bsdf.inputs['{mat_attr}'].default_value", value)

        res += f"{i_str}\n"
        res += f"{i_str}imported_obj.data.materials.append(bpy.data.materials['{mat_name}'])\n\n"

    return res


def output_fluid_domain(mesh_name, name, settings, indent=0):
    res = ""
    i_str = "    " * indent
    
    # Add a fluid modifier to the target
    res += f"{i_str}mod = imported_obj.modifiers.new(name='{name}', type='FLUID')\n"
    res += f"{i_str}mod.fluid_type = 'DOMAIN'\n\n"

    # Copy key domain properties
    res += f"{i_str}dst_settings = mod.domain_settings\n"

    top = f"{mesh_name}.DOMAIN"
    res += f"{i_str}" + chkvar(top, "dst_settings.domain_type", settings.domain_type)
    res += f"{i_str}" + chkvar(top, "dst_settings.cache_type", settings.cache_type)
    res += f"{i_str}" + chkvar(top, "dst_settings.resolution_max", settings.resolution_max)
    res += f"{i_str}" + chkvar(top, "dst_settings.use_mesh", settings.use_mesh)
    res += f"{i_str}" + chkvar(top, "dst_settings.cfl_condition", settings.cfl_condition)
    res += f"{i_str}" + chkvar(top, "dst_settings.particle_radius", settings.particle_radius)
    res += f"{i_str}" + chkvar(top, "dst_settings.particle_band_width", settings.particle_band_width)
    res += f"{i_str}" + chkvar(top, "dst_settings.cache_frame_start", settings.cache_frame_start)
    res += f"{i_str}" + chkvar(top, "dst_settings.cache_frame_end", settings.cache_frame_end)
    cache_path = cache_dir.replace('\\', '\\\\')
    res += f"{i_str}dst_settings.cache_directory = '{cache_path}'\n\n"
    
    return res


def output_fluid_flow(mesh_name, name, settings, indent=0):
    res = ""
    i_str = "    " * indent

    res += f"{i_str}mod = imported_obj.modifiers.new(name='{name}', type='FLUID')\n"
    res += f"{i_str}mod.fluid_type = 'FLOW'\n\n"
    
    # Copy key flow properties
    res += f"{i_str}dst_settings = mod.flow_settings\n"
    
    top = f"{mesh_name}.FLOW"
    res += f"{i_str}" + chkvar(top, "dst_settings.flow_type", settings.flow_type)
    res += f"{i_str}" + chkvar(top, "dst_settings.flow_behavior", settings.flow_behavior)
    res += f"{i_str}" + chkvar(top, "dst_settings.flow_source", settings.flow_source) + "\n"
    
    return res


def output_fluid_effector(mesh_name, name, settings, indent=0):
    res = ""
    i_str = "    " * indent
    
    res += f"{i_str}mod = imported_obj.modifiers.new(name='{name}', type='FLUID')\n"
    res += f"{i_str}mod.fluid_type = 'EFFECTOR'\n\n"
    
    # Copy key effector settings
    res += f"{i_str}dst_settings = mod.effector_settings\n"

    top = f"{mesh_name}.EFFECTOR"
    res += f"{i_str}" + chkvar(top, "dst_settings.surface_distance", settings.surface_distance)
    res += f"{i_str}" + chkvar(top, "dst_settings.use_plane_init", settings.use_plane_init) + "\n"

    return res


def output_rigid_body(mesh, indent=0):
    res = ""
    i_str = "    " * indent

    res += f"{i_str}bpy.context.view_layer.objects.active = imported_obj\n"
    res += f"{i_str}imported_obj.select_set(True)\n"
    res += f"{i_str}bpy.ops.rigidbody.object_add()\n\n"

    top = mesh.name
    rigid_body = mesh.rigid_body
    res += f"{i_str}imported_obj.rigid_body.type = '{rigid_body.type}'\n"
    res += f"{i_str}" + chkvar(top, "imported_obj.rigid_body.mass", rigid_body.mass)
    res += f"{i_str}imported_obj.rigid_body.collision_shape = 'MESH'\n\n"

    # get frame that rigid body starts to move
    starts_moving = 0
    if mesh.animation_data:
        for fcurve in mesh.animation_data.action.fcurves:
            if fcurve.data_path == "rigid_body.kinematic":
                for keyframe in fcurve.keyframe_points:
                    if not keyframe.co.y:
                        starts_moving = keyframe.co.x
                break

    # add approproiate keyframes
    if starts_moving > 0:
        res += f"{i_str}imported_obj.rigid_body.kinematic = True\n"
        res += f"{i_str}imported_obj.keyframe_insert(data_path='rigid_body.kinematic', frame=0)\n"

        top = f"{mesh.name}.rigid_body.start_move"
        res += f"{i_str}" + chkvar(top, "start_move", starts_moving)
        res += f"{i_str}imported_obj.rigid_body.kinematic = False\n"
        res += f"{i_str}imported_obj.keyframe_insert(data_path='rigid_body.kinematic', frame=start_move)\n\n"

    return res



def output_mesh(mesh, indent=0):
    res = ""
    i_str = "    " * indent

    # save the mesh geometry as an obj file
    save_geometry(mesh)
    
    # Write code to import the saved geometry in the generated .py file
    res += f"{i_str}###################\n"
    res += f"{i_str}# Mesh: {mesh.name}\n"
    res += f"{i_str}bpy.ops.wm.obj_import(filepath=os.path.join(base_path, '{mesh.name}.obj').replace('/', '\\\\'))\n"

    # After import, apply transform (location/rotation/scale)
    res += f"{i_str}imported_obj = bpy.context.selected_objects[0]\n"
    res += f"{i_str}imported_obj.name = '{mesh.name}'\n"

    loc = mesh.location
    rot = mesh.rotation_euler
    scale = mesh.scale
    res += f"{i_str}{chkvar(mesh.name, 'imported_obj.location', tuple(loc))}"
    res += f"{i_str}{chkvar(mesh.name, 'imported_obj.rotation_euler', tuple(rot))}"
    res += f"{i_str}{chkvar(mesh.name, 'imported_obj.scale', tuple(scale))}\n"
    res += f"{i_str}imported_obj.hide_render = {mesh.hide_render}\n"

    # Write more code to import the material for each mesh
    res += output_mesh_mat(mesh.material_slots, indent)

    # if fluid data exists, output that as well
    for mod_key in mesh.modifiers.keys():
        mod = mesh.modifiers[mod_key]
        # optional TODO: Other modifiers
        
        if mod_key == "Fluid":
            if mod.domain_settings:
                res += f"{i_str}# domain settings\n"
                res += output_fluid_domain(mesh.name, mod_key, mod.domain_settings, indent)
            if mod.flow_settings:
                res += f"{i_str}# flow settings\n"
                res += output_fluid_flow(mesh.name, mod_key, mod.flow_settings, indent)
            if mod.effector_settings:
                res += f"{i_str}# effector settings\n"
                res += output_fluid_effector(mesh.name, mod_key, mod.effector_settings, indent)
    
    # if rigid body data exists, output that as well
    if mesh.rigid_body:
        res += f"{i_str}# rigid body settings\n"
        res += output_rigid_body(mesh, indent)
    
    return res

def output_light(light, indent=0):
    res = ""
    i_str = "    " * indent
    light_data = light.data
    name = light.name

    res += f"{i_str}# Light: {name}\n"
    res += f"{i_str}light_data = bpy.data.lights.new(name='{name}', type='{light_data.type}')\n"

    res += f"{i_str}" + chkvar(name, "light_data.energy", light_data.energy)
    res += f"{i_str}" + chkvar(name, "light_data.color", tuple(light_data.color))

    if light_data.type == 'AREA':
        res += f"{i_str}" + chkvar(name, "light_data.size", light_data.size)
    elif light_data.type == 'SPOT':
        res += f"{i_str}" + chkvar(name, "light_data.spot_size", light_data.spot_size)
        res += f"{i_str}" + chkvar(name, "light_data.spot_blend", light_data.spot_blend)
    elif light_data.type == 'SUN':
        res += f"{i_str}" + chkvar(name, "light_data.angle", light_data.angle)

    res += f"{i_str}light_obj = bpy.data.objects.new(name='{name}', object_data=light_data)\n"
    loc = light.location
    rot = light.rotation_euler
    res += f"{i_str}" + chkvar(name, "light_obj.location", tuple(loc))
    res += f"{i_str}" + chkvar(name, "light_obj.rotation_euler", tuple(rot))
    res += f"{i_str}bpy.context.collection.objects.link(light_obj)\n\n"

    
    return res

def output_cam(cam, indent=0):
    res = ""
    i_str = "    " * indent

    cam_data = cam.data
    res += f"{i_str}# Camera: {cam.name}\n"
    res += f"{i_str}cam_data = bpy.data.cameras.new(name='{cam.name}')\n"
    res += f"{i_str}" + chkvar(cam.name, "cam_data.lens", cam_data.lens)
    res += f"{i_str}" + chkvar(cam.name, "cam_data.sensor_width", cam_data.sensor_width)
    res += f"{i_str}" + chkvar(cam.name, "cam_data.type", cam_data.type)
    res += f"{i_str}" + chkvar(cam.name, "cam_data.clip_start", cam_data.clip_start)
    res += f"{i_str}" + chkvar(cam.name, "cam_data.clip_end", cam_data.clip_end)
    
    res += f"{i_str}cam_obj = bpy.data.objects.new('{cam.name}', cam_data)\n"
    loc = cam.location
    rot = cam.rotation_euler
    res += f"{i_str}" + chkvar(cam.name, "cam_obj.location", tuple(loc))
    res += f"{i_str}" + chkvar(cam.name, "cam_obj.rotation_euler", tuple(rot))
    res += f"{i_str}bpy.context.collection.objects.link(cam_obj)\n"
    res += f"{i_str}bpy.context.scene.camera = cam_obj\n\n"
    
    return res

def output_metadata(indent=0):
    res = ""
    i_str = "    " * indent
    res += f"{i_str}bpy.context.scene.render.engine = '{bpy.context.scene.render.engine}'\n"
    res += f"{i_str}bpy.context.scene.cycles.samples = {bpy.context.scene.cycles.samples}\n"
    res += f"{i_str}bpy.context.scene.cycles.device = '{bpy.context.scene.cycles.device}'\n"
    res += f"{i_str}bpy.context.scene.cycles.denoising_use_gpu = True\n\n"

    scene = bpy.context.scene
    res += f"{i_str}scene = bpy.context.scene\n"
    res += f"{i_str}scene.render.image_settings.file_format = '{scene.render.image_settings.file_format}'\n"
    res += f"{i_str}scene.render.ffmpeg.format = '{scene.render.ffmpeg.format}'\n"
    res += f"{i_str}scene.render.ffmpeg.codec = '{scene.render.ffmpeg.codec}'\n"
    res += f"{i_str}scene.render.ffmpeg.constant_rate_factor = '{scene.render.ffmpeg.constant_rate_factor}'\n"
    res += f"{i_str}scene.render.ffmpeg.ffmpeg_preset = '{scene.render.ffmpeg.ffmpeg_preset}'\n"
    res += f"{i_str}scene.render.ffmpeg.video_bitrate = {scene.render.ffmpeg.video_bitrate}\n"
    res += f"{i_str}scene.render.ffmpeg.gopsize = {scene.render.ffmpeg.gopsize}\n"
    res += f"{i_str}scene.frame_start = {scene.frame_start}\n"
    res += f"{i_str}scene.frame_end = {scene.frame_end}\n\n"
    
    world = bpy.context.scene.world
    res += f"{i_str}world = bpy.context.scene.world\n"
    res += f"{i_str}world.use_nodes = True\n"
    res += f"{i_str}nodes = world.node_tree.nodes\n"
    res += f"{i_str}links = world.node_tree.links\n"
    res += f"{i_str}nodes.clear()\n"
    for node in world.node_tree.nodes:
        if node.type == 'TEX_ENVIRONMENT':
            # Texture data
            texture_path = bpy.path.abspath(node.image.filepath)
            texture_path = texture_path.replace('\\', '\\\\')
            
            res += f"{i_str}env_tex_node = nodes.new(type='ShaderNodeTexEnvironment')\n"
            res += f"{i_str}bg_node = nodes.new(type='ShaderNodeBackground')\n"
            res += f"{i_str}output_node = nodes.new(type='ShaderNodeOutputWorld')\n\n"
            
            res += f"{i_str}" + chkvar("W@texture_path", "texture_path", texture_path)
            res += f"{i_str}env_tex_node.image = bpy.data.images.load(texture_path)\n"
            
            res += f"{i_str}links.new(env_tex_node.outputs['Color'], bg_node.inputs['Color'])\n"
            res += f"{i_str}links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])\n\n"

    output_path = scene.render.filepath.replace('\\', '\\\\')
    res += f"{i_str}scene.render.filepath = '{output_path}'\n\n"

    return res

def generate_code(out_dir, indent=0):
    res = ""
    i_str = "    " * indent
    res += f"{i_str}import os\n"
    res += f"{i_str}import sys\n"
    res += f"{i_str}import bpy\n"
    res += f"{i_str}import argparse\n\n"


    res += f"{i_str}def generate_scene(\n"
    for variable in what_to_change:
        res += f"{i_str}    {what_to_change[variable]}, \n"
    res += f"{i_str}    ):\n"
        
    i_str = "    " * (indent + 1)
    res += f"{i_str}# Delete all existing objects\n"
    res += f"{i_str}bpy.ops.object.select_all(action='SELECT')\n"
    res += f"{i_str}bpy.ops.object.delete(use_global=False)\n\n"

    # categorize data
    meshes = []
    lights = []
    cameras = []
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            meshes.append(obj)
        if obj.type == "LIGHT":
            lights.append(obj)
        if obj.type == "CAMERA":
            cameras.append(obj)
         
    # 1. meshes
    res += f"{i_str}# 1. meshes\n"
    mesh_path = mesh_dir.replace('\\', '\\\\')
    res += f"{i_str}base_path = '{mesh_path}'\n\n"
    for mesh in meshes:
        res += output_mesh(mesh, indent+1)
    
    # 2. lights
    res += f"{i_str}# 2. lights\n"
    for obj in lights:
        res += output_light(obj, indent+1)
    
    # 3. cameras
    res += f"{i_str}# 3. cameras\n"
    for cam in cameras:
        res += output_cam(cam, indent+1)
    
    # 4. other data
    res += output_metadata(indent+1)
    
    # 5. add parser and execution line
    i_str = "    " * indent

    res += f"{i_str}parser = argparse.ArgumentParser()\n"
    for param in default_values:
        value = default_values[param]
        param_type = type(value).__name__
        res += f"{i_str}parser.add_argument('--{param}', type={param_type}, default={print_var(value)})\n"
    
    res += f"{i_str}args = sys.argv\n"
    res += f"{i_str}if '--' in args:\n"
    res += f"{i_str}    args = args[args.index('--') + 1:]\n"
    res += f"{i_str}    p_args = parser.parse_args(args)\n"
    res += f"{i_str}else:\n"
    res += f"{i_str}    p_args = parser.parse_args('')\n\n"

    res += f"{i_str}generate_scene(\n"
    for param in default_values:
        res += f"{i_str}{param}=p_args.{param}, \n"
    res += f"{i_str})\n\n"

    scene_path = os.path.join(out_dir, "scene_aug.blend")
    scene_path = scene_path.replace('\\', '\\\\')
    res += f"{i_str}bpy.ops.wm.save_mainfile(filepath='{scene_path}')\n"
    res += f"{i_str}exit()\n"
    
    return res


#### main ####
# assume the scene is already loaded
parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str)
args = sys.argv
if '--' in args:
    args = args[args.index('--') + 1:]
    p_args = parser.parse_args(args)
else:
    p_args = parser.parse_args()


indent = 0; default_values = {}
mesh_dir = os.path.join(p_args.out_dir, "meshes")
os.makedirs(mesh_dir, exist_ok=True)
cache_dir = os.path.join(p_args.out_dir, "fluid_cache")
os.makedirs(cache_dir, exist_ok=True)
what_to_change = load_vars(os.path.join(p_args.out_dir, "sampleconf.py"))
code = generate_code(p_args.out_dir, indent)
with open(os.path.join(p_args.out_dir, "out.py"), 'w') as f:
    f.write(code)
    f.close()
print("done!")