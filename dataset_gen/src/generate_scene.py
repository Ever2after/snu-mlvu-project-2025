import bpy
import os
import sys
import io
import argparse
import importlib.util
from mathutils import Vector
from contextlib import redirect_stdout
import platform

os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

def load_vars(conf_path):
    if not os.path.exists(conf_path):
        with open(conf_path, 'w') as f:
            f.write("#######################################################################################\n")
            f.write("##### for scene generation #####\n")
            f.write("what_to_change = {\n")
            f.write("}\n\n\n")
            f.write("#######################################################################################\n")
            f.write("##### for variable sampling #####\n")
            f.write("values = {\n")
            f.write("}\n")

    module_name = "loaded_conf_module"

    spec = importlib.util.spec_from_file_location(module_name, conf_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    what_to_change = getattr(module, "what_to_change", None)

    return what_to_change


def write_summery():
    with open(os.path.join(out_dir, "out_summary.txt"), 'w') as f:
        f.write("MESH\n")
        for mesh in summary_dict["Mesh"]:
            f.write(f"\t{mesh}\n")
            if summary_dict["Mesh"][mesh]["Fluid"]:
                f.write(f"\t\tFluid: {summary_dict['Mesh'][mesh]['Fluid']}\n")
            if summary_dict["Mesh"][mesh]["Rigid"]:
                f.write(f"\t\tRigid: {summary_dict['Mesh'][mesh]['Rigid']}\n")
            if summary_dict["Mesh"][mesh]["Keyframe"]:
                f.write(f"\t\tKeyframes: ")
                for kf in summary_dict['Mesh'][mesh]['Keyframe']:
                    f.write(f"{kf}, ")
                f.write("\n")
            f.write("\n")
        f.write("MATERIAL\n")
        for mat in summary_dict["Material"]:
            f.write(f"\t{mat}: {summary_dict['Material'][mat]}\n")
        f.write("\n")
        f.write("LIGHT\n")
        for light in summary_dict["Light"]:
            f.write(f"\t{light}: {summary_dict['Light'][light]}\n")
        f.write("\n")
        f.write("CAMERA\n")
        for camera in summary_dict["Camera"]:
            f.write(f"\t{camera}\n")
        f.write("\n")
        f.write("DEFAULTS\n")
        for param in default_values:
            value = default_values[param]
            f.write(f"\t{param}: {print_var(value)}\n")
        f.close()


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
        if var_path in what_to_change_used:
            what_to_change_used.remove(var_path)
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
        if platform.system() == 'Windows':
            mesh_path = os.path.join(mesh_dir, f"{mesh_name}.obj").replace("/", "\\")
        else:
            mesh_path = os.path.join(mesh_dir, f"{mesh_name}.obj")
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


def output_mat_bsdf(mat_name, bsdf, indent=0):
    res = ""
    i_str = "    " * indent

    res += f"{i_str}_bsdf = _nodes.get('Principled BSDF')\n"
    
    res += f"{i_str}if _bsdf is None:\n"
    res += f"{i_str}    _bsdf = _nodes.new(type='ShaderNodeBsdfPrincipled')\n"
    res += f"{i_str}    _output = _nodes.get('Material Output') or _nodes.new(type='ShaderNodeOutputMaterial')\n"
    res += f"{i_str}    _mat.node_tree.links.new(_bsdf.outputs['BSDF'], _output.inputs['Surface'])\n"

    # Albedo
    base_color = bsdf.inputs['Base Color'].default_value
    top = f"M@{mat_name}.base_color"
    res += f"{i_str}" + chkvar(top, "_mat_base_color", tuple(base_color))
    res += f"{i_str}_bsdf.inputs['Base Color'].default_value = _mat_base_color\n"

    # Metallic, Roughness, IOR, Alpha, Transmission Weight
    mat_attrs = ['Metallic', 'Roughness', 'IOR', 'Alpha', 'Transmission Weight']
    mat_attr_pyvarnames = [attr.replace(" ", "_") for attr in mat_attrs]
    for attr, var_name in zip(mat_attrs, mat_attr_pyvarnames):
        value = bsdf.inputs[attr].default_value
        top = f"M@{mat_name}.{attr}"
        res += f"{i_str}" + chkvar(top, f"_mat_{var_name}", value)
    
    for attr, var_name in zip(mat_attrs, mat_attr_pyvarnames):
        res += f"{i_str}_bsdf.inputs['{attr}'].default_value = _mat_{var_name}\n"
    
    return res


def output_mat_vol_abs(mat_name, vol_abs, indent=0):
    res = ""
    i_str = "    " * indent

    res += f"{i_str}_vol_abs = _nodes.new(type='ShaderNodeVolumeAbsorption')\n"
    res += f"{i_str}_links = _mat.node_tree.links\n"

    # Albedo
    base_color = vol_abs.inputs['Color'].default_value
    density = vol_abs.inputs['Density'].default_value
    res += f"{i_str}" + chkvar(f"M@{mat_name}.vol_color", "_mat_vol_color", tuple(base_color))
    res += f"{i_str}" + chkvar(f"M@{mat_name}.density", "_mat_density", density)
    res += f"{i_str}_vol_abs.inputs['Color'].default_value = _mat_vol_color\n"
    res += f"{i_str}_vol_abs.inputs['Density'].default_value = _mat_density\n"
    res += f"{i_str}_output = _nodes.get('Material Output')\n"
    res += f"{i_str}_links.new(_vol_abs.outputs['Volume'], _output.inputs['Volume'])\n"

    return res


def output_mesh_mat(mats, indent=0):
    res = ""
    i_str = "    " * indent

    for mat in mats:
        mat = mat.material
        bsdf = None
        vol_abs = None
        for node in mat.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                bsdf = node
            if node.type == 'VOLUME_ABSORPTION':
                vol_abs = node

        if not bsdf and not vol_abs:
            continue  # Skip non-principled materials
        
        mat_name = mat.name.replace(".", "_")

        res += f"{i_str}# Material: {mat_name}\n"
        res += f"{i_str}_mat = bpy.data.materials.new(name='{mat_name}')\n"
        res += f"{i_str}_mat.use_nodes = True\n"
        res += f"{i_str}_nodes = _mat.node_tree.nodes\n\n"

        if bsdf:
            res += output_mat_bsdf(mat_name, bsdf, indent)
            summary_dict["Material"][mat_name] = "BSDF"
        
        if vol_abs:
            res += output_mat_vol_abs(mat_name, vol_abs, indent)
            summary_dict["Material"][mat_name] = "VOL_ABS"

        res += f"{i_str}\n"
        res += f"{i_str}_imported_obj.data.materials.append(bpy.data.materials['{mat_name}'])\n\n"

    return res


def output_fluid_domain(mesh_name, name, settings, indent=0):
    res = ""
    i_str = "    " * indent
    
    # Add a fluid modifier to the target
    res += f"{i_str}_mod = _imported_obj.modifiers.new(name='{name}', type='FLUID')\n"
    res += f"{i_str}_mod.fluid_type = 'DOMAIN'\n\n"

    # Copy key domain properties
    res += f"{i_str}_dst_settings = _mod.domain_settings\n"

    top = f"{mesh_name}.DOMAIN"
    res += f"{i_str}" + chkvar(top, "_dst_settings.domain_type", settings.domain_type)
    res += f"{i_str}_dst_settings.cache_type = 'MODULAR'\n"
    res += f"{i_str}_dst_settings.cache_mesh_format = 'OBJECT'\n"
    res += f"{i_str}_dst_settings.cache_resumable = True\n"
    res += f"{i_str}" + chkvar(top, "_dst_settings.resolution_max", settings.resolution_max)
    res += f"{i_str}" + chkvar(top, "_dst_settings.use_mesh", settings.use_mesh)
    res += f"{i_str}" + chkvar(top, "_dst_settings.cfl_condition", settings.cfl_condition)
    res += f"{i_str}" + chkvar(top, "_dst_settings.particle_radius", settings.particle_radius)
    res += f"{i_str}" + chkvar(top, "_dst_settings.particle_number", settings.particle_number)
    res += f"{i_str}" + chkvar(top, "_dst_settings.particle_randomness", settings.particle_randomness)
    res += f"{i_str}" + chkvar(top, "_dst_settings.particle_band_width", settings.particle_band_width)
    res += f"{i_str}" + chkvar(top, "_dst_settings.cache_frame_start", settings.cache_frame_start)
    res += f"{i_str}" + chkvar(top, "_dst_settings.cache_frame_end", settings.cache_frame_end)
    res += f"{i_str}_dst_settings.cache_frame_end = round(_dst_settings.cache_frame_end * _fps_scale)\n"
    res += f"{i_str}_dst_settings.cache_directory = '//fluid_cache/{mesh_name}/'\n\n"

    # viscosity
    # if viscosity is invariable, use scene settings
    top = f"{mesh_name}.DOMAIN.viscosity_value"
    if top not in what_to_change:
        if settings.use_viscosity:
            res += f"{i_str}_dst_settings.use_viscosity = True\n"
            res += f"{i_str}_dst_settings.viscosity_value = {settings.viscosity_value}\n\n"
    else:
        res += f"{i_str}" + chkvar(top, "_viscosity_given", settings.viscosity_value if settings.use_viscosity else -1.0)
        res += f"{i_str}if _viscosity_given > -1.0:\n"
        i_str = "    " * (indent + 1)
        res += f"{i_str}_dst_settings.use_viscosity = True\n"
        res += f"{i_str}_dst_settings.viscosity_value = max(_viscosity_given, 0.0)\n\n"
        i_str = "    " * indent
    
    return res


def output_fluid_flow(mesh_name, name, settings, indent=0):
    res = ""
    i_str = "    " * indent

    res += f"{i_str}_mod = _imported_obj.modifiers.new(name='{name}', type='FLUID')\n"
    res += f"{i_str}_mod.fluid_type = 'FLOW'\n\n"
    
    # Copy key flow properties
    res += f"{i_str}_dst_settings = _mod.flow_settings\n"
    
    top = f"{mesh_name}.FLOW"
    res += f"{i_str}" + chkvar(top, "_dst_settings.flow_type", settings.flow_type)
    res += f"{i_str}" + chkvar(top, "_dst_settings.flow_behavior", settings.flow_behavior)
    res += f"{i_str}" + chkvar(top, "_dst_settings.flow_source", settings.flow_source)
    res += f"{i_str}_dst_settings.use_initial_velocity = True\n"
    res += f"{i_str}" + chkvar(top, "_dst_settings.velocity_coord", tuple(settings.velocity_coord)) + "\n"
    
    return res


def output_fluid_effector(mesh_name, name, settings, indent=0):
    res = ""
    i_str = "    " * indent
    
    res += f"{i_str}_mod = _imported_obj.modifiers.new(name='{name}', type='FLUID')\n"
    res += f"{i_str}_mod.fluid_type = 'EFFECTOR'\n\n"
    
    # Copy key effector settings
    res += f"{i_str}_dst_settings = _mod.effector_settings\n"

    top = f"{mesh_name}.EFFECTOR"
    res += f"{i_str}" + chkvar(top, "_dst_settings.surface_distance", settings.surface_distance)
    res += f"{i_str}" + chkvar(top, "_dst_settings.use_plane_init", settings.use_plane_init) + "\n"

    return res


def get_world_transform(obj, frame):
    bpy.context.scene.frame_set(frame)
    eval_obj = obj.evaluated_get(depsgraph)
    location = eval_obj.matrix_world.to_translation()
    rotation = eval_obj.matrix_world.to_euler('XYZ')
    return location.copy(), Vector(rotation.copy())


def get_initstate_rigid_body(mesh, start_f):
    loc1, rot1 = get_world_transform(mesh, start_f-2)
    loc2, rot2 = get_world_transform(mesh, start_f-1)
    print(rot1)
    print(rot2)
    dt = 1 / bpy.context.scene.render.fps

    linear_velocity = (loc2 - loc1) / dt
    angular_velocity = (rot2 - rot1) / dt

    return linear_velocity, angular_velocity

def output_rigid_body(mesh, indent=0):
    res = ""
    i_str = "    " * indent

    res += f"{i_str}bpy.context.view_layer.objects.active = _imported_obj\n"
    res += f"{i_str}_imported_obj.select_set(True)\n"
    res += f"{i_str}bpy.ops.rigidbody.object_add()\n\n"

    top = mesh.name
    rigid_body = mesh.rigid_body
    res += f"{i_str}_imported_obj.rigid_body.type = '{rigid_body.type}'\n"
    res += f"{i_str}" + chkvar(top, "_imported_obj.rigid_body.mass", rigid_body.mass)
    res += f"{i_str}_imported_obj.rigid_body.collision_shape = 'MESH'\n\n"

    if rigid_body.type == 'ACTIVE':
        # get frame that rigid body starts to move
        starts_moving = 0
        if mesh.animation_data:
            for fcurve in mesh.animation_data.action.fcurves:
                if fcurve.data_path == "rigid_body.kinematic":
                    for keyframe in fcurve.keyframe_points:
                        if not keyframe.co.y:
                            starts_moving = round(keyframe.co.x)
                    break
        res += f"{i_str}" + chkvar(f"{mesh.name}.rigid_body.start_move", "_start_move", starts_moving)
        res += f"{i_str}_start_move = round(_start_move * _fps_scale)\n"
        
        # get initial position & velocity
        init_v_lin, init_v_ang = get_initstate_rigid_body(mesh, starts_moving)
        res += f"{i_str}" + chkvar(f"{mesh.name}.rigid_body.vel_lin", "_vel_lin", tuple(init_v_lin))
        res += f"{i_str}" + chkvar(f"{mesh.name}.rigid_body.vel_ang", "_vel_ang", tuple(init_v_ang)) + "\n"

        # keyframe @ x - 1 (initial pos/rot)
        res += f"{i_str}_c_pos = _imported_obj.location\n"
        res += f"{i_str}_c_rot = _imported_obj.rotation_euler\n"
        res += f"{i_str}bpy.context.scene.frame_set(_start_move - 1)\n"
        res += f"{i_str}_imported_obj.keyframe_insert(data_path='location', frame=_start_move - 1)\n"
        res += f"{i_str}_imported_obj.keyframe_insert(data_path='rotation_euler', frame=_start_move - 1)\n\n"

        # keyframe @ x - 2
        res += f"{i_str}_p_pos = _c_pos - Vector(_vel_lin) / bpy.context.scene.render.fps\n"
        res += f"{i_str}_p_rot = Vector(_c_rot) - Vector(_vel_ang) / bpy.context.scene.render.fps\n"
        res += f"{i_str}bpy.context.scene.frame_set(_start_move - 2)\n"
        res += f"{i_str}_imported_obj.location = _p_pos\n"
        res += f"{i_str}_imported_obj.rotation_euler = _p_rot\n"
        res += f"{i_str}_imported_obj.keyframe_insert(data_path='location', frame=_start_move - 2)\n"
        res += f"{i_str}_imported_obj.keyframe_insert(data_path='rotation_euler', frame=_start_move - 2)\n\n"

        # add approproiate keyframes regarding "animated" attr
        if starts_moving > 0:
            res += f"{i_str}_imported_obj.rigid_body.kinematic = True\n"
            res += f"{i_str}_imported_obj.keyframe_insert(data_path='rigid_body.kinematic', frame=0)\n"
            
            res += f"{i_str}_imported_obj.rigid_body.kinematic = False\n"
            res += f"{i_str}_imported_obj.keyframe_insert(data_path='rigid_body.kinematic', frame=_start_move)\n\n"

    return res


def gather_animation(mesh):
    '''
    output struct: 
    {<frame #> : {"location": (,,), "rotation_euler": (,,), "scale": (,,)}, 
      ...
    }
    '''
    res = {}

    action = mesh.animation_data.action
    for fcurve in action.fcurves:
        if fcurve.data_path not in ["location", "rotation_euler", "scale"]:
            continue 

        for keyframe in fcurve.keyframe_points:
            # check if new frame should be added
            frame_num = int(keyframe.co.x)
            if frame_num not in res:
                res[frame_num] = {"location": [None, None, None], 
                                  "rotation_euler": [None, None, None], 
                                  "scale": [None, None, None], }
            
            # add current data
            res[frame_num][fcurve.data_path][fcurve.array_index] = keyframe.co.y
    
    # check for missing data(aka None)
    for frame_num in res:
        for Attr_name in res[frame_num]:
            Attr = res[frame_num][Attr_name]
            if None in Attr:
                # missing data; go to that frame and fill in the gap
                bpy.context.scene.frame_set(frame_num)
                if Attr_name == "location":
                    value = mesh.matrix_world.to_translation()
                elif Attr_name == "rotation_euler":
                    value = mesh.matrix_world.to_euler()
                elif Attr_name == "scale":
                    value = mesh.matrix_world.to_scale()
                res[frame_num][Attr_name] = list(value)
    
    return res

def output_animation(mesh, indent=0):
    res = ""
    i_str = "    " * indent

    keyframe_data = gather_animation(mesh)

    for f_num in keyframe_data:
        top = f"{mesh.name}.keyframe_{f_num}"
        res += f"{i_str}_kf_new_scene = round({f_num} * _fps_scale)\n"
        res += f"{i_str}bpy.context.scene.frame_set(_kf_new_scene)\n"
        res += f"{i_str}" + chkvar(top, "_imported_obj.location", tuple(keyframe_data[f_num]['location']))
        res += f"{i_str}" + chkvar(top, "_imported_obj.rotation_euler", tuple(keyframe_data[f_num]['rotation_euler']))
        res += f"{i_str}" + chkvar(top, "_imported_obj.scale", tuple(keyframe_data[f_num]['scale']))
        res += f"{i_str}_imported_obj.keyframe_insert(data_path='location', frame=_kf_new_scene)\n"
        res += f"{i_str}_imported_obj.keyframe_insert(data_path='rotation_euler', frame=_kf_new_scene)\n"
        res += f"{i_str}_imported_obj.keyframe_insert(data_path='scale', frame=_kf_new_scene)\n\n"
        
        summary_dict["Mesh"][mesh.name]["Keyframe"].append(f_num)
    
    return res

def output_mesh(mesh, indent=0):
    res = ""
    i_str = "    " * indent

    # save the mesh geometry as an obj file
    save_geometry(mesh)
    
    # Write code to import the saved geometry in the generated .py file
    res += f"{i_str}###################\n"
    res += f"{i_str}# Mesh: {mesh.name}\n"
    if platform.system() == 'Windows':
        res += f"{i_str}bpy.ops.wm.obj_import(filepath=os.path.join(_base_path, '{mesh.name}.obj').replace('/', '\\\\'))\n"
    else:
        res += f"{i_str}bpy.ops.wm.obj_import(filepath=os.path.join(_base_path, '{mesh.name}.obj'))\n"

    # After import, apply transform (location/rotation/scale)
    res += f"{i_str}_imported_obj = bpy.context.selected_objects[0]\n"
    res += f"{i_str}_imported_obj.name = '{mesh.name}'\n"

    bpy.context.scene.frame_set(bpy.context.scene.frame_end)
    loc = mesh.location # init location in case of rigid body
    rot = mesh.rotation_euler # init rotation in case of rigid body
    scale = mesh.scale
    res += f"{i_str}{chkvar(mesh.name, '_imported_obj.location', tuple(loc))}"
    res += f"{i_str}{chkvar(mesh.name, '_imported_obj.rotation_euler', tuple(rot))}"
    res += f"{i_str}{chkvar(mesh.name, '_imported_obj.scale', tuple(scale))}\n"
    res += f"{i_str}_imported_obj.hide_render = {mesh.hide_render}\n"

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
                summary_dict["Mesh"][mesh.name]["Fluid"] = "DOMAIN"
            if mod.flow_settings:
                res += f"{i_str}# flow settings\n"
                res += output_fluid_flow(mesh.name, mod_key, mod.flow_settings, indent)
                summary_dict["Mesh"][mesh.name]["Fluid"] = "FLOW"
            if mod.effector_settings:
                res += f"{i_str}# effector settings\n"
                res += output_fluid_effector(mesh.name, mod_key, mod.effector_settings, indent)
                summary_dict["Mesh"][mesh.name]["Fluid"] = "EFFECTOR"
    
    # if rigid body data exists, output that as well
    if mesh.rigid_body:
        res += f"{i_str}# rigid body settings\n"
        res += output_rigid_body(mesh, indent)
        summary_dict["Mesh"][mesh.name]["Rigid"] = mesh.rigid_body.type
    
    # if animation data exists, output that as well
    if mesh.animation_data and mesh.animation_data.action:
        res += f"{i_str}# animation keyframes\n"
        res += output_animation(mesh, indent)
    
    return res

def output_light(light, indent=0):
    res = ""
    i_str = "    " * indent
    light_data = light.data
    name = light.name

    res += f"{i_str}# Light: {name}\n"
    res += f"{i_str}_light_data = bpy.data.lights.new(name='{name}', type='{light_data.type}')\n"

    res += f"{i_str}" + chkvar(name, "_light_data.energy", light_data.energy)
    res += f"{i_str}" + chkvar(name, "_light_data.color", tuple(light_data.color))

    if light_data.type == 'AREA':
        res += f"{i_str}" + chkvar(name, "_light_data.size", light_data.size)
    elif light_data.type == 'SPOT':
        res += f"{i_str}" + chkvar(name, "_light_data.spot_size", light_data.spot_size)
        res += f"{i_str}" + chkvar(name, "_light_data.spot_blend", light_data.spot_blend)
    elif light_data.type == 'SUN':
        res += f"{i_str}" + chkvar(name, "_light_data.angle", light_data.angle)

    res += f"{i_str}_light_obj = bpy.data.objects.new(name='{name}', object_data=_light_data)\n"
    loc = light.location
    rot = light.rotation_euler
    res += f"{i_str}" + chkvar(name, "_light_obj.location", tuple(loc))
    res += f"{i_str}" + chkvar(name, "_light_obj.rotation_euler", tuple(rot))
    res += f"{i_str}bpy.context.collection.objects.link(_light_obj)\n\n"

    
    return res

def output_cam(cam, indent=0):
    res = ""
    i_str = "    " * indent

    cam_data = cam.data
    res += f"{i_str}# Camera: {cam.name}\n"
    res += f"{i_str}_cam_data = bpy.data.cameras.new(name='{cam.name}')\n"
    res += f"{i_str}" + chkvar(cam.name, "_cam_data.lens", cam_data.lens)
    res += f"{i_str}" + chkvar(cam.name, "_cam_data.sensor_width", cam_data.sensor_width)
    res += f"{i_str}" + chkvar(cam.name, "_cam_data.type", cam_data.type)
    res += f"{i_str}" + chkvar(cam.name, "_cam_data.clip_start", cam_data.clip_start)
    res += f"{i_str}" + chkvar(cam.name, "_cam_data.clip_end", cam_data.clip_end)
    
    res += f"{i_str}_cam_obj = bpy.data.objects.new('{cam.name}', _cam_data)\n"
    loc = cam.location
    rot = cam.rotation_euler
    res += f"{i_str}" + chkvar(cam.name, "_cam_obj.location", tuple(loc))
    res += f"{i_str}" + chkvar(cam.name, "_cam_obj.rotation_euler", tuple(rot))
    res += f"{i_str}bpy.context.collection.objects.link(_cam_obj)\n"
    res += f"{i_str}bpy.context.scene.camera = _cam_obj\n\n"
    
    return res

def output_metadata(indent=0):
    res = ""
    i_str = "    " * indent
    res += f"{i_str}bpy.context.scene.render.engine = '{bpy.context.scene.render.engine}'\n"
    res += f"{i_str}" + chkvar("W@cycles_samples", "_cycles_sample", bpy.context.scene.cycles.samples)
    res += f"{i_str}bpy.context.scene.cycles.samples = _cycles_sample\n"
    res += f"{i_str}bpy.context.scene.cycles.device = '{bpy.context.scene.cycles.device}'\n"
    res += f"{i_str}bpy.context.scene.cycles.denoising_use_gpu = True\n\n"

    scene = bpy.context.scene
    res += f"{i_str}_scene = bpy.context.scene\n"
    res += f"{i_str}_scene.render.image_settings.file_format = '{scene.render.image_settings.file_format}'\n"
    res += f"{i_str}_scene.render.ffmpeg.format = '{scene.render.ffmpeg.format}'\n"
    res += f"{i_str}_scene.render.ffmpeg.codec = '{scene.render.ffmpeg.codec}'\n"
    res += f"{i_str}_scene.render.ffmpeg.constant_rate_factor = '{scene.render.ffmpeg.constant_rate_factor}'\n"
    res += f"{i_str}_scene.render.ffmpeg.ffmpeg_preset = '{scene.render.ffmpeg.ffmpeg_preset}'\n"
    res += f"{i_str}_scene.render.ffmpeg.video_bitrate = {scene.render.ffmpeg.video_bitrate}\n"
    res += f"{i_str}_scene.render.ffmpeg.gopsize = {scene.render.ffmpeg.gopsize}\n"
    res += f"{i_str}" + chkvar("W@frame_start", "_frame_start", scene.frame_start)
    res += f"{i_str}" + chkvar("W@frame_end", "_frame_end", scene.frame_end)
    res += f"{i_str}" + chkvar("W@frame_step", "_frame_step", scene.frame_step)
    res += f"{i_str}_scene.frame_start = _frame_start\n"
    res += f"{i_str}_scene.frame_end = round(_frame_end * _fps_scale)\n"
    res += f"{i_str}_scene.frame_step = _frame_step\n\n"
    
    world = bpy.context.scene.world
    res += f"{i_str}_world = bpy.context.scene.world\n"
    res += f"{i_str}_world.use_nodes = True\n"
    res += f"{i_str}_nodes = _world.node_tree.nodes\n"
    res += f"{i_str}_links = _world.node_tree.links\n"
    res += f"{i_str}_nodes.clear()\n"
    for node in world.node_tree.nodes:
        if node.type == 'TEX_ENVIRONMENT':
            # Texture data
            texture_name = os.path.basename(node.image.filepath.replace('\\', '/'))
            texture_path = bpy.path.abspath(f"//texture/{texture_name}")
            if platform.system() == 'Windows':
                texture_path = texture_path.replace('\\', '\\\\')

            res += f"{i_str}_env_tex_node = _nodes.new(type='ShaderNodeTexEnvironment')\n"
            res += f"{i_str}_bg_node = _nodes.new(type='ShaderNodeBackground')\n"
            res += f"{i_str}_output_node = _nodes.new(type='ShaderNodeOutputWorld')\n\n"
            
            res += f"{i_str}" + chkvar("W@texture_path", "texture_path", texture_path)
            res += f"{i_str}_env_tex_node.image = bpy.data.images.load(texture_path)\n"
            
            res += f"{i_str}_links.new(_env_tex_node.outputs['Color'], _bg_node.inputs['Color'])\n"
            res += f"{i_str}_links.new(_bg_node.outputs['Background'], _output_node.inputs['Surface'])\n\n"

    if platform.system() == 'Windows':
        output_path = scene.render.filepath.replace('\\', '\\\\')
    else:
        output_path = scene.render.filepath
    res += f"{i_str}_scene.render.filepath = '{output_path}'\n\n"

    return res

def generate_code(out_dir, indent=0):
    res = ""
    i_str = "    " * indent
    res += f"{i_str}import os\n"
    res += f"{i_str}import sys\n"
    res += f"{i_str}import bpy\n"
    res += f"{i_str}from mathutils import Vector\n"
    res += f"{i_str}import argparse\n\n"


    res += f"{i_str}def generate_scene(\n"
    for variable in what_to_change:
        res += f"{i_str}    {what_to_change[variable]}, \n"
    res += f"{i_str}    ):\n"
        
    i_str = "    " * (indent + 1)
    res += f"{i_str}# Delete all existing objects\n"
    res += f"{i_str}bpy.ops.object.select_all(action='SELECT')\n"
    res += f"{i_str}bpy.ops.object.delete(use_global=False)\n\n"

    # set the fps early as rigid body depends on it
    res += f"{i_str}" + chkvar("W@fps", "_fps", bpy.context.scene.render.fps)
    res += f"{i_str}_fps_scale = _fps / {bpy.context.scene.render.fps}\n" # required in case fps changes during scene gen
    res += f"{i_str}bpy.context.scene.render.fps = _fps\n\n"

    # categorize data
    meshes = []
    lights = []
    cameras = []
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            meshes.append(obj)
            summary_dict["Mesh"][obj.name] = {"Fluid": None, "Rigid": None, "Keyframe": []}
        if obj.type == "LIGHT":
            lights.append(obj)
            summary_dict["Light"][obj.name] = obj.data.type
        if obj.type == "CAMERA":
            cameras.append(obj)
            summary_dict["Camera"][obj.name] = None
         
    # 1. meshes
    res += f"{i_str}# 1. meshes\n"
    mesh_path = mesh_dir.replace('\\', '\\\\')
    res += f"{i_str}_base_path = '{mesh_path}'\n\n"
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
        if param_type == 'tuple':
            res += f"{i_str}parser.add_argument('--{param}', type=lambda x: tuple(map(float, x[1:].split(','))) , default={print_var(value)})\n"
        else:
            res += f"{i_str}parser.add_argument('--{param}', type=lambda x: {param_type}(x[1:]), default={print_var(value)})\n"
    
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
    if platform.system() == 'Windows':
        scene_path = scene_path.replace('\\', '\\\\')
    res += f"{i_str}bpy.ops.wm.save_mainfile(filepath='{scene_path}')\n"
    res += f"{i_str}exit()\n"
    
    return res


#### main ####
# assume the scene is already loaded
parser = argparse.ArgumentParser()
parser.add_argument('--out_script_path', type=str)
parser.add_argument('--skip_summary', action="store_true")
args = sys.argv
if '--' in args:
    args = args[args.index('--') + 1:]
    p_args = parser.parse_args(args)
else:
    p_args = parser.parse_args()


indent = 0; default_values = {}; out_dir = os.path.dirname(p_args.out_script_path)

mesh_dir = os.path.join(out_dir, "meshes")
os.makedirs(mesh_dir, exist_ok=True)
cache_dir = os.path.join(out_dir, "fluid_cache")
os.makedirs(cache_dir, exist_ok=True)

what_to_change = load_vars(os.path.join(out_dir, "sampleconf.py"))
what_to_change_used = list(what_to_change.keys()) #  variable has not been consumed
summary_dict = {"Mesh": {}, "Light": {}, "Camera": {}, "Material": {}} # Dict for saving data to print summary info

depsgraph = bpy.context.evaluated_depsgraph_get()

code = generate_code(out_dir, indent)

# if unmatched variables exist, print a warning
if len(what_to_change_used) != 0:
    print("\033[93m" + "WARNING: unused variable: ")
    for i in what_to_change_used:
          print(f"\t{i}")
    print("\033[0m", end="")

# write code
with open(p_args.out_script_path, 'w') as f:
    f.write(code)
    f.close()

# write summary
if not p_args.skip_summary:
    write_summery()
print("done!")