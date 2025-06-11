#######################################################################################
##### for scene generation #####
what_to_change = {
    "Camera.location": "cam_loc", 
    "Camera.rotation_euler": "cam_rot", 

    "Domain.DOMAIN.viscosity_value": "viscosity", 
    "Domain.DOMAIN.particle_radius": "part_rad", 
    "Domain.DOMAIN.particle_number": "part_num", 
    "Domain.DOMAIN.particle_randomness": "part_random", 
    "Domain.DOMAIN.timesteps_min": "ts_min", 
    
    "Cube.location": "fluid_loc", 
    "Cube.scale": "fluid_size", 
    "Cylinder.keyframe_1.scale" : "obj_scale1", 
    "Cylinder.keyframe_75.scale" : "obj_scale2", 
    "Cylinder.keyframe_150.scale" : "obj_scale3", 
    
    "Sun.rotation_euler" : "light_angle", 

    "Cylinder.keyframe_75.location": "mid_loc", 
    "Cylinder.keyframe_150.location": "end_loc", 

    "Cylinder.keyframe_75.rotation_euler": "mid_rot", 
    "Cylinder.keyframe_150.rotation_euler": "end_rot", 

    "M@Water.base_color" : "color", 

    "W@cycles_samples" : "samples",
    "W@frame_step" : "step"
}


#######################################################################################
##### for variable sampling #####
values = {
    ('cam_loc', 'cam_rot') : [
        ((-10.4813, -12.1814, 10.9795), (0.9767, -0.0, -0.6581)),
        ((1.2341, -18.097, 18.5776), (0.8098, -0.0, 0.0908)),
        ((19.5479, 0.172, 6.7769), (1.3125, -0.0, 1.5917)),
        ((-0.5098, 4.2388, 25.8093), (0.2025, -0.0, 3.1556)),
    ],
    ('viscosity', 'part_rad', 'part_num', 'part_random', 'ts_min') : [
        (-1.0, 0.8, 2, 0.1, 2),
        (0.0, 1.2, 4, 1.0, 2),
        (0.001, 1.2, 4, 1.0, 2),
        (0.002, 1.2, 4, 1.0, 2),
        (0.01, 1.2, 4, 1.0, 2),
        (0.02, 1.2, 4, 1.0, 2),
    ],
    ('fluid_loc', 'fluid_size', 'obj_scale1', 'obj_scale2', 'obj_scale3') : [
        ((0.0, 0.0, 3.1613), (1.0, 1.0, 10.0), (1.0, 1.0, 0.698), (1.0, 1.0, 0.698), (1.0, 1.0, 0.698)),
        ((0.0, 0.0, 5.5713), (1.0, 1.0, 17.0), (1.0, 1.0, 1.25), (1.0, 1.0, 1.25), (1.0, 1.0, 1.35)),
    ],
    ('light_angle',) : [
        ((-1.0621, 0.1769, -2.2585),),
        ((0.02, 0.1769, -2.2585),),
    ],
    ('mid_loc', 'end_loc') : [
        ((-6.2078, 0.0, 1.0657), (6.0421, 0.0, 1.0657)),
        ((-6.2078, 0.8858, 3.2549), (6.0421, 0.0, 1.0657)),
        ((-1.5047, -0.1612, 0.8746), (6.0421, 0.0, 1.0657)),
    ],
    ('mid_rot', 'end_rot') : [
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ((4.1888, 0.6109, 0.0), (8.3776, 1.309, 0.0)),
    ],
    ('color',) : [
        #((0.8, 0.8, 0.8, 1.0),),
        #((0.3202, 0.3107, 0.8002, 1.0),),
        ((0.107, 0.107, 0.107, 1.0),),
    ],
    ('samples', 'step') : [
        (32, 3),
    ],
}
