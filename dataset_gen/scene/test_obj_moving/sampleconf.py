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

    "Object.src_file": "obj_type",

    "Icosphere.scale": "flow_size", 

    "Sun.rotation_euler" : "light_angle", 

    "Object.keyframe_75.location": "mid_loc", 
    "Object.keyframe_150.location": "end_loc", 

    "Object.keyframe_75.rotation_euler": "mid_rot", 
    "Object.keyframe_150.rotation_euler": "end_rot",

    "M@Col.base_color" : "color", 

    "W@cycles_samples" : "samples",
    "W@frame_step" : "step"
}


#######################################################################################
##### for variable sampling #####
values = {
    ('cam_loc', 'cam_rot') : [
        ((-11.4944, -28.8427, 16.1751), (1.0891, -0.0, -0.37)),
        ((18.0948, -26.9537, -10.9442), (1.885, -0.0, 0.6283)),
    ],
    ('viscosity', 'part_rad', 'part_num', 'part_random', 'ts_min') : [
        (-1.0, 0.8, 2, 0.1, 1),
        (0.0, 1.2, 4, 1.0, 2),
        (0.001, 1.2, 4, 1.0, 2),
        (0.002, 1.2, 4, 1.0, 2),
        (0.01, 1.2, 4, 1.0, 2),
        (0.02, 1.2, 4, 1.0, 2),
    ],
    ('obj_type',) : [
        ('Cylinder',),
        ('Cone',),
        ('Cube',),
    ],
    ('flow_size',) : [
        ((0.7, 0.7, 0.5),),
        ((0.3, 0.3, 0.4),),
    ],
    ('light_angle',) : [
        ((-1.0621, 0.1769, -2.2585),),
        ((-1.4974, 0.5996, -2.4294),),
    ],
    ('mid_loc', 'end_loc') : [
        ((0.0, 0.75, 3.2632), (0.0, 0.75, -4.3477)),
        ((0.0, 1.7185, 0.9661), (0.0, 1.7185, -4.3477)),
        ((-0.7762, 0.75, 3.2632), (-1.5681, -1.0765, -2.0137)),
    ],
    ('mid_rot', 'end_rot') : [
        ((0.0, -0.0, 0.0), (0.0, -0.0, 0.0)),
        ((-2.0944, -0.0, 0.0), (2.0944, -0.0, 0.0)),
    ],
    ('color',) : [
        ((0.8, 0.8, 0.8, 1.0),),
        ((0.8, 0.6542, 0.0309, 1.0),),
        ((0.112, 0.2629, 0.8001, 1.0),),
    ],
    ('samples', 'step') : [
        (16, 3),
    ],
}
