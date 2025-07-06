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

    "Icosphere.FLOW.surface_distance": "water_size", 

    "Sun.rotation_euler" : "light_angle", 

    "Plane.rotation_euler" : "plane_angle", 

    "M@Water.base_color" : "water_color", 
    "M@Water.Transmission Weight" : "water_alpha",

    "M@Plane.base_color" : "plane_color", 

    "W@cycles_samples" : "samples",
    "W@frame_step" : "step"
}


#######################################################################################
##### for variable sampling #####
values = {
    ('cam_loc', 'cam_rot') : [
        ((7.2372, -5.3956, 6.2422), (1.023, 0.0, 0.9176)),
        ((8.238, 0.1076, 2.4686), (1.3125, 0.0, 1.6057)),
        ((0.5809, -8.9213, 2.5753), (1.295, 0.0, 0.0559)),
        ((7.3008, -0.2399, 8.9949), (0.6213, 0.0, 1.5708)),
    ],
    ('viscosity', 'part_rad', 'part_num', 'part_random', 'ts_min') : [
        (-1.0, 0.8, 2, 0.1, 1),
        (0.0, 1.2, 4, 1.0, 2),
        (0.001, 1.2, 4, 1.0, 2),
        (0.002, 1.2, 4, 1.0, 2),
        (0.01, 1.2, 4, 1.0, 2),
        (0.02, 1.2, 4, 1.0, 2),
    ],
    ('water_size',) : [
        (5.0,),
        (15.0,),
    ],
    ('light_angle',) : [
        ((0.3963, 0.127, -0.7037),),
        ((-0.808, 0.2841, -0.7037),),
        ((-7.4613, 0.0031, -1.7356),),
    ],
    ('plane_angle',) : [
        ((0.0, 0.35, 0.0),),
        ((0.0, 0.1766, 0.0),),
    ],
    ('water_color', 'water_alpha') : [
        ((0.8, 0.8, 0.8, 1.0), 0.718),
        ((0.8, 0.5696, 0.2066, 1.0), 0.718),
    ],
    ('plane_color',) : [
        ((0.8, 0.8, 0.8, 1.0),),
        ((0.0468, 0.0287, 0.2982, 1.0),),
        ((0.0248, 0.0062, 0.0, 1.0),),
    ],
    ('samples', 'step') : [
        (16, 3),
    ],
}
