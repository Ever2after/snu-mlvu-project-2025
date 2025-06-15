#######################################################################################
##### for scene generation #####
what_to_change = {
    "Camera.location": "cam_loc", 
    "Camera.rotation_euler": "cam_rot", 

    "Cube_002.DOMAIN.viscosity_value": "viscosity", 
    "Cube_002.DOMAIN.particle_radius": "part_rad", 
    "Cube_002.DOMAIN.particle_number": "part_num", 
    "Cube_002.DOMAIN.particle_randomness": "part_random", 
    "Cube_002.DOMAIN.timesteps_min": "ts_min", 

    "Sun.rotation_euler" : "light_angle", 

    "Plane.FLOW.stop_flow": "stop_flow", 

    "Sphere.location": "sphere_loc",
    "Cube.location": "cube_loc",

    "M@mat_cube.base_color" : "cube_color", 
    "M@mat_sphere.base_color" : "sphere_color", 

    "M@Water.base_color" : "water_color", 

    "W@cycles_samples" : "samples",
    "W@frame_step" : "step"
}


#######################################################################################
##### for variable sampling #####
values = {
    ('cam_loc', 'cam_rot') : [
        ((-6.0318, -16.642, 13.4058), (0.9805, 0.0, -0.3902)),
        ((-8.40339, 16.8564, 8.03589), (1.2247, 0.0, -2.7061)),
        ((14.0975, -0.491662, 11.8147), (0.8818, 0.0, -4.7402)),
        ((0.138317, 1.15085, 19.1647), (-0.0671, 0.0, -0.0026)),
    ],
    ('viscosity', 'part_rad', 'part_num', 'part_random', 'ts_min') : [
        (-1.0, 0.8, 2, 0.1, 2),
        (0.0, 1.2, 4, 1.0, 2),
        (0.001, 1.2, 4, 1.0, 2),
        (0.002, 1.2, 4, 1.0, 2),
        (0.01, 1.2, 4, 1.0, 2),
        (0.02, 1.2, 4, 1.0, 2),
    ],
    ('light_angle',) : [
        ((-1.0621, 0.1769, -2.2585),),
        ((1.4587, -3.1914, -4.784),),
        ((4.8718, -3.122, -4.1463),),
    ],
    ('stop_flow',) : [
        (61.0,),
        (36.0,),
        (121.0,),
    ],
    ('sphere_loc', 'cube_loc') : [
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ((4.7179, -0.2993, 0.0), (-5.0961, -0.2993, 0.0)),
    ],
    ('sphere_color', 'cube_color') : [
        ((0.8, 0.8, 0.8, 1.0), (0.8, 0.8, 0.8, 1.0)),
        ((0.8003, 0.0769, 0.0612, 1.0), (0.213, 0.8001, 0.0682, 1.0)),
    ],
    ('water_color',) : [
        ((0.8, 0.8, 0.8, 1.0),),
        ((0.1896, 0.3616, 0.8002, 1.0),),
    ],
    ('samples', 'step') : [
        (32, 3),
    ],
}
