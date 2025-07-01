#######################################################################################
##### for scene generation #####
what_to_change = {
    "Camera.location": "cam_loc", 
    "Camera.rotation_euler": "cam_rot", 

    "Domain_i.DOMAIN.viscosity_value": "viscosity_i", 
    "Domain_i.DOMAIN.particle_radius": "part_rad_i", 
    "Domain_i.DOMAIN.particle_number": "part_num_i", 
    "Domain_i.DOMAIN.particle_randomness": "part_random_i", 
    "Domain_i.DOMAIN.timesteps_min": "ts_min_i", 
    "Plane_i.FLOW.velocity_coord": "flow_init_v", 

    "Domain_r.DOMAIN.viscosity_value": "viscosity_r", 
    "Domain_r.DOMAIN.particle_radius": "part_rad_r", 
    "Domain_r.DOMAIN.particle_number": "part_num_r", 
    "Domain_r.DOMAIN.particle_randomness": "part_random_r", 
    "Domain_r.DOMAIN.timesteps_min": "ts_min_r", 

    "Sun.rotation_euler" : "light_angle", 

    "Sphere_i.location": "sphere_loc",
    "Cube_i.location": "cube_loc",
    "Cube_i.rotation_euler": "cube_rot",

    "Icosphere_r.scale": "ripple_size",

    "M@Water_i.base_color" : "water_color_i", 
    "M@Water_r.base_color" : "water_color_r",

    "W@cycles_samples" : "samples",
    "W@frame_step" : "step"
}


#######################################################################################
##### for variable sampling #####
values = {
    ('cam_loc', 'cam_rot') : [
        ((1.4327, -6.5218, 1.9709), (1.3653, 0.0, 0.0215)),
    ],
    ('viscosity_i', 'part_rad_i', 'part_num_i', 'part_random_i', 'ts_min_i', 'flow_init_v') : [
        (-1.0, 0.8, 2, 0.1, 1, (0.0, 0.0, 0.0)),
        (0.0001, 0.8, 2, 0.1, 2, (30.0, 0, 0)),
        (0.001, 1.2, 4, 1.0, 2, (60.0, 0, 0)),
    ],
    ('viscosity_r', 'part_rad_r', 'part_num_r', 'part_random_r', 'ts_min_r') : [
        (-1.0, 0.8, 2, 0.1, 1),
        (0.001, 1.2, 4, 1.0, 2),
        (0.01, 1.2, 4, 1.0, 2),
    ],
    ('light_angle',) : [
        ((-1.0621, 0.1769, -2.2585),),
        ((1.4587, -3.1914, -4.784),),
        ((1.8718, -3.122, -4.1463),),
    ],
    ('sphere_loc', 'cube_loc', 'cube_rot') : [
        ((0.0795, 0.2095, 0.1726), (0.0795, 0.2095, 0.1726), (0.0, 0.0, 0.0)),
        
        ((4.7179, -0.2993, 0.0), (-5.0961, -0.2993, 0.0), (0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.25), (1.5708, 0.0, 0.0)),
        ((4.7179, -0.2993, 0.0), (-5.0961, -0.2993, 0.25), (1.5708, 0.0, 0.0)),
    ],
    ('ripple_size',) : [
        ((1.5123, 1.5123, 1.5123),),
        ((3.1, 3.1, 3.1),),
    ],
    ('water_color_i', 'water_color_r') : [
        ((0.8, 0.8, 0.8, 1.0), (0.8, 0.8, 0.8, 1.0)),
    ],
    ('samples', 'step') : [
        (64, 1),
    ],
}
