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
    "Icosphere.FLOW.velocity_coord": "flow_v", 

    "Icosphere.location": "water_loc", 

    "Icosphere.FLOW.surface_distance": "water_size", 

    "Sun.rotation_euler" : "light_angle", 

    "M@Water.base_color" : "water_color", 

    "M@Water.Transmission Weight" : "water_alpha", 

    "W@cycles_samples" : "samples",
    "W@frame_step" : "step"
}


#######################################################################################
##### for variable sampling #####
values = {
    ('cam_loc', 'cam_rot') : [
        ((3.9416, -2.8738, 3.1844), (0.9358, 0.0, 0.9176)),
        ((-0.019107, 1.01672, 8.50865), (0.109, 0.0, 3.1115)),
        ((-3.06275, 3.34001, 0.731767), (1.2646, 0.0, 3.9101)),
        ((-6.6413, 5.99124, 4.29006), (1.0579, 0.0, 3.9241)),
    ],
    ('viscosity', 'part_rad', 'part_num', 'part_random', 'ts_min', 'flow_v') : [
        (-1.0, 0.8, 2, 0.1, 1, (0, 0, 0)),
        (0.0, 1.2, 4, 1.0, 2, (0, 0, 0)),
        (0.001, 1.2, 4, 1.0, 2, (0, 0, -3)),
        (0.002, 1.2, 4, 1.0, 2, (0, 0, -3)),
        (0.01, 1.2, 4, 1.0, 2, (0, 0, -10)),
        (0.02, 1.2, 4, 1.0, 2, (0, 0, -20)),
    ],
    ('water_loc',) : [
        ((-0.1745, -0.0233, 2.2169),),
        ((-0.1745, 1.535, 2.2169),),
    ],
    ('water_size',) : [
        (2.0,),
        (5.0,),
    ],
    ('light_angle',) : [
        ((0.3963, 0.127, -0.7037),),
        ((-0.5865, -1.1677, -0.5713),),
    ],
    ('water_color',) : [
        ((0.8, 0.8, 0.8, 1.0),),
        ((0.286, 0.4393, 0.8, 1.0),),
        ((0.3427, 0.2505, 0.1456, 1.0),),
    ],
    ('water_alpha',) : [
        (0.718,),
        (0.995,),
    ],
    ('samples', 'step') : [
        (16, 3),
    ],
}
