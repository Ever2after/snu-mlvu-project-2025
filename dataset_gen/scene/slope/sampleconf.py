#######################################################################################
##### for scene generation #####
what_to_change = {
    "Camera.location": "cam_loc", 
    "Camera.rotation_euler": "cam_rot", 

    "domain.DOMAIN.viscosity_value": "viscosity", 
    "domain.DOMAIN.particle_radius": "part_rad", 
    "domain.DOMAIN.particle_number": "part_num", 
    "domain.DOMAIN.particle_randomness": "part_random", 
    "domain.DOMAIN.timesteps_min": "ts_min", 

    "slope.location": "slope_loc", 
    "slope.rotation_euler": "slope_rot", 
    "icosphere.location": "src_loc", 
    "icosphere.FLOW.velocity_coord": "flow_init_v", 

    "M@Water.Transmission Weight" : "alpha", 

    "M@Sink.base_color" : "sink_color", 
    "M@Sink2.base_color" : "slope_color", 

    "W@cycles_samples" : "samples",
    "W@frame_step" : "step"
}


#######################################################################################
##### for variable sampling #####
values = {
    ('cam_loc', 'cam_rot') : [
        ((5.96631, -5.84698, 7.81959), (0.9214, 0.0, 0.8478)),
        ((2.39662, 8.15721, 10.3665), (0.7476, 0.0, 2.6282)),
        ((8.35464, 0.305898, 10.6395), (0.7899, 0.0, 1.5666)),
        ((5.0468, -5.9116, 4.4038), (1.1093, 0.0, 0.8149)),
    ],
    ('viscosity', 'part_rad', 'part_num', 'part_random', 'ts_min') : [
        (-1.0, 0.8, 2, 0.1, 2),
        (0.0, 1.2, 4, 1.0, 2),
        (0.001, 1.2, 4, 1.0, 2),
        (0.002, 1.2, 4, 1.0, 4), #mod
        (0.01, 1.2, 4, 1.0, 4), #mod
        (0.02, 1.2, 4, 1.0, 4), #mod
    ],
    ('slope_loc', 'slope_rot', 'src_loc', 'flow_init_v') : [
        ((0.0, 0.0, 0.0), (0.0, -0.0914, -0.0), (1.7578, 0.0312, 0.5758), (0, 0, 0.0)),
        ((-0.36, 0.0, 0.51), (0.0, -0.5236, -0.0), (1.2278, 0.0312, -0.1842), (0, 0, -0.2)),
        ((-0.36, 0.0, 0.21), (0.0, -0.2793, -0.0), (0.9378, 0.0312, 0.1858), (0, 0, -0.1)),
        ((-0.27, 0.0, -0.36), (0.0, 0.1878, -0.0), (2.9578, 0.0312, 0.2558), (0, 0, 0.0)), #mod
    ],
    ('alpha',) : [
        (0.718,),
        (0.995,),
    ],
    ('sink_color',) : [
        ((0.2462, 0.4564, 0.7991, 1.0),),
        ((0.7992, 0.5992, 0.5447, 1.0),),
        ((0.0714, 0.0486, 0.0486, 1.0),),
    ],
    ('slope_color',) : [
        ((0.2462, 0.4564, 0.7991, 1.0),),
        ((0.7991, 0.5088, 0.1573, 1.0),),
    ],
    ('samples', 'step') : [
        (32, 3),
    ],
}
