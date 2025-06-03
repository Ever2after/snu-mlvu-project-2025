#######################################################################################
##### for scene generation #####
what_to_change = {
    "domain.DOMAIN.resolution_max": "flow_res", 
    "domain.DOMAIN.viscosity_value": "viscosity", 
    "Icosphere.scale": "flow_size", 
    "Icosphere.location": "flow_loc", 
    "Camera.location": "cam_loc", 
    "Camera.rotation_euler": "cam_rot", 
    "Sun.rotation_euler" : "light_angle", 
    "M@Sink.base_color" : "color", 
    "M@Sink.Metallic" : "sink_metal"
}


#######################################################################################
##### for variable sampling #####
values = {
    ("cam_loc", "cam_rot", ) : [
        ((6.78, -5.55, 4.25), (1.235, 0, 0.88),),
        ((8.66, 1.32, 4.25), (1.235, 0, 1.72),), 
        ((9.69, -0.33, 0.85), (1.60, 0, 1.51),), 
        ((5.70, -2.83, 8.81), (0.72, 0, 1.13),), 
        ((3.86, 3.28, 1.63), (1.40, 0, 2.3),), 
    ],
    ("viscosity",) : [
        (-1.0,), # no viscosity
        (0.0,), 
        (0.02,), 
        (0.05,), 
        (0.1,), 
    ],
    ("flow_size", "flow_res", ) : [
        ((1, 1, 1), 90),
        ((1.8, 1.8, 1.8), 90),
    ],
    ("flow_loc", ) : [
        ((0, 0, 0),),
        ((0.5, 0.5, 0),), 
    ],
    ("light_angle", ) : [
        ((0.76, 0.91, -2.39),),
        ((3.46, -3.17, -1.78),), 
    ],
    ("color", ) : [
        ((0.533, 0.706, 0.906, 1),),
        ((0.906, 0.792, 0.452, 1),),
        ((0.292, 0.259, 0.361, 1),),
    ],
    ("sink_metal", ) : [
        (0,), 
        (0.8,), 
    ],
}