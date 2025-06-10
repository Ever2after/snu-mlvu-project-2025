#######################################################################################
##### for scene generation #####
what_to_change = {
                    #"sink.location" : "sink_loc",
                    #"M@Water.inputs['Base Color'].default_value" : "water_color",
                    "domain.DOMAIN.resolution_max" : "sim_res",
                    "W@frame_end" : "end_frame",
                    "Camera.location" : "cam_pos",
                    "Camera.rotation_euler" : "cam_rot",
                    #"W@texture_path" : "texture_path"
                }


#######################################################################################
##### for variable sampling #####
values = {
    ('sim_res',) : [
        (96,)
    ], 
    ('cam_pos', 'cam_rot',) : [
        ((2.8, -3.4, 2.5), (1.11, 0, 0.815)),
        ((3.2, 1.7, 2.5), (1.07, 0, 2)),
    ],
    ('end_frame',) : [
        (100,)
    ],
}