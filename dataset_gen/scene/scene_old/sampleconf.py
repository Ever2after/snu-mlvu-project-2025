#######################################################################################
##### for scene generation #####
what_to_change = {
                    #"sink.location" : "sink_loc",
                    #"M@Water.inputs['Base Color'].default_value" : "water_color",
                    "domain.DOMAIN.resolution_max" : "sim_res",
                    #"W@frame_end" : "end_frame",
                    #"Camera.lens" : "cam_lens",
                    #"W@texture_path" : "texture_path"
                }


#######################################################################################
##### for variable sampling #####
values = {
    ('sim_res',) : [
        (72,)
    ], 
}