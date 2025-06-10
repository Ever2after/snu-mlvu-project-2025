#######################################################################################
##### for scene generation #####
what_to_change = {
    #"Suzanne.rigid_body.start_move" : "water_vol",
    "water.scale" : "water_vol",
    "Suzanne.rigid_body.vel_lin" : "val_lin",
    "domain.DOMAIN.resolution_max" : "sim_res",
    "W@frame_end" : "end_frame"
}


#######################################################################################
##### for variable sampling #####
values = {
    ('sim_res',) : [
        (80,),
    ], 
    ('end_frame',) : [
        (10,),
    ], 
    ('water_vol',) : [
        ((0.85, 0.85, 0.85),), 
        ((0.45, 0.45, 0.45),),
    ],
    ('val_lin',) : [
        ((0, 0, 0),),
        ((0.5, 0, 0),),
        ((0, -0.5, 0),),
    ]
}