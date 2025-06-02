#######################################################################################
##### for scene generation #####
what_to_change = {
    "Cube.002.location": "loc",
    "Cylinder.keyframe_1.location": "loc1",
    "Cylinder.keyframe_75.location": "loc2",
    "Cylinder.keyframe_150.location": "loc3",
    "W@cycles_samples": "sample"
}


#######################################################################################
##### for variable sampling #####
values = {
    ("loc",) : [
        (None,),
    ],
    ("loc1",) : [
        ((5.2944, 0.0000, 0.6),),
    ],
    ("loc2",) : [
        ((-6.2078, 0.0000, 0.6),),
    ],
    ("loc3",) : [
        ((6.0421, 0.0000, 0.6),),
    ],
    ("sample",) : [
        (16,),
    ],
}
