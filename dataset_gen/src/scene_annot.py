def _flow_des(a):
    size = a["flow_size"]
    vs = a["viscosity"]
    if vs != "high":
        s = ""
        if size == "big":
            s = "The fluid impacts the container and rapidly fills it."
        else:
            s = "The fluid impacts the container and gradually fills it."
        if a["flow_loc"] == "corner":
            s = s + " It first pours out at the container’s corner."
        return s
    else:
        return "The fluid slowly accumulates in the container, spreading at an extremely slow pace."
    

def _move_des(a) :
# camera에 따른 방향 변화
    if a["cam_loc"] == [-0.5098, 4.2388, 25.8093]:
        pos_1 = "left side"
        pos_2 = "to the right side"
    elif a["cam_loc"] == [19.5479, 0.172, 6.7769]:
        pos_1 = "down side"
        pos_2 = "upward"
    else:
        pos_1 = "right side"
        pos_2 = "to the left side"
    
    if a["mid_rot"] == [0, 0, 0]:
        s = "The object moves through the fluid, displacing the surrounding liquid.\n"
    else:
        s = "The object rolls and flips through the fluid, carrying liquid upward as it moves.\n"
        
    if a["mid_loc"] == [-6.2078, 0.0, 1.0657]:
        s = s + f"It starts at the {pos_1} of the fluid, moves all the way {pos_2}, and then moves back to {pos_1}.\n"
    elif a["mid_loc"] == [-6.2078, 0.8858, 3.2549]:
        s = s + f"It starts at the {pos_1} of the fluid, moves {pos_2}, and then moves back to {pos_1}.\n"
    else:
        s = s + f"It starts at the {pos_1} of the fluid, moves partway {pos_2}, and then moves back to {pos_1}.\n"
    
    return s

		


def _pi(a, scene_type):
    if scene_type == "free_fall":
        reason = {
            "low": (
                "the fluid lands and spreads immediately without building up, "
                "overflowing the container at once"
            ),
            "medium": (
                "the fluid lands and accumulates slightly before spreading, "
                "clinging briefly to the container walls before overflows"
            ),
            "high": (
                "the fluid lands and builds up in layers, spreading very slowly"
            )
        }[a["viscosity"]]
    if scene_type == "obj_moving":
        reason = {
            "low": (
                "the object shears through the fluid, generating ripples and leaving almost no fluid clinging behind"
            ),
            "medium": (
                "the object shears through the fluid with a thin film of fluid that clings briefly"
            ),
            "high": (
                "the object shears through the fluid with no ripple behind and a thick film of fluid that clings and decays slowly"
            )
        }[a["viscosity"]]

    return (
        f"As {reason}, "
        f"the fluid’s viscosity is estimated as {a['viscosity']}."
    )


def extract_annotation (param, type):
    info = param.copy()
    #common
    vc = param.get("viscosity")
    if vc is not None:
        info["viscosity"] = (
            "low" if vc <= 0
            else "medium" if vc < 0.008
            else "high"
        )
        

    if type == "free_fall":
        #param: flow size, flow loc, color, sink_metal
        fs = param.get("flow_size")
        if fs is not None:
            info["flow_size"] = (
                "big"
                if fs[0] > 1
                else "small"
            )
        sm = param.get("sink_metal")
        if sm is not None:
            info["sink_metal"] = (
                "reflective, "
                if sm != 0
                else ""
            )
        if "flow_loc" in param.keys():
            info["flow_loc"] = (
                "center" if param["flow_loc"] == [0,0,0]
                else  "corner" if  param["flow_loc"] == [0.5,0.5,0]
                else ""
            )

        if "color" in param.keys():
           info["color"] = (
                "gray" if param["color"] == [0.292, 0.259, 0.361, 1]
                else "sky-blue" if param["color"] == [0.533, 0.706, 0.906, 1]
                else "yellow" if param["color"] == [0.906, 0.792, 0.452, 1]
                else ""
            )

        info["container_description"] = f"{info['sink_metal']}{info['color']}"
        annot = (
            f"There is a {info['container_description']} container.\n"
            f"A fluid is released from a point directly above the container’s {info["flow_loc"]}, "
            f"and falls freely under gravity.\n" 
            f"{_flow_des(info)} \n"
            f"{_pi(info, type)}"
        )
        return annot
    elif type == "obj_moving":
        if "color" in param.keys():
            info["color"] = (
                "white-colored" if param["color"] == [0.8, 0.8, 0.8, 1.0]
                else  "slate-colored" if  param["color"] == [0.3202, 0.3107, 0.8002, 1.0]
                else "dark-colored"
            )   

            annot = (
                f"There is a {info['color']} fluid that contains a white cylinderical object.\n"
                f"{_move_des(info)}"
                f"{_pi(info, type)}"
            )    
            return annot

    else:
        print("Undefined scene:", type)
        return ""
