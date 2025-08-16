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

def _slope_des(a):
    if a['viscosity'] == "high" :
        s = "The fluid clings to the slope and moves very slowly downward.\n"
    else:
        s = "The fluid flows down the slope and into the container.\n"	
    return s

def _obj_des(a):
    if a["cam_loc"] == "left" :
        if a["loc"] == "left" :
            loc_1 = "spherical"
            loc_2 = "rectangular"
        else:
            loc_1 = "rectangular" 
            loc_2 = "spherical"
    elif a["cam_loc"] == "right" :
        if a["loc"] == "right" :
            loc_1 = "spherical"
            loc_2 = "rectangular"
        else:
            loc_1 = "rectangular" 
            loc_2 = "spherical"    
    else:
        loc_1 = "rectangular" 
        loc_2 = "spherical"  
    return  f"In the scene two objects appear, with a {a[loc_1]} {loc_1} object on the left and a {a[loc_2]} {loc_2} object to its right.\n"

def _col (a):
    if a['loc'] == "left": # 구가 먼저 충돌
        loc_1 = "spherical"
        loc_2 = "rectangular"
    else:
        loc_1 = "rectangular"
        loc_2 = "spherical"        
    col_1 = f"The fluid collides with {loc_1} object."
    col_2 = f"And then it collides with {loc_2} object."

    if a['viscosity'] == "high":
        return col_1 + "\n"
    else:
        return col_1 + " " +  col_2 + "\n"

def _rip_des(a):
    if a['viscosity'] == "low" :
        if a['water_size'] == "small":
            s = "When the droplet contacts the surface, pronounced ripples radiate outward swiftly.\n"
        else:
            s = "When the droplet contacts the surface, it splashes on impact and pronounced ripples radiate outward swiftly.\n"
    elif a['viscosity'] == "medium" :
        s = "When the droplet contacts the surface, gentle ripples form and dissipate quickly.\n"
    else:
        s = "When the droplet contacts the surface, it is absorbed with minimal ripple formation.\n"
    return s

def _slope2_des(a):
    vc = a["viscosity"]
    if vc == "low":
        return "The fluid falls onto the plank, then rapidly slides down and merges with the pool.\n"
    elif vc == "medium":
        return "The fluid falls onto the plank, slides down, and merges with the pool.\n"
    else:
        return "The fluid falls onto the plank, adheres to its surface, and then slowly slides down to join the pool.\n"

def _t_obj_des(a):
    mv = a["movement"]
    vc = a["viscosity"]

    if vc == "low":
        if mv == "in the middle":
            return "The fluid strikes the object, splashes, and pools quickly on the floor.\n"
        elif mv == "at side":
            return "The fluid grazes the object, splashes, and pools quickly on the floor.\n"
        else:  
            return "The fluid hits and brushes past the object, splashes, and pools quickly on the floor.\n"

    elif vc == "middle":
        if mv == "in the middle":
            return "The fluid strikes the object, runs along it, and pools on the floor.\n"
        elif mv == "at side":
            return "The fluid passes the object and pools on the floor.\n"
        else: 
            return "The fluid hits and passes the object, and pools on the floor.\n"

    else:
        if mv == "in the middle":
            return "The fluid strikes the object, then adheres strongly.\n"
        elif mv == "at side":
            return "The fluid passes the object and pools slowly with strong adhesion.\n"
        else: 
            return "The fluid hits and passes the object, and pools slowly with strong adhesion.\n"
		

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
    elif scene_type == "obj_moving":
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
    elif scene_type == "slope":
        reason = {
            "low": (
                "the fluid glides down the slope with almost no resistance"
            ),
            "medium": (
                "the fluid clings lightly to the slope and flows slowly downward"
            ),
            "high": (
                "the fluid sticks to the incline and barely moves"
            )
        }[a["viscosity"]]
    elif scene_type == "obj_interaction":
        reason = {
            "low": (
                "the fluid flows swiftly around the objects with minimal adhesion"
            ),
            "medium": (
                "the fluid wraps gently around the object and advances at a steady pace"
            ),
            "high": (
                "the fluid adheres strongly to the object and creeps forward very slowly"
            )
        }[a["viscosity"]]
    elif scene_type == "ripple":
        reason = {
            "low": (
                "the droplet generates pronounced ripples that radiate outward rapidly"
            ),
            "medium": (
                "the droplet creates ripples that form but quickly dissipate"
            ),
            "high": (
                "the droplet is absorbed with almost no ripples formed"
            )
        }[a["viscosity"]]
    elif scene_type == "test_slope":
        reason = {
            "low": (
                "the fluid slides swiftly down the incline and generates strong ripples in the pool"
            ),
            "medium": (
                "the fluid slides down at a moderate pace and produces moderate rippling in the pool"
            ),
            "high": (
                "the fluid clings to the incline and creeps slowly, leaving the pool surface nearly calm"
            )
        }[a["viscosity"]]
    elif scene_type == "test_obj_moving":
        reason = {
            "low": (
                "the fluid briefly wraps each object and then separates cleanly with minimal adhesion"
            ),
            "medium": (
                "the fluid partially coats the objects; a thin film clears soon after contact"
            ),
            "high": (
                "the fluid adheres strongly, forming a thick coating that drains slowly"
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
        if type == "obj_interaction":
            info["viscosity"] = (
                "low" if vc <= 0
                else "medium" if vc < 0.0008
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
    
    elif type == "slope":
        if "sink_color" in param.keys():
            info["sink_color"] = (
                "sky-blue" if param["sink_color"] == [0.2462, 0.4564, 0.7991, 1.0]
                else "white" if param["sink_color"] == [0.7992, 0.5992, 0.5447, 1.0]
                else "dark gray"
            )   
        if "alpha" in param.keys():
            info["alpha"] = (
                "white" if param["alpha"] == 0.718
                else "clear"
            )  
        if "slope_color" in param.keys():
            info["slope_color"] = (
                "sky-blue" if param["slope_color"] == [0.2462, 0.4564, 0.7991, 1.0]
                else "yellow"
            )  
        if "slope_rot" in param.keys():
            info["slope_rot"] = (
                ", steep" if param["slope_rot"][1] == 0.1878
                else ", gentle" if param["slope_rot"][1] == -0.5236
                else ""
            )     
        annot = (
        f"A {info['alpha']} fluid initially rests atop a {info['slope_color']}{info['slope_rot']} slope.\n"
        f"Below the slope lies a {info['sink_color']} container.\n"
        f"{_slope_des(info)}"
        f"{_pi(info, type)}"
        )
        return annot
    
    elif type == "obj_interaction":
        if "water_color" in param.keys():
            info["water_color"] = (
                "white" if param["water_color"] == [0.8, 0.8, 0.8, 1.0]
                else "blue"
            ) 
        if "sphere_color" in param.keys():
            info["spherical"] = (
                "white" if param["sphere_color"] == [0.8, 0.8, 0.8, 1.0]
                else "red" if param["sphere_color"] == [0.8003, 0.0769, 0.0612, 1.0]
                else "black"
            )   
        if "cube_color" in param.keys():
            info["rectangular"] = (
                "green" if param["cube_color"] == [0.213, 0.8001, 0.0682, 1.0]
                else "white"
            )
        if "cam_loc" in param.keys():
            info["cam_loc"] = (
                "right" if param["cam_loc"] == [-8.40339, 16.8564, 8.03589]
                else "upward" if param["cam_loc"] == [14.0975, -0.491662, 11.8147]
                else "left"
            )
        if "cube_loc" in param.keys(): #첫번째 캠위치 기준
            info["loc"] = (
                "left" if param["cube_loc"][0] == 0.0 #구가 왼
                else "right"
            )
        annot = (  
            f"{_obj_des(info)}"
            f"A {info['water_color']} fluid flows from {info['cam_loc']} toward the objects.\n"
            f"{_col(info)}"
            f"{_pi(info, type)}"
        )
        return annot
    elif type == "ripple": 
        if "water_color" in param.keys():
            info["water_color"] = (
                "colorless" if param["water_color"] == [0.8, 0.8, 0.8, 1.0]
                else "blue" if param["water_color"] == [0.286, 0.4393, 0.8, 1.0]
                else "dark"
            ) 
        if "water_alpha" in param.keys():
            info["water_alpha"] = (
                "clear, " if param["water_alpha"] == 0.995
                else ""
            )
        if "water_size" in param.keys():
            info["water_size"] = (
                "small" if param["water_size"] == 2.0
                else "large"
            )
        annot = (
            f"There sits a pool of {info['water_alpha']}{info['water_color']} fluid below.\n"
            f"A single droplet then falls onto its surface.\n"
            f"{_rip_des(info)}"
            f"{_pi(info, type)}"
        )
        return annot
    elif type == "test_slope":
        if "water_color" in param.keys():
            info["water_color"] = (
                "colorless" if param["water_color"] == [0.8, 0.8, 0.8, 1.0]
                else "yellow"
            )   
        if "plane_color" in param.keys():
            info["plane_color"] = (
                "sky-blue" if param["plane_color"] == [0.8, 0.8, 0.8, 1.0]
                else "blue" if param["plane_color"] == [0.0468, 0.0287, 0.2982, 1.0]
                else "black"
            )  
        if "plane_angle" in param.keys():
            info["plane_angle"] = (
                "steep" if param["plane_angle"] == [0.0, 0.35, 0.0]
                else "gentle"
            )
        if "water_size" in param.keys():
	        info["water_size"] = (
		        "little" if param["water_size"] == 5
		        else "large"
		)
        annot = (
            f"A {info['water_color']} fluid rests in a pool below a {info['plane_color']} plank placed at a {info['plane_angle']} incline.\n"
            f"{_slope2_des(info)}"
            f"{_pi(info, type)}"
        )
        return annot    
    elif type == "test_obj_moving": 
        if "color" in param.keys():
            info["obj_color"] = (
                "white" if param["color"] == [0.8, 0.8, 0.8, 1.0]
                else "yellow" if param["color"] == [0.8, 0.6542, 0.0309, 1.0]
                else "sky-blue"
            ) 
        if "flow_size" in param.keys():
            info["water_size"] = (
                "small" if param["flow_size"][0] == 0.3
                else "large"
            )
        if "obj_type" in param.keys():
		        info["obj"] = param["obj_type"]
        if "mid_rot" in param.keys():
            info["rot"] = (
                False if param["mid_rot"][0] == 0.0
                else True
            ) 
        if "end_loc" in param.keys():
            info["movement"] = (
                "in the middle" if param["end_loc"][1] == 0.75
                else "at side" if param["end_loc"][1] == 1.7185
                else "upward in the middle and downward at side"
            )                   
        annot = (
            f"A colorless fluid falls from above.\n"
            f"In mid-air, a {info['obj_color']} {info['obj']} moves upward and downward{', while rotating' if info['rot'] else ''}.\n"
            f"{_t_obj_des(info)}"
            f"{_pi(info, type)}"
        )
        return annot
    
    else:
        print("Undefined scene:", type)
        return ""

import yaml

def extract_formatted_annotation (param, type):
    info = {
        "fluidPresent":   True,
        "fluidColor":     None,
        "fluidBehavior":  None,
        "flowDirection":  None,
        "flowSize":     None,
        "objects":        None,
        "objectMoving": False,
        "slopeAngle":     None,
        "viscosityLevel": None
    }
    #common
    vc = param.get("viscosity")
    if vc is not None:
        info["viscosityLevel"] = (
            "low" if vc <= 0
            else "middle" if vc < 0.008
            else "high"
        )
        if type == "obj_interaction":
            info["viscosityLevel"] = (
                "low" if vc <= 0
                else "middle" if vc < 0.0008
                else "high"
            )

        
 ##done
    if type == "free_fall":
        #param: flow size, flow loc, color, sink_metal
        info["fluidPresent"] = True
        info["fluidDirection"] = "downward"
        info["fluidColor"] = "colorless"
        fs = param.get("flow_size")
        if fs is not None:
            info["flowSize"] = (
                "big"
                if fs[0] > 1
                else "small"
            )
        #sm = param.get("sink_metal")
        #if sm is not None:
        #    info["sink_metal"] = (
        #        "reflective, "
        #        if sm != 0
        #        else ""
        #    )
        #if "flow_loc" in param.keys():
        #    info["flow_loc"] = (
        #        "center" if param["flow_loc"] == [0,0,0]
        #        else  "corner" if  param["flow_loc"] == [0.5,0.5,0]
        #        else ""
        #    )
        obj_list = []
        if "color" in param.keys():
            if param["color"] == [0.292, 0.259, 0.361, 1]:
                obj_list.append("gray container")
            elif param["color"] == [0.533, 0.706, 0.906, 1]:
                obj_list.append("sky-blue container")
            elif param["color"] == [0.906, 0.792, 0.452, 1]:
                obj_list.append("yellow container")
        info["objects"] = obj_list
        info["fluidBehavior"] = (
            "The fluid falls freely under gravity, lands on the container."
        )

### Slope
    elif type == "slope":
        info["fluidPresent"] = True
        info["fluidColor"] = "colorless"
        info["flowDirection"] = "downward"
        obj_list = []
        if "sink_color" in param.keys():
            if param["sink_color"] == [0.7992, 0.5992, 0.5447, 1.0]:
                obj_list.append("white container")
            elif param["sink_color"] == [0.2462, 0.4564, 0.7991, 1.0]:
                obj_list.append("sky-blue container")
            else:
                obj_list.append("dark gray container")

        #if "alpha" in param.keys():
        #    info["alpha"] = (
        #        "white" if param["alpha"] == 0.718
        #        else "clear"
        #    )  
        if "slope_color" in param.keys():
            if param["slope_color"] == [0.2462, 0.4564, 0.7991, 1.0]:
                obj_list.append("sky-blue slope")
            else:
                obj_list.append("yellow slope")
        info["objects"] = obj_list
        if "slope_rot" in param.keys():
            info["slopeAngle"] = (
                "steep" if param["slope_rot"][1] == 0.1878
                else "gentle" if param["slope_rot"][1] == -0.5236
                else "middle"
            ) 
            ##고점도는 container에 흘러내리지 않음
        if param["slope_rot"][1] == -0.5236:
            info["fluidBehavior"] = (
                "The fluid falls on slope, flows down the slope."
            )
        elif param["slope_rot"][1] == 0.1878:
            if info["viscosityLevel"] == "low":
                info["fluidBehavior"] = (
                    "The fluid falls on slope, slides down the slope, falls on the container."
                )
            else: ### high에서 유체가 아예 안 보이는 현상
                info["fluidBehavior"] = (
                    "The fluid falls on slope, slides down the slope."
                )
        else:
            if info["viscosityLevel"] == "high":
                info["fluidBehavior"] = (
                    "The fluid falls on slope, slides down the slope."
                )
            else:
                info["fluidBehavior"] = (
                    "The fluid falls on slope, slides down the slope, falls on the container."
                )

    ##
    elif type == "obj_interaction":
        info["fluidPresent"] = True

        if "water_color" in param.keys():
            info["fluidColor"] = (
                "colorless" if param["water_color"] == [0.8, 0.8, 0.8, 1.0]
                else "blue"
            ) 
        obj_list = []
        if "sphere_color" in param.keys():
            if param["sphere_color"] == [0.8, 0.8, 0.8, 1.0]:
                obj_list.append("white sphere")
            elif param["sphere_color"] == [0.8003, 0.0769, 0.0612, 1.0]:
                obj_list.append("red sphere")
            else:
                obj_list.append("black sphere")
        if "cube_color" in param.keys():
            if param["cube_color"] == [0.213, 0.8001, 0.0682, 1.0]:
                obj_list.append("green cube")  
            else:
                obj_list.append("white cube")
        info["objects"] = obj_list
        if "cam_loc" in param.keys():
            info["flowDirection"] = (
                "from right to left" if param["cam_loc"] == [-8.40339, 16.8564, 8.03589]
                else "downward" if param["cam_loc"] == [14.0975, -0.491662, 11.8147]
                else "from left to right"
            )
        if "cube_loc" in param.keys(): #첫번째 캠위치 기준
            if param["cube_loc"][0] == 0.0: #구가 왼
                first_col = "sphere"
                second_col = "cube"
            else:
                first_col = "cube"
                second_col = "sphere"

        if info["viscosityLevel"] == "high":
            info["fluidBehavior"] = (
                f"The fluid flows toward the objects, collides with {first_col}."
            )
        else:
            info["fluidBehavior"] = (
                f"The fluid flows toward the objects, collides with {first_col} and then with {second_col}."
            )
    elif type == "ripple": 
        info["fluidPresent"] = True
        info["flowDirection"] = "downward"
        if "water_color" in param.keys():
            if param["water_color"] == [0.8, 0.8, 0.8, 1.0]:
                info["fluidColor"] = "colorless"
            elif param["water_color"] == [0.286, 0.4393, 0.8, 1.0]:
                info["fluidColor"] = "blue"
            else:
                info["fluidColor"] = "dark"
        if "water_size" in param.keys():
            info["flowSize"] = (
                "small" if param["water_size"] == 2.0
                else "large"
            )
        if info["viscosityLevel"] == "high":
            info["fluidBehavior"] = (
                "The fluid falls onto the pool, merges with the pool."
            )
        else:
            info["fluidBehavior"] = (
                "The fluid falls onto the pool, merges with the pool, generates ripples."
            )

    elif type == "obj_moving":
        info["fluidPresent"] = True
        info["flowDirection"] = "therein"
        info["objectMoving"] = True
        info["objects"] = ["white cylinder"]
        if "color" in param.keys():
            info["fluidColor"] = (
                "white" if param["color"] == [0.8, 0.8, 0.8, 1.0]
                else "slate" if param["color"] == [0.3202, 0.3107, 0.8002, 1.0]
                else "dark"
            )
        info["fluidBehavior"] = (
            "The fluid contains a white cylinderical object, is displaced by moving object."
        )  

    else:
        print("Undefined scene:", type)
        return ""
    
    annot = yaml.safe_dump(
    info,
    default_flow_style=False, 
    sort_keys=False  
    )
    return annot





def get_single_scene_qas(param, type):
    qas = []
    # output: [q_type: str, options: list, answer: int] :list
    if type == "free_fall":
        # sink_color, flow_loc
        color = param.get("color")
        color = 0 if color == [0.292, 0.259, 0.361, 1] else 1 if color == [0.533, 0.706, 0.906, 1] else 2
        qas.append({
            "q_type": "obj_color",
            "question": "What is the color of the container?",
            "options": ["gray", "sky-blue", "yellow"],
            "answer": color
        })
        flow_loc = param.get("flow_loc")
        flow_loc = 0 if flow_loc == [0, 0, 0] else 1
        qas.append({
            # 컨테이너의 어떤 위치에 fluid가 release되는지 (공중에서)
            "q_type": "fluid_loc",
            "question": "Which location does the fluid fall from above the container?",
            "options": ["center", "corner"],
            "answer": flow_loc
        })
    elif type == "slope":
        # sink_color, slope_color
        sink_color = param.get("sink_color")
        sink_color = 0 if sink_color == [0.7992, 0.5992, 0.5447, 1.0] else 1 if sink_color == [0.2462, 0.4564, 0.7991, 1.0] else 2
        qas.append({
            "q_type": "obj_color",
            "question": "What is the color of the container?",
            "options": ["white", "sky-blue", "dark gray"],
            "answer": sink_color
        })
        slope_color = param.get("slope_color")
        slope_color = 0 if slope_color == [0.2462, 0.4564, 0.7991, 1.0] else 1
        qas.append({
            "q_type": "obj_color",
            "question": "What is the color of the slope?",
            "options": ["sky-blue", "yellow"],
            "answer": slope_color
        })
    elif type == "ripple":
        # water_color
        water_color = param.get("water_color")
        water_color = 0 if water_color == [0.8, 0.8, 0.8, 1.0] else 1 if water_color == [0.286, 0.4393, 0.8, 1.0] else 2
        qas.append({
            "q_type": "fluid_color",
            "question": "What is the color of the water?",
            "options": ["colorless", "blue", "dark"],
            "answer": water_color
        })
    elif type == "obj_moving":
        # fluid_color, obj_rot, obj_loc
        fluid_color = param.get("color")
        fluid_color = 0 if fluid_color == [0.8, 0.8, 0.8, 1.0] else 1 if fluid_color == [0.3202, 0.3107, 0.8002, 1.0] else 2
        qas.append({
            "q_type": "fluid_color",
            "question": "What is the color of the fluid?",
            "options": ["white", "slate", "dark"],
            "answer": fluid_color
        })
        obj_rot = param.get("mid_rot")
        obj_rot = 0 if obj_rot == [0, 0, 0] else 1
        qas.append({
            "q_type": "obj_rot",
            "question": "Does the object rotate?",
            "options": ["no", "yes"],
            "answer": obj_rot
        })
        cam_loc = param.get("cam_loc")
        cam_loc = 0 if cam_loc == [-0.5098, 4.2388, 25.8093] else 1 if cam_loc == [19.5479, 0.172, 6.7769] else 2
        options = ["left side", "down side", "right side"]
        # object가 fluid의 어떤 위치에서 움직이기 시작하는지 (되돌아오는 위치)
        qas.append({
            "q_type": "obj_loc",
            "question": "Where does the object start moving in the fluid?",
            "options": options,
            "answer": cam_loc
        })
    elif type == "obj_interaction":
        # water_color, sphere/cube_color, cube_loc, collision
        water_color = param.get("water_color")
        water_color = 0 if water_color == [0.8, 0.8, 0.8, 1.0] else 1
        qas.append({
            "q_type": "fluid_color",
            "question": "What is the color of the fluid?",
            "options": ["white", "blue"],
            "answer": water_color
        })
        sphere_color = param.get("sphere_color")
        sphere_color = 0 if sphere_color == [0.8, 0.8, 0.8, 1.0] else 1 if sphere_color == [0.8003, 0.0769, 0.0612, 1.0] else 2
        qas.append({
            "q_type": "obj_color",
            "question": "What is the color of the spherical object?",
            "options": ["white", "red", "black"],
            "answer": sphere_color
        })
        cube_color = param.get("cube_color")
        cube_color = 0 if cube_color == [0.213, 0.8001, 0.0682, 1.0] else 1
        qas.append({
            "q_type": "obj_color",
            "question": "What is the color of the rectangular object?",
            "options": ["green", "white"],
            "answer": cube_color
        })
        cam_loc = param.get("cam_loc")
        cam_loc = 0 if cam_loc == [-8.40339, 16.8564, 8.03589] else 1 if cam_loc == [14.0975, -0.491662, 11.8147] else 2
        options = ["right", "upward", "left"]
        # fluid가 어떤 위치에서부터 object에 접근하는지
        qas.append({
            "q_type": "fluid_direction",
            "question": "From which location does the fluid approach the objects?",
            "options": options,
            "answer": cam_loc
        })
        # obj와 fluid가 충돌하는 순서
        loc = param.get("cube_loc")
        loc = 0 if loc[0] == 0.0 else 1
        qas.append({
            "q_type": "collision",
            "question": "Which object does the fluid collide with first?",
            "options": ["spherical", "rectangular"],
            "answer": loc
        })
    else:
        raise ValueError(f"Unknown scene type: {type}")

    return qas


def get_mixed_scene_qa(param1, param2, type):
    if type == "free_fall":
        # flow_size
        size1 = param1.get("flow_size")
        size2 = param2.get("flow_size")
        return {
            "q_type": "flow_amount",
            "question": "Which scene has a larger amount of fluid released?",
            "options": ["left", "right", "equal"],
            "answer": 0 if size1[0] > size2[0] else 1 if size1[0] < size2[0] else 2
        }
    elif type == "ripple":
        # water_size
        size1 = param1.get("water_size")
        size2 = param2.get("water_size")
        return {
            "q_type": "fluid_amount",
            "question": "Which scene has a larger droplet of fluid released?",
            "options": ["left", "right", "equal"],
            "answer": 0 if size1 > size2 else 1 if size1 < size2 else 2
        }
    elif type == "mixed":
        viscosity1 = param1.get("viscosity")
        viscosity2 = param2.get("viscosity")
        viscosity1 = 0 if viscosity1 < 0.0001 else 1 if viscosity1 < 0.001 else 2 if viscosity1 < 0.01 else 3
        viscosity2 = 0 if viscosity2 < 0.0001 else 1 if viscosity2 < 0.001 else 2 if viscosity2 < 0.01 else 3
        if viscosity1 == viscosity2:
            return None
        return {
            "q_type": "fluid_viscosity",
            "question": "Which scene has a higher viscosity of fluid?",
            "options": ["left", "right"],
            "answer": 0 if viscosity1 > viscosity2 else 1
        }
    else:
        raise ValueError(f"Unknown scene type: {type}")