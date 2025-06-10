def _flow_des(a):
    size = a["flow_size"]
    vs = a["viscosity"]
    if vs != "high":
        s = ""
        if size == "big":
            s = "The fluid impacts the container, rapidly filling it and spilling over the edges."
        else:
            s = "The fluid impacts the container and gradually fills it."
        if a["flow_loc"] == "corner":
            s = s + " It first pours out at the container’s corners."
        return s
    else:
        return "The fluid slowly accumulates in the container, spreading at an extremely slow pace."

def _pi(a, scene_type):
    if scene_type == "free_fall":
        reason = {
            "low": (
                "the fluid spreads smoothly across the container without coalescing, "
                "showing minimal surface tension as it gently overflows"
            ),
            "medium": (
                "the fluid collects briefly before spreading evenly across the surface"
            ),
            "high": (
                "the fluid holds its shape and piles up with minimal spreading"
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
        

    if type == "free_fall":
        #param: flow size, flow loc, color, sink_metal
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
            f"Visual Perception: "
            f"There is a {info['container_description']} container.\n"
            f"A fluid is released from a point directly above the container’s {info["flow_loc"]}, "
            f"and falls freely under gravity.\n" 
            f"{_flow_des(info)} \n"
            f"Physical Inference: "
            f"{_pi(info, type)}"
        )
        return annot
    else:
        print("Undefined scene:", type)
        return ""
