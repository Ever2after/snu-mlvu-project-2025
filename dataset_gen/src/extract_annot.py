import os
import ast
import json
from scene_annot import extract_annotation

current_dir = os.path.dirname(os.path.abspath(__file__))
datagen_dir = os.path.normpath(os.path.join(current_dir, os.pardir))
scene_dir = os.path.normpath(os.path.join(current_dir, os.pardir, "scene"))

videos_dataset = []
annots = []

scene_types = ["obj_interaction", "slope", "ripple"]
annot_dir = ["output_1", "output_2", "output"]

for scene_type in scene_types:
    base_dir = os.path.join(scene_dir, scene_type)
    for subfolder in os.listdir(base_dir):
        if subfolder not in annot_dir:
            continue
        sub_path = os.path.join(base_dir, subfolder)
        if not os.path.isdir(sub_path):
            continue  

        i = 0
        while True:
            log_path = os.path.join(sub_path, str(i), "params.log")
            if not os.path.isfile(log_path):
                break

            with open(log_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                try:
                    params_dict = ast.literal_eval(content)
                except (ValueError, SyntaxError):
                    try:
                        params_dict = json.loads(content)
                    except json.JSONDecodeError:
                        print(f"Warning: cannot parse '{log_path}'.")
                        params_dict = {}

            # annotation 생성
            annotation = extract_annotation(params_dict, scene_type)

            only_annot = {f"{i}" : annotation}
            annots.append(only_annot)

            video_entry = {
                "video": f"{scene_type}/{subfolder}/{i}/render.mkv", #비디오 제목에 따라 지정
                "conversations": [
                    {
                        "from": "human",
                        "value": "<video>\nDescribe the video."
                    },
                    {
                        "from": "gpt",
                        "value": annotation
                    }
                ]
            }
            videos_dataset.append(video_entry)
            i += 1
    # 파일별 annotation 저장, 예) ouput file 하나 내에 모든 i에 대해 annot만 저장
    #  json_path = os.path.join(sub_path, "annotation.json")
    # with open(json_path, "w", encoding="utf-8") as out_f:
        #    json.dump(annots, out_f, ensure_ascii=False, indent=4)


json_path = os.path.join(datagen_dir, "video.json")
with open(json_path, "w", encoding="utf-8") as out_f:
    json.dump(videos_dataset, out_f, ensure_ascii=False, indent=4)

print(f"Saved {len(annots)} entries to {json_path}")
