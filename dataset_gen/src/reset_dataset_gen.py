import os
import sys
import shutil
os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

save_file = ["meshes", "texture", "sampleconf.py", "scene.blend"]

with open(f"../scene/_ignore.txt") as f:
    ignore_list = f.readlines()
    for i in range(len(ignore_list)):
        ignore_list[i] = ignore_list[i].replace("\n", "")

for i in os.listdir("../scene"):
    if i == "_ignore.txt" or i in ignore_list: continue
    print(f"Current scene: {i}\n\n")

    for content in os.listdir(f"../scene/{i}"):
        if content not in save_file:
            if os.path.isdir(f"../scene/{i}/{content}"):
                shutil.rmtree(f"../scene/{i}/{content}")
            else:
                os.remove(f"../scene/{i}/{content}")