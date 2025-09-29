import multiprocessing
import objaverse

import os
import json
import urllib
import gzip
import shutil
import psutil

processes = multiprocessing.cpu_count()
uids = objaverse.load_uids()

def check_disk_space(required_space_gb):
    disk_usage = psutil.disk_usage('/')
    available_gb = disk_usage.free / (1024 ** 3)
    return available_gb > required_space_gb

def find_highest_count(base_folder):
    highest_count = 0

    for folder_name in os.listdir(base_folder):
        if folder_name.startswith("000-"):
            try:
                # JA: Extract the number after "000-"
                count = int(folder_name.split('-')[1])
                if count > highest_count:
                    highest_count = count
            except (IndexError, ValueError):
                # Ignore invalid folder names
                continue

    # JA: The next number after the highest count
    next_number = highest_count + 1

    # JA: Check for gaps; this is not necessary for the final task of getting the next folder index
    # but it is a good check just in case we are missing intermediate folders.
    for i in range(1, highest_count + 1):
        expected_folder = f"000-{i:03}"
        if expected_folder not in os.listdir(base_folder):
            print(f"Missing folder: {expected_folder}")
            return None

    return next_number

base_folder_path = os.path.join(os.path.expanduser("~"), 'mnt/db_1/jaehoon/objaverse/zero123plus-dataset')
starting_index = find_highest_count(base_folder_path)

while True:
    # JA: The object_paths file loading code was retrieved from _load_object_paths function in
    # /home/sogang/mnt/db_2/anaconda3/envs/objaverse/lib/python3.10/site-packages/objaverse/__init__.py
    object_paths_file = "object-paths.json.gz"
    local_path = os.path.join(os.path.expanduser("~"), ".objaverse/hf-objaverse-v1", object_paths_file)

    if not os.path.exists(local_path):
        hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{object_paths_file}"
        # wget the file and put it in local_path
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        urllib.request.urlretrieve(hf_url, local_path)

    with gzip.open(local_path, "rb") as f:
        object_paths = json.load(f)
    # JA: End of code retrieved from _load_object_paths

    # Format the starting index as a three-digit number
    folder_name = f"glbs/000-{starting_index:03}"

    # Filter the dictionary to include only the entries with the desired folder name
    filtered_object_paths = {
        key: value for key, value in object_paths.items() if value.startswith(folder_name)
    }

    objects = objaverse.load_objects(
        uids = filtered_object_paths.keys(),
        download_processes=multiprocessing.cpu_count()
    )

    BASE_PATH = os.path.join(os.path.expanduser("~"), ".objaverse")
    _VERSIONED_PATH = os.path.join(BASE_PATH, "hf-objaverse-v1")

    dir_to_move = os.path.join(_VERSIONED_PATH, folder_name)
    new_mesh_folder = os.path.join(os.path.expanduser("~"), "mnt/db_1/jaehoon/objaverse/glbs")
    shutil.move(dir_to_move, new_mesh_folder)

    folder_path = os.path.join(new_mesh_folder, f"000-{starting_index:03}")
    glb_dict = {}

    # Ensure the base_path includes the 'glbs/' prefix
    base_path = os.path.join("glbs", os.path.basename(os.path.normpath(folder_path)))

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".glb"):
                file_id = os.path.splitext(file)[0]
                relative_path = os.path.join(base_path, file).replace("\\", "/")
                glb_dict[file_id] = relative_path

    output_json_path = "/home/sogang/jaehoon/objects_temp.json"

    with open(output_json_path, 'w') as json_file:
        json.dump(glb_dict, json_file, indent=4)

    import subprocess
    command = (
        f"python /home/sogang/jaehoon/objaverse-rendering/scripts/distributed.py"
        f" --num_gpus 4 --workers_per_gpu 4 --input_models_path {output_json_path}"
    )
    subprocess.run(command, shell=True)

    if os.path.exists(output_json_path):
        os.remove(output_json_path)

    if os.path.exists(new_mesh_folder):
        os.remove(new_mesh_folder)

    starting_index += 1
