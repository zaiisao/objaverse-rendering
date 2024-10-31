import os
import glob
import json
import multiprocessing
import shutil
import subprocess
import time
import struct

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import numpy as np

import torch
import kaolin

import boto3
import tyro
import wandb


@dataclass
class Args:
    workers_per_gpu: int
    """number of workers per gpu"""

    input_models_path: str
    """Path to a json file containing a list of 3D object files"""

    upload_to_s3: bool = False
    """Whether to upload the rendered images to S3"""

    log_to_wandb: bool = False
    """Whether to log the progress to wandb"""

    num_gpus: int = -1
    """number of gpus to use. -1 means all available gpus"""

def load_mesh(mesh_path):
    mesh = kaolin.io.gltf.import_mesh(mesh_path)

    def normalize_mesh(vertices):
        # Compute bounding box
        bbox_min = torch.min(vertices, dim=0).values
        bbox_max = torch.max(vertices, dim=0).values
        
        # Compute scale factor to fit within unit cube
        scale = 1.0 / torch.max(bbox_max - bbox_min)
        
        # Scale vertices
        scaled_vertices = vertices * scale
        
        # Recompute bounding box after scaling
        bbox_min = torch.min(scaled_vertices, dim=0).values
        bbox_max = torch.max(scaled_vertices, dim=0).values

        # Compute translation to center mesh at origin
        bbox_center = (bbox_min + bbox_max) / 2.0

        # Translate vertices
        vertices_from_bbox_center = scaled_vertices - bbox_center

        # JA: scaled_vertices on the right-hand side refers to the coordiantes of the vertices from
        # the world coordinate system

        return vertices_from_bbox_center

    mesh_vertices = normalize_mesh(mesh.vertices)
    mesh_faces = mesh.faces
    mesh_uvs = mesh.uvs
    mesh_face_uvs_idx = mesh.face_uvs_idx

    return mesh_vertices, mesh_faces, mesh_uvs, mesh_face_uvs_idx

def convert_path(original_path):
    # Convert the original path to a Path object
    path_obj = Path(original_path)
    
    # Extract the parts of the path
    parts = path_obj.parts
    
    # Replace 'glbs' with 'kaolin-tensors'
    new_parts = list(parts)
    new_parts[-3] = 'zero123plus-dataset'
    
    # Remove the .glb extension
    new_parts[-1] = new_parts[-1].replace('.glb', '')
    
    # Construct the new path
    new_path = Path(*new_parts)
    
    return new_path

def sample_azimuth_theta():
    # MJ: In Blender, the azimuth angle is measured from the X axis
    # First, decide which range to sample from based on the probabilities

    if np.random.rand() < 0.7:
        # 70% chance to sample the "frontal" azimuth angles around the forward axis Y.
        return np.random.randint(180, 360)
    else:
        # 30% chance to sample between 90° and 270°
        return np.random.randint(0, 180)

def sample_polar_phi():
    if np.random.rand() < 0.7:
        # 70% chance to sample the "middle" polar angles measured from the vertical Z axis
        return np.random.randint(60, 120) #MJ: 120 = 90 + 30
    else:
        # 30% chance to sample the upper or bottom polar angles
        if np.random.rand() < 0.5:
            return np.random.randint(0, 60)
        else:
            return np.random.randint(120,180) #MJ: 120 = 90 + 30

def save_tensor_to_path(tensor, tensor_name, original_path):
    # Convert the original path to the new path
    new_path = convert_path(original_path)
    
    # Ensure the directory exists
    os.makedirs(new_path, exist_ok=True)
    
    # Define the tensor file path
    tensor_file_path = new_path / f'{tensor_name}.pt'
    
    # Save the tensor to the file
    torch.save(tensor, tensor_file_path)
    
    print(f"Tensor saved to {tensor_file_path}")

def save_int_as_bin(integer, bin_name, original_path):
    # Convert the original path to the new path
    new_path = convert_path(original_path)
    
    # Ensure the directory exists
    os.makedirs(new_path, exist_ok=True)
    
    # Define the tensor file path
    tensor_file_path = new_path / f'{bin_name}.bin'

    binary_data = struct.pack('h', integer)

    with open(tensor_file_path, 'wb') as file:
        file.write(binary_data)

    with open(tensor_file_path, 'rb') as file:
        binary_data = file.read()
        decoded_azimuth = struct.unpack('h', binary_data)[0]

def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    gpu: int,
    s3: Optional[boto3.client],
    display_num: int,
) -> None:
    while True:
        item = queue.get()
        if item is None:
            break

        # Perform some operation on the item
        print(item, gpu)

        result_path = convert_path(item)

        cond_polar = sample_polar_phi()
        cond_azimuth = sample_azimuth_theta()

        command = (
            f"xvfb-run -n {display_num}"
            f" blender-3.2.2-linux-x64/blender -b -P scripts/blender_script.py --"
            f" --object_path {item} --output_dir {result_path}/.."
            f" --cond_polar {cond_polar} --cond_azimuth {cond_azimuth}"
        )
        subprocess.run(command, shell=True)

        try:
            mesh_vertices, mesh_faces, mesh_uvs, mesh_face_uvs_idx = load_mesh(item)
            if mesh_vertices is None or mesh_faces is None or mesh_uvs is None or mesh_face_uvs_idx is None:
                raise ValueError

            save_tensor_to_path(mesh_vertices, "mesh_vertices", item)
            save_tensor_to_path(mesh_faces, "mesh_faces", item)
            save_tensor_to_path(mesh_uvs, "mesh_uvs", item)
            save_tensor_to_path(mesh_face_uvs_idx, "mesh_face_uvs_idx", item)

            save_int_as_bin(cond_azimuth, "azimuth", item)
        except:
            print(f"Skipping mesh tensor saving of {item}")

        if args.upload_to_s3:
            if item.startswith("http"):
                uid = item.split("/")[-1].split(".")[0]
                for f in glob.glob(f"views/{uid}/*"):
                    s3.upload_file(
                        f, "objaverse-images", f"{uid}/{f.split('/')[-1]}"
                    )
            # remove the views/uid directory
            shutil.rmtree(f"views/{uid}")

        with count.get_lock():
            count.value += 1

        queue.task_done()


if __name__ == "__main__":
    args = tyro.cli(Args)

    s3 = boto3.client("s3") if args.upload_to_s3 else None
    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    if args.log_to_wandb:
        wandb.init(project="objaverse-rendering", entity="prior-ai2")

    # Start worker processes on each of the GPUs
    display_num_base = 1000
    for gpu_i in range(args.num_gpus):
        for worker_i in range(args.workers_per_gpu):
            worker_i = gpu_i * args.workers_per_gpu + worker_i
            display_num = display_num_base + worker_i
            process = multiprocessing.Process(
                target=worker, args=(queue, count, gpu_i, s3, display_num)
            )
            process.daemon = True
            process.start()

    # Add items to the queue
    with open(args.input_models_path, "r") as f:
        model_paths = json.load(f)
    for item in model_paths.values():
        queue.put("/home/sogang/mnt/db_1/jaehoon/objaverse/" + item)

    # update the wandb count
    if args.log_to_wandb:
        while True:
            time.sleep(5)
            wandb.log(
                {
                    "count": count.value,
                    "total": len(model_paths),
                    "progress": count.value / len(model_paths),
                }
            )
            if count.value == len(model_paths):
                break

    # Wait for all tasks to be completed
    queue.join()

    # Add sentinels to the queue to stop the worker processes
    for i in range(args.num_gpus * args.workers_per_gpu):
        queue.put(None)
