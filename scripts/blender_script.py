"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.


#MJ: Install
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


Example usage:
    blender -b -P blender_script.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 12 \
        --camera_dist 1.2

Here, input_model_paths.json is a json file containing a list of paths to .glb.
"""

import argparse
import math
import os
import random
import sys
import time
import shutil
import glob
import urllib.request
from typing import Tuple

import bpy
import numpy as np
from mathutils import Vector

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default="./views")
parser.add_argument(
    "--engine", type=str, default="BLENDER_EEVEE", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--num_images", type=int, default=7)
parser.add_argument("--camera_dist", type=float, default=1.5)
parser.add_argument("--cond_polar", type=int)
parser.add_argument("--cond_azimuth", type=int)

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)


def enable_gpus(device_type, use_cpus=False):
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cycles_preferences.refresh_devices()
    devices = cycles_preferences.devices

    if not devices:
        raise RuntimeError("Unsupported device type")

    activated_gpus = []
    for device in devices:
        if device.type == "CPU":
            device.use = use_cpus
        else:
            device.use = True
            activated_gpus.append(device.name)
            print('activated gpu', device.name)

    cycles_preferences.compute_device_type = device_type
    bpy.context.scene.cycles.device = "GPU"

    return activated_gpus


enable_gpus("CUDA")

context = bpy.context
scene = context.scene
render = scene.render

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 320
render.resolution_y = 320
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 32  #MJ: scene.cycles.samples = 128  in objaverse-xl:  Samples determine the number of rays Cycles shoots per pixel. 
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True


def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )


def add_lighting() -> None:
    # delete the default light
    bpy.data.objects["Light"].select_set(True)
    bpy.ops.object.delete()
    # add a new light
    bpy.ops.object.light_add(type="AREA")
    light2 = bpy.data.lights["Area"]
    light2.energy = 30000
    bpy.data.objects["Area"].location[2] = 0.5
    bpy.data.objects["Area"].scale[0] = 100
    bpy.data.objects["Area"].scale[1] = 100
    bpy.data.objects["Area"].scale[2] = 100


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    print(f"MJ: Trying to load object from {object_path}")
    
    if not os.path.exists(object_path):
        print(f"MJ: File not found and return: {object_path}")
        return

    if object_path.endswith(".glb"):
        #MJ: This command is specific to Blender and:
        # Imports a .gltf or .glb file into Blender's scene as 3D objects.
        # It reads the 3D data and ADDS it to Blender’s scene,
        # allowing you to visualize and manipulate the objects within the Blender environment.
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)

    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".gltf"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)    
    else:
        raise ValueError(f"Unsupported file type: {object_path}")

def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj

def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

#MJ: We use normalize_scene() of  objaverse, not objaverse-xl, to make it work in the same way as
# # normalize_scene of kaolin renderer
def normalize_scene() -> None:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        None
    """

    # #MJ: there will typically be only one root object left in get_scene_root_objects(), which is the parent_empty object itself.
    
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    #MJ:  the operation max(bbox_max - bbox_min) finds the largest dimension of the bounding box defined by bbox_min and bbox_max.
    # By setting scale = 1 / max(bbox_max - bbox_min), you’re calculating a scale factor that will shrink the largest dimension to 1 unit.

    for obj in scene_root_objects():
        obj.scale = obj.scale * scale #MJ: The scaling operation (obj.scale = obj.scale * scale) will only be applied to parent_empty, not to the child objects directly.

    #MJ: Blender's data model may not immediately reflect changes made by the script, 
    # especially when performing operations like setting parent-child relationships, moving objects, or modifying transforms.
    bpy.context.view_layer.update() #MJ: ensures that all pending updates are processed.
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    
    #MJ: does change the object frame of parent_empty (or any other root object) to a new location 
    # by modifying its world-space translation. 
    # y default, new objects in Blender are created at the world origin (0, 0, 0) unless specified otherwise.
    #MJ: initially, parent_empty.matrix_world.translation will typically be (0, 0, 0), 
    # meaning that the object is located at the world origin.
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
          #MJ: this code is designed to move the center of the bounding box of the entire scene to the origin of the world frame.
    bpy.ops.object.select_all(action="DESELECT")
    #MJ: => before selecting a specific object or set of objects for a particular operation (e.g., parenting, transforming, or deleting), it's a good practice to clear any prior selection state. 
    # This ensures that the subsequent selections are precise and apply only to the intended objects.
    # When running scripts, especially ones that manipulate scenes or objects, having no objects selected initially 
    # creates a stable and predictable environment. The script’s behavior won't be influenced 
    # by any previous user selections or states, making the script's execution more reliable and consistent.

    # unparent the camera
    #MJ: 
    # When you parent an object to another (in this case, potentially parent the camera to parent_empty), 
    # the child object (the camera) inherits transformations (translation, rotation, scale) from the parent.
    # If you want the camera to have independent movement or transformations, 
    # you need to unparent it by setting its parent to None.
    bpy.data.objects["Camera"].parent = None #Make the camera object as a root node

def setup_camera():
    cam = scene.objects["Camera"] #MJ: get the reference to the scene camera object
    cam.location = (0, 1.2, 0)  #MJ: (x,y,z)

    #MJ: set the fov directly
    cam.data.angle = math.radians(30)  # Set FOV to 30 degrees
    
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    #MJ: This adds a Track To constraint to the camera.
    # A constraint is a mechanism in Blender that limits or guides an object's transformations based on the behavior of another object or an axis.
    # The Track To constraint forces the camera to always point toward a specific target. After this line, the camera is ready to be pointed at something (though the target is not set in this function yet).

    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    #MJ: In Blender, the negative Z axis typically points forward in a camera's local space, so this setting makes the camera face its target by orienting its front side.
    
    cam_constraint.up_axis = "UP_Y"
    #MJ: UP_Y specifies that the camera’s local Y axis should remain aligned with Blender’s global Z axis 
    # (the world’s "up" direction) as it tracks the target, helping maintain a natural, upright orientation for the camera
    
    return cam, cam_constraint

def save_images(object_file: str) -> None:
    """Saves rendered images of the object in the scene."""
    os.makedirs(args.output_dir, exist_ok=True)

    # MJ: Resets the scene to a clean state:
    # Delete everything that isn't part of a camera or a light; the default Cube object in the blender scene is deleted
    reset_scene()
    normalize_scene()          
    add_lighting() #MJ: delete the default light and add the area light
    
    #MJ: Get the references to camera object and camera constraint
    cam, cam_constraint = setup_camera()
    
    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    # MJ: An empty object is an invisible helper object in Blender that doesn’t render 
    # the None argument specifies that the empty object has no associated data, such as a mesh, curve, or other data block type.
    # and is often used as a reference point for positioning or constraints.
    
    scene.collection.objects.link(empty)
    cam_constraint.target = empty
    #MJ: links the empty object to the active scene’s collection, making it part of the scene.
    #MJ: when you create a new empty object in Blender, its location is automatically set to the origin (0, 0, 0) of the world frame by defaul

    scene.use_nodes = True
    tree = scene.node_tree

    for node in tree.nodes:
        tree.nodes.remove(node)

    scene.view_layers["ViewLayer"].use_pass_z = True

    render_layers_node = tree.nodes.new('CompositorNodeRLayers')
    normalize_node = tree.nodes.new("CompositorNodeNormalize")
    invert_node = tree.nodes.new("CompositorNodeInvert")
    viewer_node = tree.nodes.new("CompositorNodeViewer")
    file_output_image_node = tree.nodes.new("CompositorNodeOutputFile")
    file_output_depth_node = tree.nodes.new("CompositorNodeOutputFile")

    # Set base_path to the directory
    object_uid = os.path.basename(object_file).split(".")[0]
    base_path = os.path.join(args.output_dir, object_uid)

    file_output_image_node.base_path = base_path
    file_output_depth_node.base_path = base_path

    #MJ: The fixed 6 view points for the target images (zero123plus frame, the same as the Blender frame);
    azimuths_o = [30, 90, 150, 210, 270, 330]
    elevations_w = [20, -10, 20, -10, 20, -10]

    #MJ: Blender uses the active camera for rendering. 
    # The active camera is the camera that is set as the primary camera in the scene for rendering purposes.
    # This is stored in scene.camera:  scene.objects["Camera"]:
    bpy.context.scene.camera = scene.objects["Camera"]

    def render_from_angle(phi_from_Z_w, theta_from_X_w, camera_dist):
        #MJ: compute the camera position (X,Y,Z) in the world frame    
        point = (
            camera_dist * math.sin(phi_from_Z_w) * math.cos(theta_from_X_w),    # = x
            camera_dist * math.sin(phi_from_Z_w) * math.sin(theta_from_X_w),    # = y
            camera_dist * math.cos(phi_from_Z_w),                               # = z
        )
        #MJ: set the current camera location in the world frame
        
        cam.location = point

        image_file_name = f"{i:03d}.png"
        depth_file_name = f"{i:03d}_depth.png"

        file_output_image_node.file_slots[0].path = image_file_name
        file_output_depth_node.file_slots[0].path = depth_file_name

        # Link nodes
        links = tree.links

        links.new(render_layers_node.outputs['Depth'], normalize_node.inputs['Value'])
        links.new(normalize_node.outputs['Value'], invert_node.inputs['Color'])

        links.new(invert_node.outputs['Color'], viewer_node.inputs['Image'])

        links.new(render_layers_node.outputs['Image'], file_output_image_node.inputs['Image'])
        links.new(invert_node.outputs['Color'], file_output_depth_node.inputs['Image'])

        bpy.ops.render.render(write_still=True) # JA: Rendering is done by this function, using the current active cam
        #MJ: bpy.ops.render.render(write_still=True) renders the scene specified by the active context, which in your case is context.scene.
        #MJ: The write_still=True argument specifies that Blender should save the rendered image to the file path set in scene.render.filepath.

    # MJ: refer to https://github.com/TencentARC/InstantMesh/issues/114
    camera_dist = np.random.uniform(2.5, 5.0, 1)[0]

    # JA: Set the elevation angle (absolute) and azimuth angle (relative) provided from arguments
    phi_from_Z_w = math.radians(args.cond_polar)
    theta_from_X_w = math.radians(args.cond_azimuth)

    starting_theta_from_X_w = theta_from_X_w

    render_from_angle(phi_from_Z_w, theta_from_X_w, camera_dist)

    for i in range(len(azimuths_o)):
        #MJ: 
        camera_dist = 0.5 / np.tan(np.radians(30/2))  #MJ: refer to https://github.com/SUDO-AI-3D/zero123plus/issues/73
        
        theta_from_X_w = starting_theta_from_X_w + math.radians(azimuths_o[i]) #30, 90, 150, 210, 270, 330
        #Conver the relative azimuth angle theta_o to the absolute angle theta_w
        # theta_from_x_o is measured staring from the x axis, on the zx plane:
        
        #MJ: convert elevation angle to polar angle
        phi_from_Z_w = math.radians(90 - elevations_w[i])

        render_from_angle(phi_from_Z_w, theta_from_X_w, camera_dist)

    for file_path in glob.glob(os.path.join(base_path, "*.png*")):
        # Construct the new file name by removing the frame number
        new_file_name = file_path.split('.png')[0] + '.png'
        os.rename(file_path, new_file_name)

def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    # uid = uuid.uuid4()
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path


if __name__ == "__main__":
    try:
        start_i = time.time()
        if args.object_path.startswith("http"):
            local_path = download_object(args.object_path)
        else:
            local_path = args.object_path
            
        save_images(local_path)
        
        end_i = time.time()
        print("Finished", local_path, "in", end_i - start_i, "seconds")

        if args.object_path.startswith("http"):
            os.remove(local_path)
    except Exception as e:
        print("Failed to render", args.object_path)
        print(e)
