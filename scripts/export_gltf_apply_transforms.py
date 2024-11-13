import argparse
import sys
import bpy

parser = argparse.ArgumentParser()
parser.add_argument(
    "--import_path",
    type=str,
    required=True,
    help="Path to the import file",
)
parser.add_argument(
    "--export_path",
    type=str,
    required=True,
    help="Path to the export file",
)


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

def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def export_gltf_apply_transforms(input_path, export_path):
    reset_scene()

    bpy.ops.import_scene.gltf(filepath=input_path, merge_vertices=True, 
                                      guess_original_bind_pose=False, bone_heuristic='BLENDER') 

    bpy.ops.object.select_all(action='SELECT')

    # MJ: The following operation is NEEDED for the kaolin render to work as intended.
    # The transform_apply operation works at the object level, meaning:
    # For an armature, it applies transformations to the entire armature object, that is the root node of the
    # skeletal model but does not affect individual bones  or their rest pose.
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    bpy.ops.export_scene.gltf(
        filepath=export_path,
        export_format='GLB',
        export_yup=True,          # Ensure Y-up orientation
        export_apply=True,        # Apply transformations during export
        export_normals=True,
        export_materials='EXPORT',
        export_lights=False,
        export_cameras=False,
        export_animations=False   # Disable animations in export
    )

if __name__ == "__main__":
    export_gltf_apply_transforms(args.import_path, args.export_path)
