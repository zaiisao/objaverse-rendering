import os
import json
import sys

def create_glb_json(folder_paths):
    glb_dict = {}
    
    for folder_path in folder_paths:
        # Ensure the base_path includes the 'glbs/' prefix
        base_path = os.path.join("glbs", os.path.basename(os.path.normpath(folder_path)))
        
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".glb"):
                    file_id = os.path.splitext(file)[0]
                    relative_path = os.path.join(base_path, file).replace("\\", "/")
                    glb_dict[file_id] = relative_path
    
    # Create a single JSON file named 'combined.json'
    output_json_path = "000-000_000-009.json"

    with open(output_json_path, 'w') as json_file:
        json.dump(glb_dict, json_file, indent=4)

    print(f"JSON file created at {output_json_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py path/to/objaverse/glbs/000-000/ [path/to/another/folder/ ...]")
    else:
        folder_paths = sys.argv[1:]
        create_glb_json(folder_paths)