import os
import numpy as np
import trimesh
from tqdm import tqdm
import open3d as o3d


def write_np_pcd(filename, num_points=500000, output_dir="../data/ShapeNetCore.v2_3d"):
    # Load the mesh
    output_filename = filename.replace(".obj", ".npy")
    if os.path.exists(output_filename):
        print(f"File already exists: {output_filename}")
        return
    mesh = trimesh.load(filename)
    # Handle Scene object
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    # Sample points from the mesh
    pcd = mesh.sample(num_points)
    pcd_np = np.asarray(pcd)
    np.save(output_filename, pcd_np)


if __name__ == "__main__":
    # Define the directory containing the .obj files
    input_dir = "../data/ShapeNetCore.v2_3d"
    total_files = 0
    for class_name in os.listdir(input_dir):
        # check if it is a directory
        if not os.path.isdir(os.path.join(input_dir, class_name)):
            continue
        class_folder = os.path.join(input_dir, class_name)
        loop = tqdm(
            os.listdir(class_folder),
            desc=f"Processing {class_name}",
            total=len(os.listdir(class_folder)),
            colour="green",
        )
        for instance_folder in loop:
            instance_folder = os.path.join(class_folder, instance_folder, "models")
            # print(f"instance folder: {instance_folder}")
            for filename in os.listdir(instance_folder):
                if filename.endswith(".obj"):
                    total_files += 1
                    print(f"Processing: {instance_folder.split('/')[-2]}")
                    write_np_pcd(
                        os.path.join(instance_folder, filename), num_points=4096 * 4
                    )
    print(f"Total files processed: {total_files}")
