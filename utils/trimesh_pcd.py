import matplotlib

matplotlib.use("Agg")
import gc
import json
import os
import shutil
import time

# matplotlib.use('TkAgg')  # Or 'Qt5Agg', 'GTK3Agg', etc., depending on your system
from io import BytesIO
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib import cm
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def apply_rotation_to_pcd_trimesh(pcd, roll, pitch, yaw):
    # Create a rotation matrix from roll, pitch, and yaw (in degrees)
    rotation_matrix = R.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()
    # Apply the rotation to the points
    pcd.vertices = np.dot(pcd.vertices, rotation_matrix.T)
    return pcd


def compute_colors_by_depth(points, colormap=cm.viridis):
    """
    Compute colors for points based on their depth (distance from the camera or origin).

    Parameters:
        points (np.ndarray): Array of shape (N, 3) representing point cloud coordinates.
        colormap (matplotlib.colors.Colormap): Colormap to use for coloring.

    Returns:
        np.ndarray: Array of shape (N, 3) with RGB colors for each point.
    """
    # Compute depth as the z-coordinate (or distance from origin)
    depths = np.linalg.norm(points, axis=1)  # Euclidean distance from origin

    # Normalize depths to the range [0, 1]
    normalized_depths = (depths - depths.min()) / (depths.max() - depths.min())

    # Map normalized depths to colors
    colors = colormap(normalized_depths)[:, :3]  # Get RGB values (exclude alpha)
    return (colors * 255).astype(np.uint8)  # Convert to 8-bit RGB


def mesh_to_pcd_with_depth_colors(mesh, num_points=1000):
    """
    Convert a trimesh mesh to a point cloud and colorize it based on depth.

    Parameters:
        mesh (trimesh.Trimesh): Input mesh.
        num_points (int): Number of points to sample from the mesh surface.

    Returns:
        trimesh.points.PointCloud: Colored point cloud.
    """
    # Sample points from the mesh
    points = mesh.sample(num_points)

    # Compute colors based on depth
    colors = compute_colors_by_depth(points)

    # Create a point cloud with colors
    pcd = trimesh.points.PointCloud(points, colors=colors)
    return pcd


def mesh_to_pcd(mesh, num_points=1000):
    """
    Convert a trimesh mesh to a point cloud.

    Parameters:
        mesh (trimesh.Trimesh): The input mesh.
        num_points (int): Number of points to sample from the mesh surface.

    Returns:
        trimesh.points.PointCloud: The resulting point cloud.
    """
    # Sample points from the surface of the mesh
    points = mesh.sample(num_points)

    # Assign colors if the mesh has vertex colors (optional)
    if mesh.visual.kind == "vertex" and hasattr(mesh.visual, "vertex_colors"):
        vertex_colors = mesh.visual.vertex_colors[:, :3]  # Exclude alpha if present
        face_colors = np.mean(
            vertex_colors[mesh.faces], axis=1
        )  # Average color per face
        sampled_colors = face_colors[np.random.randint(0, len(face_colors), num_points)]
    else:
        # Default color: white
        sampled_colors = np.tile([255, 255, 255], (num_points, 1))

    # Create a point cloud
    pcd = trimesh.points.PointCloud(points, colors=sampled_colors)
    return pcd


def render_pcd_with_intrinsics(pcd, image_resolution=(512, 512), intrinsic_matrix=None):
    # Define default intrinsic matrix if not provided
    if intrinsic_matrix is None:
        fx = fy = 10  # Focal length in pixels
        cx, cy = image_resolution[0] // 2, image_resolution[1] // 2  # Image center
        intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Convert the point cloud to a NumPy array
    points = np.array(pcd.vertices)

    # Filter out points behind the camera (Z <= 0)
    points = points[points[:, 2] > 0]

    # Project points to the image plane
    points_normalized = points[:, :2] / points[:, 2].reshape(-1, 1)  # X/Z, Y/Z
    points_homo = np.hstack(
        [points_normalized, np.ones((points.shape[0], 1))]
    )  # Add 1 for homogeneous coordinates
    projected_points = (intrinsic_matrix @ points_homo.T).T  # Apply intrinsic matrix

    # Extract pixel coordinates
    pixel_coords = projected_points[:, :2]  # Ignore homogeneous scaling
    pixel_coords = pixel_coords.astype(int)

    # Create an empty image
    image = np.zeros((image_resolution[1], image_resolution[0], 3), dtype=np.uint8)

    # Fill in the pixels (simple approach: color white)
    for x, y in pixel_coords:
        if 0 <= x < image_resolution[0] and 0 <= y < image_resolution[1]:
            image[y, x] = [255, 255, 255]  # White point

    return image


def render_pcd_to_image_trimesh(pcd, image_resolution=(512, 512)):
    # Create a scene with the point cloud
    scene = trimesh.Scene(pcd)
    # Render the scene to an image
    png_data = scene.save_image(resolution=image_resolution, visible=False)
    # image = render_pcd_with_intrinsics(pcd)

    # Convert PNG data to a NumPy array
    image = plt.imread(BytesIO(png_data), format="png")
    return image


def process_file(
    filename,
    two_d_prefix="../data/ShapeNetCore.v2_2d",
    three_d_prefix="../data/ShapeNetCore.v2_3d",
    viewpoints=[
        (0, 90, 0),
        (0, 45, 0),
        (0, 45, 120),
        (0, 45, 240),
        (0, 0, 0),
        (0, 0, 120),
        (0, 0, 240),
        (0, -45, 0),
        (0, -45, 120),
        (0, -45, 240),
        (0, -90, 0),
        (0, -90, 120),
    ],
    num_points=500000,
):
    class_name = filename.split("/")[0]
    pcd_file_name = filename.split("/")[1]
    output_dir = f"{two_d_prefix}/{class_name}/{pcd_file_name}"
    if os.path.exists(output_dir):
        # check for existing files of 12 viewpoints
        existing_files = os.listdir(output_dir)
        if len(existing_files) >= 12:
            print(f"Skipping: {filename}")
        else:
            print(f"Number of files in directory: {len(existing_files)}")
            print(f"Removing incomplete directory: {filename}")
            shutil.rmtree(output_dir)
        return
    os.makedirs(output_dir, exist_ok=True)
    # insert models  before models_normalized.obj
    filename_formatted = filename.replace(
        "model_normalized.obj", "models/model_normalized.obj"
    )
    mesh = trimesh.load(os.path.join(three_d_prefix, filename_formatted))

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.to_geometry())

    pcd = mesh_to_pcd_with_depth_colors(mesh, num_points=num_points)

    for roll, pitch, yaw in viewpoints:
        rotated_pcd = apply_rotation_to_pcd_trimesh(
            pcd, roll=roll, pitch=pitch, yaw=yaw
        )
        image = render_pcd_to_image_trimesh(rotated_pcd)
        plt.imsave(f"{output_dir}/roll_{roll}_pitch_{pitch}_yaw_{yaw}.png", image)

    del mesh, pcd, rotated_pcd, image
    gc.collect()
    print(f"Processed: {filename}")


def main():
    base_dir = "../data/ShapeNetCore.v2_3d"
    split_files_json = "../data/ShapeNetCore.v2_3d/split_files.json"
    with open(split_files_json, "r") as f:
        split_files = json.load(f)
    train_files = split_files["train"]
    test_files = split_files["test"]
    train_files = split_files["test"]
    print(f"Found {len(train_files)} training files and {len(test_files)} test files")
    # Use all available CPUs for parallel processing
    num_processes = cpu_count() - 3
    # num_processes = 1
    with Pool(num_processes) as pool:
        with tqdm(total=len(train_files), desc="Processing Files", unit="file") as pbar:
            for _ in pool.imap_unordered(process_file, train_files):
                pbar.update(1)


if __name__ == "__main__":
    main()
