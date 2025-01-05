import glob
import open3d as o3d
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
import matplotlib
import os
import shutil

matplotlib.use("TkAgg")  # Or 'Qt5Agg', 'WXAgg', etc., depending on your system
import matplotlib.pyplot as plt
from tqdm import tqdm

# Function to apply rotations to the point cloud
def apply_rotation_to_pcd(pcd, roll, pitch, yaw):
    # Create a rotation matrix from roll, pitch, and yaw (in degrees)
    rotation_matrix = R.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()
    # Get the points as a numpy array
    points = np.asarray(pcd.points)
    # Apply the rotation matrix to the points
    rotated_points = points @ rotation_matrix.T
    # Update the point cloud with the rotated points
    pcd.points = o3d.utility.Vector3dVector(rotated_points)
    return pcd


def render_pcd_to_image(pcd, roll=0, pitch=0, yaw=0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)

    # Capture the scene
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    # Save the image
    # plt.imshow(np.asarray(image))
    return np.asarray(image)
    # plt.axis("off")
    # plt.savefig(f"roll_{roll}_pitch_{pitch}_yaw_{yaw}.png")
    # plt.show()


# Define the path pattern
path_pattern = "../data/3d/**/*.off"

# Use glob to find all .off files in the subfolders
off_files = glob.glob(path_pattern, recursive=True)


# Function to read and visualize a point cloud from an .off file
def visualize_off_file(file_path, roll=0, pitch=0, yaw=0):
    # Read the .off file
    mesh = o3d.io.read_triangle_mesh(file_path)
    # Convert the mesh to a point cloud
    pcd = mesh.sample_points_uniformly(number_of_points=1000000)
    # Visualize the point cloud with specified roll, pitch, and yaw
    pcd = apply_rotation_to_pcd(pcd, roll, pitch, yaw)
    img = render_pcd_to_image(pcd, roll, pitch, yaw)
    # pil_image = Image.fromarray(img, mode="RGB")
    # pil_image.show()
    # pil_image.save("test.png")
    return img

three_d_prefix = "../data/3d/ModelNet40"
two_d_prefix = "../data/2d/ModelNet40"

# # Visualize the first .off file with custom views
# loop = tqdm(off_files, total=len(off_files))
# for file in loop:
#     print(f"Visualizing file: {file}")
#     # Define roll, pitch, yaw combinations for the 12 viewpoints
#     viewpoints = [
#         (0, 90, 0),  # Top view
#         (0, 45, 0),  # Oblique view 1
#         (0, 45, 120),  # Oblique view 2
#         (0, 45, 240),  # Oblique view 3
#         (0, 0, 0),  # Side view 1
#         (0, 0, 120),  # Side view 2
#         (0, 0, 240),  # Side view 3
#         (0, -45, 0),  # Oblique bottom view 1
#         (0, -45, 120),  # Oblique bottom view 2
#         (0, -45, 240),  # Oblique bottom view 3
#         (0, -90, 0),  # Bottom view 1
#         (0, -90, 120),  # Bottom view 2
#     ]

    
from multiprocessing import Pool, cpu_count

def process_file(file):
    class_name = file.split("/")[-3]
    train_or_test = file.split("/")[-2]
    off_file_name = file.split("/")[-1].split(".")[0]

    output_dir = f"{two_d_prefix}/{class_name}/{train_or_test}/{off_file_name}"
    if os.path.exists(output_dir):
        return

    os.makedirs(output_dir, exist_ok=True)

    viewpoints = [
        (0, 90, 0), (0, 45, 0), (0, 45, 120), (0, 45, 240),
        (0, 0, 0), (0, 0, 120), (0, 0, 240), (0, -45, 0),
        (0, -45, 120), (0, -45, 240), (0, -90, 0), (0, -90, 120),
    ]

    for roll, pitch, yaw in viewpoints:
        img = visualize_off_file(file, roll, pitch, yaw)
        plt.imsave(f"{output_dir}/roll_{roll}_pitch_{pitch}_yaw_{yaw}.png", img)
    print(f"Processed: {file}")
    
if __name__ == "__main__":
    # Use a pool of workers to process files
    with Pool(cpu_count()) as pool:
        # Wrap the iterable in tqdm
        for _ in tqdm(pool.imap(process_file, off_files), total=len(off_files)):
            pass