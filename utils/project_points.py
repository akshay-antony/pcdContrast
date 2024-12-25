import open3d as o3d
import numpy as np
import cv2


def pointcloud_to_image(pcd_file, extrinsic_params, width=800, height=600):
    """
    Project a point cloud into a 2D image.

    Parameters:
        pcd_file (str): Path to the point cloud file.
        extrinsic_params (np.ndarray): 4x4 matrix defining the extrinsic transformation.
        width (int): Width of the output image.
        height (int): Height of the output image.

    Returns:
        np.ndarray: The rendered 2D image as a NumPy array.
    """
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(pcd_file)

    # Set up the Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(pcd)

    # Access view control and set extrinsic parameters
    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()
    camera_params.extrinsic = extrinsic_params
    view_control.convert_from_pinhole_camera_parameters(camera_params)

    # Capture the screen as an image
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.destroy_window()

    # Convert the image to NumPy array
    image_np = (np.asarray(image) * 255).astype(np.uint8)
    return image_np

# Example usage
if __name__ == "__main__":
    # Path to the point cloud file
    pcd_path = "your_point_cloud.ply"

    # Define extrinsic transformation (camera pose)
    extrinsic_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],  # Adjust Y translation
        [0, 0, 1, 0],  # Adjust Z translation
        [0, 0, 0, 1]
    ])

    # Convert the point cloud to an image
    img = pointcloud_to_image(pcd_path, extrinsic_matrix)

    # Save the image using OpenCV
    cv2.imwrite("pointcloud_image.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print("Image saved as pointcloud_image.png")
