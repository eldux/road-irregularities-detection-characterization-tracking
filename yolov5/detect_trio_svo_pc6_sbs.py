# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights1 yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights1 yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlpackage          # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import math
import os
import platform
import sys
import time
from pathlib import Path
from functools import reduce

import numpy as np
import torch
import copy
from collections import defaultdict
from typing import Tuple, Optional
import open3d as o3d
from sympy import false

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode

objects = []
new_id = 0
last_timestamp = 0

def apply_transformation_to_depth_map(depth_map: np.ndarray,
                                      transformation: np.ndarray) -> np.ndarray:
    """
    Applies a 4x4 transformation matrix to a 2D depth map by treating each depth value
    as the Z coordinate of a point on the optical axis (i.e. [0, 0, z, 1]^T) and
    ignoring any translation in the transformation.

    Parameters:
        depth_map (np.ndarray): 2D array (H, W) containing depth values.
        transformation (np.ndarray): A 4x4 transformation matrix.
                                     Only the rotation/scaling part affecting Z is used.

    Returns:
        np.ndarray: A 2D array of the same shape as depth_map, where each value is recalculated
                    as the transformed Z coordinate.
    """
    # For a point [0, 0, z, 1]^T, the transformed point is:
    # [x', y', z', 1]^T = transformation @ [0, 0, z, 1]^T
    # Ignoring translation, the new Z is given by:
    # new_z = transformation[2,2] * z
    new_depth_map = transformation[2, 2] * depth_map
    return new_depth_map

def filter_points_to_np_array(point_cloud: np.ndarray, max_depth: float) -> np.ndarray:
    """
    Filters an organized point cloud to remove points that contain NaN or Inf in any coordinate,
    or have a Z coordinate greater than max_depth. Returns a 2D NumPy array of valid 3D points (X, Y, Z).

    The input point cloud is expected to be a 3D NumPy array of shape (H, W, d), where d >= 3.

    Parameters:
        point_cloud (np.ndarray): Organized point cloud with shape (H, W, d), where d >= 3.
        max_depth (float): Maximum allowed value for the Z coordinate.

    Returns:
        np.ndarray: A 2D array of valid 3D points with shape (N, 3), where N is the number of valid points.
    """
    # Validate the input shape.
    if point_cloud.ndim != 3 or point_cloud.shape[2] < 3:
        raise ValueError("Input point cloud must be a 3D array with shape (H, W, d) and d >= 3.")

    # Flatten the point cloud to shape (N, d), where N = H * W.
    points = point_cloud.reshape(-1, point_cloud.shape[2])

    # Create a mask that filters out points with any non-finite values (NaN or Inf)
    finite_mask = np.isfinite(points).all(axis=1)




    # Create a mask for points with Z coordinate less than or equal to max_depth (assumed at index 2)
    depth_mask = points[:, 2] >= max_depth

    # Combine the masks to get valid points
    valid_mask = finite_mask & depth_mask

    # Filter the points and extract only the first three channels (X, Y, Z)
    valid_points_3d = points[valid_mask][:, :3]

    # Return the filtered points as a 2D NumPy array
    return valid_points_3d

def extract_median_xyz(bbox_point_cloud: np.ndarray) -> Tuple[float, float, float]:
    """
    Extracts the median X, Y, and Z coordinates from a cropped point cloud after filtering out
    points with non-finite values or with a Z value greater than max_depth.

    Parameters:
        bbox_point_cloud (np.ndarray): Cropped point cloud with shape (H, W, d), where
                                       the first three channels represent the XYZ coordinates.

    Returns:
        Tuple[float, float, float]: The median X, Y, and Z coordinates.

    Raises:
        ValueError: If no valid points remain after filtering.
    """
    # Filter the point cloud to a 2D array of valid 3D points.
    bbox_point_cloud_filtered_unstructured = filter_points_to_np_array(bbox_point_cloud, max_depth=-30)

    if bbox_point_cloud_filtered_unstructured.size == 0:
        return 0, 0, 0

    # Calculate medians along the points axis.
    median_x = np.median(bbox_point_cloud_filtered_unstructured[:, 0])
    median_y = np.median(bbox_point_cloud_filtered_unstructured[:, 1])
    median_z = np.median(bbox_point_cloud_filtered_unstructured[:, 2])

    return median_x, median_y, median_z

def display_point_cloud_measure(point_cloud: o3d.geometry.PointCloud):
    """
    Open a separate Open3D window where you can pick points and print their XYZ.
    Also prints delta and Euclidean distance if >=2 points were picked.

    Controls (Open3D VisualizerWithEditing):
      - Press 'P' to enter point-picking mode
      - Hold Shift + Left Click to pick points
      - Press 'Q' or close window to finish
    """
    if point_cloud is None or len(point_cloud.points) == 0:
        print("[3D] No point cloud to display.")
        return

    # ---- COLOR SAFETY CHECKS (non-destructive) ----
    # Open3D shows colors if point_cloud.colors exists and matches points length (Nx3 floats in [0,1]).
    has_colors = point_cloud.has_colors()
    if not has_colors:
        print("[3D] Warning: point cloud has no colors attached (point_cloud.colors is empty).")
    else:
        # Defensive: sometimes colors exist but size mismatch; Open3D will ignore or error depending on version.
        n_pts = np.asarray(point_cloud.points).shape[0]
        n_col = np.asarray(point_cloud.colors).shape[0]
        if n_pts != n_col:
            print(f"[3D] Warning: colors count ({n_col}) != points count ({n_pts}). Colors may not display.")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="3D Point Cloud (Pick points to measure)", width=1400, height=900)
    vis.add_geometry(point_cloud)

    # ---- RENDER OPTIONS: make colors obvious ----
    opt = vis.get_render_option()
    opt.point_size = 2.0                 # increase if points look too thin
    opt.background_color = np.array([0, 0, 0], dtype=np.float32)  # black background helps see colors
    # If you ever use normals + lighting, keep this. For pure colors it's fine either way.
    opt.light_on = True

    # Fit camera to bounding box
    ctr = vis.get_view_control()
    bbox = point_cloud.get_axis_aligned_bounding_box()
    ctr.set_lookat(bbox.get_center())
    ctr.set_front([0, 0, 1])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.25)

    print("\n[3D] Controls: press 'P' -> Shift+Click points -> 'Q' to quit the 3D window.")
    vis.run()

    picked = vis.get_picked_points()
    vis.destroy_window()

    pts = np.asarray(point_cloud.points)
    cols = np.asarray(point_cloud.colors) if point_cloud.has_colors() else None

    if not picked:
        print("[3D] No points picked.")
        return

    print(f"[3D] Picked {len(picked)} point(s):")
    picked_xyz = []
    for k, idx in enumerate(picked, start=1):
        x, y, z = pts[idx]
        picked_xyz.append((x, y, z))

        if cols is not None and idx < len(cols):
            r, g, b = cols[idx]
            # show both float [0..1] and 8-bit [0..255] for sanity
            r8, g8, b8 = int(round(r * 255)), int(round(g * 255)), int(round(b * 255))
            print(f"  #{k} idx={idx} -> X={x:.6f}, Y={y:.6f}, Z={z:.6f} | RGB={r:.3f},{g:.3f},{b:.3f} ({r8},{g8},{b8})")
        else:
            print(f"  #{k} idx={idx} -> X={x:.6f}, Y={y:.6f}, Z={z:.6f}")

    # If user picked >=2 points, print delta and distance (first two)
    if len(picked_xyz) >= 2:
        x1, y1, z1 = picked_xyz[0]
        x2, y2, z2 = picked_xyz[1]
        dx, dy, dz = (x2 - x1), (y2 - y1), (z2 - z1)
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        print("[3D] Between first two picked points:")
        print(f"  dX={dx:.6f}, dY={dy:.6f}, dZ={dz:.6f},  distance={dist:.6f}")

def apply_transformation_to_zed_point_cloud(point_cloud_data: np.ndarray,
                                            transformation: np.ndarray) -> np.ndarray:
    """
    Applies a 4x4 transformation matrix to the XYZ coordinates in a ZED2 point cloud.

    Parameters:
        point_cloud_data (np.ndarray): Input point cloud as a NumPy array of shape (H, W, 4).
                                       The first three channels represent XYZ coordinates and the
                                       fourth channel contains color information.
        transformation (np.ndarray): A 4x4 transformation matrix.

    Returns:
        np.ndarray: The transformed point cloud data with the same shape as the input.
    """
    H, W, C = point_cloud_data.shape
    # Extract XYZ coordinates and reshape to (N, 3)
    xyz = point_cloud_data[:, :, :].reshape(-1, 4)

    # Convert to homogeneous coordinates (N, 4)
    xyz[:,3] = 1

    # Apply the transformation matrix
    transformed_xyz_hom = (transformation @ xyz.T).T

    # Convert back to 3D coordinates (drop the homogeneous coordinate)
    transformed_xyz = transformed_xyz_hom[:, :3]

    # Copy the original point cloud data and update the XYZ coordinates with the transformed values.
    point_cloud_data[:, :, :3] = transformed_xyz.reshape(H, W, 3)

    return point_cloud_data

def display_point_cloud_with_interaction(point_cloud):
    """
    Displays the point cloud using an interactive Open3D visualizer (with point-picking mode)
    and forces the camera view to show the entire point cloud based on its bounding box.
    """
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Accumulated Point Cloud", width=1400, height=800)
    vis.add_geometry(point_cloud)

    # Get the view control and adjust camera parameters.
    ctr = vis.get_view_control()
    bbox = point_cloud.get_axis_aligned_bounding_box()
    center = bbox.get_center()

    # Set the camera to look at the center of the bounding box.
    ctr.set_lookat(center)

    # Set a front vector and up vector; these may need tweaking for your scene.
    ctr.set_front([0, 0, 1])
    ctr.set_up([0, 1, 0])

    # Set zoom so the entire bounding box is visible. You might need to adjust this value.
    ctr.set_zoom(0.2)

    print("Press 'P' to enter point-picking mode. Hold 'Shift' and click on points to select them.")
    vis.run()

    picked_points = vis.get_picked_points()
    if picked_points:
        points = np.asarray(point_cloud.points)
        for idx in picked_points:
            print(f"Point {idx} has coordinates ({points[idx][0]:.10f}, {points[idx][1]:.10f}, {points[idx][2]:.10f})")
    else:
        print("No points were picked.")
    vis.destroy_window()

def align_bottom_surface_to_horizontal_shift_y_to_zero(plane_model):
    """
    Computes a transformation matrix (rotation + translation) that aligns the detected plane
    with the Y-axis and translates it so that a reference point on the plane has Y = 0.

    Parameters:
        point_cloud (o3d.geometry.PointCloud): The input point cloud.
        plane_model (list or array): The plane equation parameters [a, b, c, d] for ax + by + cz + d = 0.

    Returns:
        transformation (numpy.ndarray): A 4x4 transformation matrix combining the rotation and translation.
    """
    # Extract and normalize the plane normal from the plane model.
    normal_vector = np.array(plane_model[:3])
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # Define the target normal: along the Y-axis.
    target_normal = np.array([0, 1, 0])

    # Compute the rotation axis and angle.
    rotation_axis = np.cross(normal_vector, target_normal)
    axis_norm = np.linalg.norm(rotation_axis)

    if axis_norm < 1e-6:
        # The surface is already aligned; use the identity rotation.
        rotation_matrix = np.identity(3)
        print("Surface is already aligned to the Y-axis.")
    else:
        rotation_axis = rotation_axis / axis_norm
        angle = np.arccos(np.clip(np.dot(normal_vector, target_normal), -1.0, 1.0))
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)

    # Compute a representative point on the plane.
    # We set x = 0 and z = 0, then solve for y using a*0 + b*y + c*0 + d = 0  =>  y = -d/b.
    if abs(plane_model[1]) < 1e-6:
        raise ValueError("The plane's Y coefficient is too small; cannot compute translation reliably.")
    point_on_plane = np.array([0, -plane_model[3] / plane_model[1], 0])

    # Apply the computed rotation to the point.
    rotated_point = rotation_matrix.dot(point_on_plane)

    # Compute translation: shift along Y so that the rotated point's Y value becomes 0.
    translation = np.array([0, -rotated_point[1], 0])

    # Build the full 4x4 transformation matrix.
    transformation = np.eye(4)
    transformation[:3, :3] = rotation_matrix
    transformation[:3, 3] = translation

    return transformation

def convert_point_cloud_to_open3d_and_filter_edges(point_cloud,
                                                   num_bottom_lines=0,
                                                   num_top_lines=0,
                                                   num_left_columns=0,
                                                   num_right_columns=0,
                                                   z_min=-30.0,
                                                   z_max=0.0,
                                                   x_max=0.0):
    height,  width = point_cloud.shape[:2]

    valid_rows_mask = (np.arange(height) >= num_top_lines) & (np.arange(height) < (height - num_bottom_lines))
    valid_cols_mask = (np.arange(width) >= num_left_columns) & (np.arange(width) < (width - num_right_columns))
    valid_rows_mask = valid_rows_mask[:, np.newaxis]
    valid_cols_mask = valid_cols_mask[np.newaxis, :]
    valid_mask = np.dot(valid_rows_mask, valid_cols_mask).flatten()

    points = point_cloud[:, :, :3].reshape(-1, 3)
    colors = point_cloud[:, :, 3].reshape(-1)
    points = points[valid_mask]
    colors = colors[valid_mask]

    valid_mask_nan = np.isfinite(points).all(axis=1)
    points = points[valid_mask_nan]
    colors = colors[valid_mask_nan]

    colors = np.ascontiguousarray(colors)
    colors = ((colors.view(np.uint8).reshape(-1, 4)[:, :3]) / 255).astype(np.float64)

    # Filter points based on the Z coordinate
    z_filter = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    points = points[z_filter]
    colors = colors[z_filter]

    x_filter = (points[:, 0] >= -x_max) & (points[:, 0] <= x_max)
    points = points[x_filter]
    colors = colors[x_filter]

    o3d_pc = o3d.geometry.PointCloud()
    points = points.astype(np.float64)
    o3d_pc.points = o3d.utility.Vector3dVector(points)
    o3d_pc.colors = o3d.utility.Vector3dVector(colors)
    return o3d_pc


def convert_point_cloud_to_open3d(point_cloud_data) -> o3d.geometry.PointCloud:
    """
    Converts a full ZED point cloud to an Open3D point cloud without cropping any image edges.
    This function does minimal filtering (e.g., removing non-finite points) and converts the
    color information appropriately, as well as scaling points from millimeters to meters.

    Parameters:
        point_cloud: The input ZED point cloud (assumed to have methods get_width(), get_height(),
                     and get_data() returning a NumPy array of shape (height, width, 4), where the first
                     3 channels are XYZ and the 4th channel encodes color information).

    Returns:
        An Open3D PointCloud object with points in meters and colors in the [0, 1] range.
    """
    #point_cloud_data = point_cloud.get_data()  # shape: (height, width, 4)

    # Flatten the point cloud to an (N x 3) array for points and (N,) for colors.
    points = point_cloud_data[:, :, :3].reshape(-1, 3)
    colors = point_cloud_data[:, :, 3].reshape(-1)

    # Remove non-finite points.
    valid_mask = np.isfinite(points).all(axis=1)
    points = points[valid_mask]
    colors = colors[valid_mask]

    # Convert color values:
    # The color channel is stored in a single float; view it as 4 uint8 values, and take the first 3 as RGB.
    colors = np.ascontiguousarray(colors)
    colors = ((colors.view(np.uint8).reshape(-1, 4)[:, :3]) / 255).astype(np.float64)

    # Build the Open3D point cloud.
    o3d_pc = o3d.geometry.PointCloud()

    points = points.astype(np.float64)
    o3d_pc.points = o3d.utility.Vector3dVector(points)
    o3d_pc.colors = o3d.utility.Vector3dVector(colors)

    return o3d_pc

def filter_point_cloud(point_cloud, nb_neighbors=16, std_ratio=4.0):
    """
    Applies statistical outlier removal to the given point cloud.
    """
    #print("Applying statistical outlier removal...")
    filtered_cloud, _ = point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    #print(f"Point cloud filtered, retained {len(filtered_cloud.points)} points.")
    return filtered_cloud

def voxel_downsample_point_cloud(point_cloud, voxel_size):
    """
    Downsamples the point cloud using voxel grid filtering.
    """
    #print(f"Applying voxel down-sampling with voxel size: {voxel_size}")
    downsampled_pc = point_cloud.voxel_down_sample(voxel_size)
    #print(f"Down-sampled point cloud has {len(downsampled_pc.points)} points.")
    return downsampled_pc

def detect_bottom_surface(point_cloud: o3d.geometry.PointCloud,
                          voxel_size: float = 0.5,
                          distance_threshold: float = 0.5,
                          ransac_n: int = 16,
                          num_iterations: int = 5000,
                          horizontal_threshold: float = 0.98,
                          x_range: Tuple[float, float] = (-2, 2),
                          z_range: Tuple[float, float] = (-10, -5)) -> Tuple[
    Optional[list], Optional[o3d.geometry.PointCloud], Optional[list]]:
    """
    Detects a near-horizontal plane (e.g., the road surface) in a point cloud using RANSAC,
    and crops the inlier points to a region specified by the x_range and z_range.

    Parameters:
        point_cloud (o3d.geometry.PointCloud): The input point cloud.
        voxel_size (float): Voxel size for downsampling the point cloud before processing.
        distance_threshold (float): Maximum distance a point can have to an estimated plane
                                    to be considered as an inlier.
        ransac_n (int): Number of points to sample for plane estimation.
        num_iterations (int): Number of RANSAC iterations.
        horizontal_threshold (float): Minimum absolute value of the normalized Y-component of the
                                      plane normal to consider the plane as horizontal.
        x_range (Tuple[float, float]): Tuple defining the min and max X values for cropping.
        z_range (Tuple[float, float]): Tuple defining the min and max Z values for cropping.
                                       (Assumes the car's coordinate system.)

    Returns:
        plane_model (list of float): The plane equation coefficients [a, b, c, d] for ax + by + cz + d = 0.
        cropped_plane_cloud (o3d.geometry.PointCloud): The cropped point cloud representing the detected plane,
                                                       limited to the specified x_range and z_range.
        inliers (list of int): Indices of the inlier points in the original (downsampled) point cloud.

        Returns (None, None, None) if a valid horizontal plane is not detected.
    """
    # Downsample the point cloud.
    downsampled_cloud = voxel_downsample_point_cloud(point_cloud, voxel_size)

    # Use RANSAC to segment a plane from the downsampled point cloud.
    try:
        plane_model, inliers = downsampled_cloud.segment_plane(distance_threshold=distance_threshold,
                                                               ransac_n=ransac_n,
                                                               num_iterations=num_iterations)
    except Exception as e:
        print("Plane model could not be estimated properly.")
        return None, None, None

    if len(plane_model) != 4:
        print("Plane model could not be estimated properly.")
        return None, None, None

    a, b, c, d = plane_model

    # Normalize the plane normal vector.
    normal_vector = np.array([a, b, c])
    norm = np.linalg.norm(normal_vector)
    if norm == 0:
        print("Invalid plane normal vector.")
        return None, None, None
    normal_vector /= norm

    # Check if the detected plane is near-horizontal using the normalized Y component.
    if np.abs(normal_vector[1]) < horizontal_threshold:
        print("Detected plane is not horizontal enough (normalized Y component < {:.2f}). Skipping this plane.".format(
            horizontal_threshold))
        return None, None, None

    #print(f"Plane equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
    #print(f"Detected plane has {len(inliers)} inlier points.")

    # Extract inlier points from the original point cloud.
    plane_cloud = point_cloud.select_by_index(inliers)

    # Crop the extracted plane cloud based on provided x and z ranges.
    points = np.asarray(plane_cloud.points)
    mask = np.logical_and(
        np.logical_and(points[:, 0] >= x_range[0], points[:, 0] <= x_range[1]),
        np.logical_and(points[:, 2] >= z_range[0], points[:, 2] <= z_range[1])
    )
    cropped_indices = np.where(mask)[0]
    cropped_plane_cloud = plane_cloud.select_by_index(cropped_indices)

    #print(f"Cropped plane cloud has {len(cropped_indices)} points within X{x_range} and Z{z_range}.")

    return plane_model, cropped_plane_cloud, inliers

def update_objects_with_transformation(objects, matrix):
    """
    Update the center coordinates (centerX, centerY) of each object using a 2D affine transformation matrix.

    Parameters:
        objects (list of dict): List of objects where each object has keys 'centerX' and 'centerY'.
        matrix (numpy.ndarray): A 2x3 affine transformation matrix.
            For a point (x, y), the transformation is:
                new_x = matrix[0,0] * x + matrix[0,1] * y + matrix[0,2]
                new_y = matrix[1,0] * x + matrix[1,1] * y + matrix[1,2]

    Returns:
        list of dict: The updated list of objects with new center coordinates.
    """
    for obj in objects:
        x, y = obj["centerX"], obj["centerY"]
        new_x = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
        new_y = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
        obj["centerX"] = new_x
        obj["centerY"] = new_y
    return objects

def add_or_update_object(objects, cls, c_X, c_Y, width, height, depth, timestamp):
    """
    Add a new object to the list or update an existing object if there is significant overlap.

    An object is updated if it is of the same class and the new object overlaps
    more than 50% (by area) with it. Overlap is computed relative to the area of the new object.

    Each object is assumed to be a dict with the following keys:
      - 'id'
      - 'class'
      - 'centerX'
      - 'centerY'
      - 'width'
      - 'height'

    The new_object parameter should contain the same keys except for 'id' (which will be assigned
    if the object is new).

    The function also ensures that IDs always increment, and new objects are added only if
    the new ID would be <= 1000.

    Parameters:
        objects (list): List of existing objects.
        new_object (dict): New object to add or update.

    Returns:
        list: The updated list of objects.
    """
    global new_id

    # Flag to check if update occurred.
    updated = False
    distance_min = 100

    obj_selected = None
    found = False

    # Iterate over existing objects of the same class.
    for idx, obj in enumerate(objects):
        if obj["timestamp"] == timestamp:
            continue

        distance = np.sqrt(np.square(obj["centerX"]-c_X)+np.square(obj["centerY"]-c_Y))
        if distance < distance_min and distance < 2:
            distance_min = distance
            obj_selected = idx
            found = True

    if found:
        objects[obj_selected]["class"] = cls
        objects[obj_selected]["centerX"] = c_X
        objects[obj_selected]["centerY"] = c_Y
        objects[obj_selected]["width"] += 0.4 * (width - objects[obj_selected]["width"])
        objects[obj_selected]["height"] += 0.4 * (height - objects[obj_selected]["height"])
        objects[obj_selected]["depth"] = depth
        updated = True

    # If no update occurred, add the new object.
    if not updated:
        # Determine the next id. If no objects exist, start with id = 1.
        new_id = new_id + 1
        if new_id > 1000:
            new_id = 0

        # Assign the new id.
        new_object = {"id":new_id, "class":cls, "lastX":c_X, "lastY":c_Y, "centerX":c_X, "centerY":c_Y, "width":width, "height":height, "depth":depth, "speedX":0, "speedY":0, "timestamp": timestamp}
        objects.append(new_object)

    return objects

def transform_xz(
    x: float,
    z: float,
    rot_m: np.ndarray,
    trans_m: np.ndarray
) -> Tuple[float, float]:
    """
    Rotate the point (x, 0, z) by a 3×3 rotation matrix R,
    then translate by t = (tx, ty, tz), and return (x_new, z_new),
    ignoring the y component of the final result.

    Args:
        x: original x coordinate
        z: original z coordinate
        rot_m: 3×3 rotation matrix (array-like)
        trans_m: translation vector of length 3 (array-like)

    Returns:
        (x_new, z_new): the x and z of the transformed point
    """
    # build the homogeneous vector in 3D (y=0)
    v = np.array([x, 0.0, z])
    # apply rotation, then translation
    v_transformed = rot_m.dot(v) + trans_m
    # unpack and return only x and z
    x_new, _, z_new = v_transformed
    return x_new, z_new

@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=True,  # show results
    save_txt=False,  # save results to *.txt
    save_format=0,  # save boxes coordinates in YOLO format or Pascal-VOC format (0 for YOLO and 1 for Pascal-VOC)
    save_csv=True,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
    tracking_overlay=False,
    save_speed=False,
):
    """
    Runs YOLOv5 detection inference on various sources like images, videos, directories, streams, etc.

    Args:
        weights (str | Path): Path to the model weights file or a Triton URL. Default is 'yolov5s.pt'.
        source (str | Path): Input source, which can be a file, directory, URL, glob pattern, screen capture, or webcam
            index. Default is 'data/images'.
        data (str | Path): Path to the dataset YAML file. Default is 'data/coco128.yaml'.
        imgsz (tuple[int, int]): Inference image size as a tuple (height, width). Default is (640, 640).
        conf_thres (float): Confidence threshold for detections. Default is 0.25.
        iou_thres (float): Intersection Over Union (IOU) threshold for non-max suppression. Default is 0.45.
        max_det (int): Maximum number of detections per image. Default is 1000.
        device (str): CUDA device identifier (e.g., '0' or '0,1,2,3') or 'cpu'. Default is an empty string, which uses the
            best available device.
        view_img (bool): If True, display inference results using OpenCV. Default is False.
        save_txt (bool): If True, save results in a text file. Default is False.
        save_csv (bool): If True, save results in a CSV file. Default is False.
        save_conf (bool): If True, include confidence scores in the saved results. Default is False.
        save_crop (bool): If True, save cropped prediction boxes. Default is False.
        nosave (bool): If True, do not save inference images or videos. Default is False.
        classes (list[int]): List of class indices to filter detections by. Default is None.
        agnostic_nms (bool): If True, perform class-agnostic non-max suppression. Default is False.
        augment (bool): If True, use augmented inference. Default is False.
        visualize (bool): If True, visualize feature maps. Default is False.
        update (bool): If True, update all models' weights. Default is False.
        project (str | Path): Directory to save results. Default is 'runs/detect'.
        name (str): Name of the current experiment; used to create a subdirectory within 'project'. Default is 'exp'.
        exist_ok (bool): If True, existing directories with the same name are reused instead of being incremented. Default is
            False.
        line_thickness (int): Thickness of bounding box lines in pixels. Default is 3.
        hide_labels (bool): If True, do not display labels on bounding boxes. Default is False.
        hide_conf (bool): If True, do not display confidence scores on bounding boxes. Default is False.
        half (bool): If True, use FP16 half-precision inference. Default is False.
        dnn (bool): If True, use OpenCV DNN backend for ONNX inference. Default is False.
        vid_stride (int): Stride for processing video frames, to skip frames between processing. Default is 1.
        tracking_overlay (bool): controls 2D tracking overlay
        save_speed (bool): controls saving vehicle velocity to file
    Returns:
        None

    Examples:
        ```python
        from ultralytics import run

        # Run inference on an image
        run(source='data/images/example.jpg', weights='yolov5s.pt', device='0')

        # Run inference on a video with specific confidence threshold
        run(source='data/videos/example.mp4', weights='yolov5s.pt', conf_thres=0.4, device='0')
        ```
    """
    global objects
    save_speed = 0
    bird_view = 0
    bird_view_sbs = tracking_overlay
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, pt = model.stride, model.pt

    dict = model.names

    names = dict

    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))

    last_transformation_matrix = None

    z_c = 0
    x_c = 0

    velocity = 0
    velocity_available = 0

    for path, im, im0s, vid_cap, s, depth_map, point_cloud, rotation, translation, timestamp, frame_nr in dataset:

        global last_timestamp

        # do update once per cycle
        if last_timestamp > 0 and last_timestamp != timestamp and rotation is not None and translation is not None:
             for obj in objects:
                 obj["lastX"] = obj["centerX"]
                 obj["lastY"] = obj["centerY"]
                 obj["centerY"], obj["centerX"] = transform_xz(obj["centerY"], obj["centerX"], rotation, translation)
                 obj["speedX"] = -(obj["centerX"] - obj["lastX"]) / (timestamp - last_timestamp)
                 obj["speedY"] = (obj["centerY"] - obj["lastY"]) / (timestamp - last_timestamp)
                 if obj["centerX"] < -10 or obj["centerX"] > 30 or obj["centerY"] > 10 or obj["centerY"] < -10:
                    objects.remove(obj)

        if translation is not None:
            if timestamp != last_timestamp:
                velocity = -3.6 * translation[2] / (timestamp - last_timestamp)
                velocity_available = 1
        #if last_timestamp > 0 and translation is not None:
        #    if timestamp != last_timestamp:
        #        velocity = -3.6 * translation[2] * 60
        #        velocity_available = 1
        else:
            velocity_available = 0

        last_timestamp = timestamp

        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Combine detections
        lists = [pred]
        pred = [reduce(lambda x, y: torch.cat((x, y), dim=0), tensors) for tensors in zip(*lists)]

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = str(Path(str(save_dir / os.path.basename(path)) ).with_suffix(".csv"))
        csv_speed_path = str(Path(str(save_dir / (os.path.basename(path)+"_speed"))).with_suffix(".csv"))

        # Create or append to the CSV file //p.name, frame_nr, class_name, conf, x, -z, object_depth*1000
        def write_to_csv(image_name, frame, prediction, confidence, x, x_c, y, y_c, y_closest, width, height, depth):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Frame": frame, "Prediction": prediction, "Confidence": confidence, "x": x, "x_c": x_c, "y": y, "y_c": y_c, "y_closest": y_closest, "width": width, "height": height, "depth": depth*1000}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not os.path.isfile(csv_path):
                    writer.writeheader()
                writer.writerow(data)

        def write_speed_to_csv(image_name, frame, velocity, velocity_available, timestamp):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name, "Frame": frame, "velocity": velocity, "available": velocity_available, "timestamp": timestamp}
            with open(csv_speed_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not os.path.isfile(csv_speed_path):
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "{:g}x{:g} ".format(*im.shape[2:])  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, font_size=0.5, example=str(names))

            print(f'translation: {translation}')
            print(f'rotation: {rotation}')

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                start_time = time.perf_counter()
                #o3d_pc = convert_point_cloud_to_open3d(point_cloud)
                im_height, im_width, _ = point_cloud.shape

                o3d_pc = convert_point_cloud_to_open3d_and_filter_edges(
                    point_cloud,
                    num_bottom_lines=int(im_height * 0.30),
                    num_top_lines=int(im_height * 0.4),
                    num_left_columns=int(im_width * 0.2),
                    num_right_columns=int(im_width * 0.2),
                    z_min=-14.0,
                    z_max=-4.0,
                    x_max=1.0
                )
                end_time = time.perf_counter()
                print(f"convert_point_cloud_to_open3d Execution time: {end_time - start_time:.6f} seconds")

                start_time = time.perf_counter()
                plane_model, plane_cloud, inliers_downsampled = detect_bottom_surface(o3d_pc,
                                                                                      voxel_size=0.1,
                                                                                      distance_threshold=0.1,
                                                                                      ransac_n=5,
                                                                                      num_iterations=500,
                                                                                      horizontal_threshold=0.99)
                end_time = time.perf_counter()
                print(f"detect_bottom_surface Execution time: {end_time - start_time:.6f} seconds")

                if plane_model is None or isinstance(plane_model, type(None)):
                    if last_transformation_matrix is None:
                        transformation_matrix = None
                    else:
                        transformation_matrix = last_transformation_matrix
                else:
                    transformation_matrix = align_bottom_surface_to_horizontal_shift_y_to_zero(plane_model)

                if transformation_matrix is None:
                    point_cloud_aligned = point_cloud
                else:
                    point_cloud_aligned = apply_transformation_to_zed_point_cloud(point_cloud, transformation_matrix)

                print(
                    f"apply_transformation_to_zed_point_cloud Execution time: {end_time - start_time:.6f} seconds")

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    coords = (torch.tensor(xyxy).view(1, 4)).view(-1).int().tolist()

                    # Bounding box extraction remains as is
                    x_min, y_min, x_max, y_max = coords
                    # Clip bounding box to depth map dimensions
                    height, width = depth_map.shape
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(width-1, x_max)
                    y_max = min(height-1, y_max)

                    # Extract depth map within the bounding box
                    bbox_depth_map = depth_map[y_min:y_max, x_min:x_max]
                    if transformation_matrix is not None:
                        bbox_depth_map_aligned = apply_transformation_to_depth_map(bbox_depth_map, transformation_matrix)
                        bbox_depth_map_top_z = bbox_depth_map_aligned[0, :]
                        bbox_depth_map_bottom_z = bbox_depth_map_aligned[-1, :]
                    else:
                        bbox_depth_map_top_z = bbox_depth_map[0, :]
                        bbox_depth_map_bottom_z = bbox_depth_map[-1, :]

                    bbox_depth_map_top_z = np.mean(bbox_depth_map_top_z)
                    bbox_depth_map_bottom_z = np.mean(bbox_depth_map_bottom_z)

                    bbox_point_cloud = point_cloud_aligned[y_min:y_max, x_min:x_max, :]

                    cx = int((x_max + x_min) / 2)
                    cy = int((y_max + y_min) / 2)

                    temp1 = point_cloud_aligned[max(0,cy-2):min(cy+2,height-1), max(0,x_min-5):min(x_min, width-1), 2]
                    temp1 = temp1[np.isfinite(temp1)]
                    z_c1 = np.mean(temp1)
                    temp2 = point_cloud_aligned[max(0,cy-2):min(cy+2,height-1), max(0,x_max):min(x_max+5, width-1), 2]
                    temp2 = temp2[np.isfinite(temp2)]
                    z_c2 = np.mean(temp2)
                    z_c = (z_c1+z_c2)/2

                    temp = point_cloud_aligned[
                        max(0, cy - 5):min(cy + 5, height - 1), max(0, cx - 5):min(cx + 5, width - 1), 0]
                    temp = temp[np.isfinite(temp)]
                    x_c = np.mean(temp)

                    x, y, z = extract_median_xyz(bbox_point_cloud)

                    c = int(cls)  # integer class

                    temp = point_cloud_aligned[max(0,cy-2):min(cy+2,height-1), max(0,x_min-1):min(x_min+1, width-1), 0]
                    temp = temp[np.isfinite(temp)]
                    left_edge = np.mean(temp)
                    temp = point_cloud_aligned[max(0,cy-2):min(cy+2,height-1), max(0,x_max-1):min(x_max+1, width-1), 0]
                    temp = temp[np.isfinite(temp)]
                    right_edge = np.mean(temp)
                    if np.isfinite(left_edge) and np.isfinite(right_edge):
                        width_m = abs(right_edge - left_edge)
                    else:
                        width_m = np.nan
                    temp = point_cloud_aligned[max(0,y_max-1):min(y_max+1, height-1), max(0,cx-5):min(cx+5,width-1), 2]
                    temp = temp[np.isfinite(temp)]
                    near_edge = np.mean(temp)
                    temp = point_cloud_aligned[max(0,y_max-1):min(y_max+1, height-1), max(cx-5, x_min):min(cx+5, width-1), 1]
                    temp = temp[np.isfinite(temp)]
                    near_edge_height = np.mean(temp)
                    temp = point_cloud_aligned[max(0,cy-1):min(cy+1,height-1), max(cx-5, x_min):min(cx+5, width-1), 1]
                    temp = temp[np.isfinite(temp)]
                    center_line_height = np.mean(temp)
                    temp = point_cloud_aligned[max(0,y_min-1):min(y_min+1, height-1), max(cx-5, 0):min(cx+5, width-1), 2]
                    temp = temp[np.isfinite(temp)]
                    far_edge = np.mean(temp)
                    temp = point_cloud_aligned[max(0,y_min-1):min(y_min+1, height-1), max(cx-5, x_min):min(cx+5, width-1), 1]
                    temp = temp[np.isfinite(temp)]
                    far_edge_height = np.mean(temp)
                    if np.isfinite(bbox_depth_map_top_z) and np.isfinite(bbox_depth_map_bottom_z):
                        height_z = bbox_depth_map_top_z - bbox_depth_map_bottom_z
                    elif np.isfinite(far_edge) and np.isfinite(near_edge):
                        height_z = abs(far_edge - near_edge)
                    else:
                        height_z = np.nan
                    #object_depth = (near_edge_height + far_edge_height) / 2 - center_line_height #for all objects
                    object_depth = near_edge_height - center_line_height #for ramps

                    if c == 2:
                        if -near_edge < 13 and -far_edge < 20 and -z < 14 and np.isfinite(near_edge) and np.isfinite(far_edge):
                            if abs(far_edge - near_edge) > 2:
                                class_name = "long-spdbmp"
                            else:
                                class_name = "short-spdbmp"
                        else:
                            class_name = "speedbump"
                    else:
                        class_name = names[c]

                    if np.isfinite(x) and np.isfinite(z) and -z > 4:
                        objects = add_or_update_object(objects, c, -z, x, width_m, height_z, object_depth, timestamp)

                    if save_txt:  # Write to file
                        if save_format == 0:
                            coords = (
                                (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            )  # normalized xywh
                        else:
                            coords = (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()  # xyxy
                        line = (cls, *coords, conf) if save_conf else (cls, *coords)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        label = None if hide_labels else (class_name if hide_conf else f"{class_name} {conf:.2f} {x:.2f} {-z:.2f} {bbox_depth_map_bottom_z:.2f} {(object_depth*1000):.0f}mm")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / class_name / f"{p.stem}.jpg", BGR=True)
                    if save_csv:
                        write_to_csv(p.name, frame_nr, class_name, conf.item(), x, x_c, -z, -z_c, bbox_depth_map_bottom_z, width_m, height_z, object_depth)
            if save_speed:
                write_speed_to_csv(p.name, frame_nr, velocity, velocity_available, timestamp)
            # Stream results
            im0 = annotator.result()

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            text_color = (255, 255, 0)  # yellow

            cv2.putText(im0, f'Est. speed: {velocity:.0f} km/h, available {velocity_available}', (10, 30), font, font_scale, (100, 255, 100),thickness)

            if bird_view:

                # output 2D view from the top to show results
                if im0 is None:
                    raise ValueError("Image not found or path is incorrect")
                # Example list of objects (coordinates and size are in meters)

                # Define colors for each object class (BGR format)
                class_colors = {
                    0: (0, 255, 0),  # Green
                    1: (255, 0, 0),  # Blue
                    2: (0, 0, 255)   # Red
                }

                default_color = (0, 255, 255)  # Yellow for undefined classes

                # Get image dimensions
                img_height, img_width = im0.shape[:2]

                # Define the grey rectangle dimensions: full width, and 20% of the image height
                x_span_m = 20
                bar_height = int(img_width // 40 * x_span_m)

                # Draw the grey rectangle (the "bar") at the top of the image
                grey_color = (128, 128, 128)
                overlay = im0.copy()
                cv2.rectangle(overlay, (0, 0), (img_width, bar_height), grey_color, thickness=-1)

                # Define scaling factors:
                # Horizontally: meters from -10 to +30 (span = 40 m) mapped to img_width pixels.
                x_scale = img_width / 40.0  # pixels per meter horizontally

                # Vertically: meters from -10 to +10 (span = 20 m) mapped to bar_height pixels.
                y_scale = bar_height / x_span_m # pixels per meter vertically

                #Vertical scae
                cv2.line(overlay, (int(x_scale * 5), 0), (int(x_scale * 5), bar_height), (150, 150, 150), thickness=3)
                cv2.line(overlay, (int(x_scale * 10), 0), (int(x_scale * 10), bar_height), (200, 200, 200), thickness=3)
                cv2.line(overlay, (int(x_scale * 15), 0), (int(x_scale * 15), bar_height), (150, 150, 150), thickness=3)
                cv2.line(overlay, (int(x_scale * 20), 0), (int(x_scale * 20), bar_height), (150, 150, 150), thickness=3)
                cv2.line(overlay, (int(x_scale * 25), 0), (int(x_scale * 25), bar_height), (150, 150, 150), thickness=3)
                cv2.line(overlay, (int(x_scale * 30), 0), (int(x_scale * 30), bar_height), (150, 150, 150), thickness=3)
                cv2.line(overlay, (int(x_scale * 35), 0), (int(x_scale * 35), bar_height), (150, 150, 150), thickness=3)
                # Horizontal scale
                cv2.line(overlay, (0, bar_height // 2), (img_width, bar_height // 2), (150, 150, 150), thickness=3)
                cv2.line(overlay, (0, int(bar_height // 2 + y_scale * 5)), (img_width, int(bar_height // 2 + y_scale * 5)), (150, 150, 150), thickness=3)
                cv2.line(overlay, (0, int(bar_height // 2 - y_scale * 5)), (img_width, int(bar_height // 2 - y_scale * 5)), (150, 150, 150), thickness=3)
                #Tics
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.3
                thickness = 1
                text_color = (255, 255, 255)  # white

                cv2.putText(overlay, "-5", (int(x_scale * 5)+3, bar_height - 10), font, font_scale, text_color, thickness)
                cv2.putText(overlay, "-0", (int(x_scale * 10)+3, bar_height - 10), font, font_scale, text_color, thickness)
                cv2.putText(overlay, "5", (int(x_scale * 15)+3, bar_height - 10), font, font_scale, text_color, thickness)
                cv2.putText(overlay, "10", (int(x_scale * 20)+3, bar_height - 10), font, font_scale, text_color, thickness)
                cv2.putText(overlay, "15", (int(x_scale * 25)+3, bar_height - 10), font, font_scale, text_color, thickness)
                cv2.putText(overlay, "20", (int(x_scale * 30)+3, bar_height - 10), font, font_scale, text_color, thickness)
                cv2.putText(overlay, "25", (int(x_scale * 35)+3, bar_height - 10), font, font_scale, text_color, thickness)

                cv2.putText(overlay, "-5", (3, int(bar_height // 2 - y_scale * 5)- 10), font, font_scale, text_color, thickness)
                cv2.putText(overlay, "0", (3, bar_height // 2 - 10), font, font_scale, text_color, thickness)
                cv2.putText(overlay, "5", (3, int(bar_height // 2 + y_scale * 5) - 10), font, font_scale, text_color, thickness)

                # Car
                cv2.rectangle(overlay, (int(x_scale * 6), int(bar_height // 2 - y_scale * 0.9)),
                              (int(x_scale * 11), int(bar_height // 2 + y_scale * 0.9)), (0, 0, 0), thickness=-1)
                # Car windows
                cv2.rectangle(overlay, (int(x_scale * 6.1), int(bar_height // 2 - y_scale * 0.8)),
                              (int(x_scale * 6.6), int(bar_height // 2 + y_scale * 0.8)), (100, 100, 100), thickness=-1)
                cv2.rectangle(overlay, (int(x_scale * 9.1), int(bar_height // 2 - y_scale * 0.8)),
                              (int(x_scale * 9.8), int(bar_height // 2 + y_scale * 0.8)), (140, 140, 140), thickness=-1)

                # For each object, map its center and dimensions from meters to pixel coordinates on the grey bar.
                for obj in objects:
                    id = obj["id"]
                    obj_class = obj["class"]
                    centerX_c = obj["centerX"]  # in meters
                    centerY_c = obj["centerY"]  # in meters
                    width_c = obj["width"]
                    height_c = obj["height"]

                    if centerX_c > 30 or centerX_c < -10 or centerY_c > 10 or centerY_c < -10:
                        continue

                    # Determine the color for the object
                    color = class_colors.get(obj_class, default_color)

                    # Map the X coordinate:
                    # -10 m maps to pixel 0, +20 m maps to pixel img_width.
                    center_x_px = int((centerX_c + 10) * x_scale)

                    # Map the Y coordinate:
                    # +4 m maps to the top (pixel 0) and -4 m maps to the bottom (pixel = bar_height).
                    center_y_px = int((centerY_c + x_span_m // 2) * y_scale)

                    # Convert the object's size from meters to pixels
                    if not np.isfinite(height_c):
                        if np.isfinite(width_c):
                            height_c = width_c

                    if np.isfinite(width_c) and np.isfinite(height_c):
                        width_px = int(height_c * x_scale)
                        height_px = int(width_c * y_scale)

                        # Calculate the top-left and bottom-right coordinates of the object's rectangle
                        top_left_x = center_x_px - width_px // 2
                        top_left_y = center_y_px - height_px // 2
                        bottom_right_x = center_x_px + width_px // 2
                        bottom_right_y = center_y_px + height_px // 2

                        # Clip the rectangle to ensure it fits within the grey rectangle boundaries
                        top_left_x = max(0, top_left_x)
                        top_left_y = max(0, top_left_y)
                        bottom_right_x = max(min(img_width, bottom_right_x), 0)
                        bottom_right_y = max(min(bar_height, bottom_right_y), 0)

                        # Draw the colored rectangle on the grey bar
                        cv2.rectangle(overlay, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, thickness=-1)
                    else:
                        top_left_x = center_x_px - 10
                        # Draw the colored rectangle on the grey bar
                        try:
                            cv2.rectangle(overlay, (max(center_x_px-10,0), max(center_y_px-10,0)), (min(center_x_px+10,img_width), min(center_y_px+10, bar_height)), color,
                                          thickness=-1)
                        except Exception as e:
                            print(f'Error: {e}')

                    # Prepare text settings: small font size, white color.
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 2
                    text = f'id:{id}'
                    text_color = (255, 255, 255)  # white

                    # Determine text size to adjust position if needed
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                    # Place the text above the rectangle if there's space, otherwise inside it.
                    text_x = top_left_x + 3
                    text_y = center_y_px + text_height // 2 # try to place above the box

                    cv2.putText(overlay, text, (text_x, text_y), font, font_scale, text_color, thickness)
                    # objects = [] # uncomment to not use tracking
                alpha = 0.4
                im0 = cv2.addWeighted(overlay, alpha, im0, 1 - alpha, 0)

            if bird_view_sbs:

                # output 2D view from the top to show results
                if im0 is None:
                    raise ValueError("Image not found or path is incorrect")
                # Example list of objects (coordinates and size are in meters)

                # Define colors for each object class (BGR format)
                class_colors = {
                    0: (0, 255, 0),  # Green
                    1: (255, 0, 0),  # Blue
                    2: (0, 0, 255)   # Red
                }

                default_color = (0, 255, 255)  # Yellow for undefined classes

                # Get image dimensions
                img_height, img_width = im0.shape[:2]

                bev_height = int(img_height)
                bev_width = int(bev_height / 2)

                # Define the grey rectangle dimensions: full width, and 20% of the image height
                x_span_m = 20
                bar_height = int(bev_height // 40 * x_span_m)

                # Draw the grey rectangle (the "bar") at the top of the image
                overlay = np.ones((bev_height, bev_width, 3), dtype=np.uint8) * 128

                # Define scaling factors:
                # Horizontally: meters from -10 to +30 (span = 40 m) mapped to img_width pixels.
                x_scale = bev_height / 40.0  # pixels per meter horizontally

                # Vertically: meters from -10 to +10 (span = 20 m) mapped to bar_height pixels.
                y_scale = bev_width / x_span_m # pixels per meter vertically

                #Horizontal scale
                cv2.line(overlay, (0, int(x_scale * 35)), (bar_height, int(x_scale * 35)), (150, 150, 150), thickness=3)
                cv2.line(overlay, (0, int(x_scale * 30)), (bar_height, int(x_scale * 30)), (200, 200, 200), thickness=3)
                cv2.line(overlay, (0, int(x_scale * 25)), (bar_height, int(x_scale * 25)), (150, 150, 150), thickness=3)
                cv2.line(overlay, (0, int(x_scale * 20)), (bar_height, int(x_scale * 20)), (150, 150, 150), thickness=3)
                cv2.line(overlay, (0, int(x_scale * 15)), (bar_height, int(x_scale * 15)), (150, 150, 150), thickness=3)
                cv2.line(overlay, (0, int(x_scale * 10)), (bar_height, int(x_scale * 10)), (150, 150, 150), thickness=3)
                cv2.line(overlay, (0, int(x_scale * 5)), (bar_height, int(x_scale * 5)), (150, 150, 150), thickness=3)

                # Vertical scale
                cv2.line(overlay, (bar_height // 2, 0), (bar_height // 2, img_width), (150, 150, 150), thickness=3)
                cv2.line(overlay, (int(bar_height // 2 + y_scale * 5), 0), (int(bar_height // 2 + y_scale * 5), img_width), (150, 150, 150), thickness=3)
                cv2.line(overlay, (int(bar_height // 2 - y_scale * 5), 0), (int(bar_height // 2 - y_scale * 5), img_width), (150, 150, 150), thickness=3)
                #Tics
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                text_color = (255, 255, 255)  # white

                cv2.putText(overlay, "25", (bar_height - 25, int(x_scale * 5) - 6), font, font_scale, text_color, thickness)
                cv2.putText(overlay, "20", (bar_height - 25, int(x_scale * 10) - 6), font, font_scale, text_color, thickness)
                cv2.putText(overlay, "15", (bar_height - 25, int(x_scale * 15) - 6), font, font_scale, text_color, thickness)
                cv2.putText(overlay, "10", (bar_height - 25, int(x_scale * 20) - 6), font, font_scale, text_color, thickness)
                cv2.putText(overlay, "5", (bar_height - 15, int(x_scale * 25) - 6), font, font_scale, text_color, thickness)
                cv2.putText(overlay, "0", (bar_height - 15, int(x_scale * 30) - 6), font, font_scale, text_color, thickness)
                cv2.putText(overlay, "-5", (bar_height - 30, int(x_scale * 35) - 6), font, font_scale, text_color, thickness)

                cv2.putText(overlay, "-5", (int(bar_height // 2 - y_scale * 5) - 30, bev_height - 6), font, font_scale, text_color, thickness)
                cv2.putText(overlay, "0", (bar_height // 2 - 15, bev_height - 6), font, font_scale, text_color, thickness)
                cv2.putText(overlay, "5", (int(bar_height // 2 + y_scale * 5) - 15, bev_height - 6), font, font_scale, text_color, thickness)

                # Car
                cv2.rectangle(overlay, (int(bev_width // 2 - y_scale * 0.9), bev_height - int(x_scale * 6)),
                              (int(bev_width // 2 + y_scale * 0.9), bev_height - int(x_scale * 11)), (0, 0, 0),
                              thickness=-1)
                # Car windows
                cv2.rectangle(overlay, (int(bev_width // 2 - y_scale * 0.8), bev_height - int(x_scale * 6.1)),
                              (int(bev_width // 2 + y_scale * 0.8), bev_height - int(x_scale * 6.6)), (100, 100, 100),
                              thickness=-1)
                cv2.rectangle(overlay, (int(bev_width // 2 - y_scale * 0.8), bev_height - int(x_scale * 9.1)),
                              (int(bev_width // 2 + y_scale * 0.8), bev_height - int(x_scale * 9.8)), (140, 140, 140),
                              thickness=-1)

                # For each object, map its center and dimensions from meters to pixel coordinates on the grey bar.
                for obj in objects:
                    id = obj["id"]
                    obj_class = obj["class"]
                    centerX_c = obj["centerX"]  # in meters
                    centerY_c = obj["centerY"]  # in meters
                    width_c = obj["width"]
                    height_c = obj["height"]

                    if centerX_c > 30 or centerX_c < -10 or centerY_c > 10 or centerY_c < -10:
                        continue

                    # Determine the color for the object
                    color = class_colors.get(obj_class, default_color)

                    # Map the X coordinate:
                    # -10 m maps to img_height 0, +30 m maps to 0.
                    center_x_px = bev_height - int((centerX_c + 10) * x_scale)

                    # Map the Y coordinate:
                    # -10 m maps to the left (pixel 0) and +10 m maps to the rigth (pixel = bev_width).
                    center_y_px = int((centerY_c + x_span_m // 2) * y_scale)

                    # Convert the object's size from meters to pixels
                    if not np.isfinite(height_c):
                        if np.isfinite(width_c):
                            height_c = width_c

                    if np.isfinite(width_c) and np.isfinite(height_c):
                        width_px = int(width_c * x_scale)
                        height_px = int(height_c * y_scale)

                        # Calculate the top-left and bottom-right coordinates of the object's rectangle
                        top_left_x = center_x_px - height_px // 2
                        top_left_y = center_y_px - width_px // 2
                        bottom_right_x = center_x_px + height_px // 2
                        bottom_right_y = center_y_px + width_px // 2

                        # Clip the rectangle to ensure it fits within the grey rectangle boundaries
                        top_left_x = max(0, top_left_x)
                        top_left_y = max(0, top_left_y)
                        bottom_right_x = max(min(bev_height, bottom_right_x), 0)
                        bottom_right_y = max(min(bev_width, bottom_right_y), 0)

                        # Draw the colored rectangle on the grey bar
                        cv2.rectangle(overlay, (top_left_y, top_left_x), (bottom_right_y, bottom_right_x), color, thickness=-1)
                    else:
                        top_left_x = center_x_px - 10
                        # Draw the colored rectangle on the grey bar
                        try:
                            cv2.rectangle(overlay, (max(center_y_px-10,0), max(center_x_px-10,0)), (min(center_y_px+10,bev_width), min(center_x_px+10, bev_height)), color,
                                          thickness=-1)
                        except Exception as e:
                            print(f'Error: {e}')

                    # Prepare text settings: small font size, white color.
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    text = f'id:{id}'
                    text_color = (255, 255, 255)  # white

                    # Determine text size to adjust position if needed
                    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                    # Place the text above the rectangle if there's space, otherwise inside it.
                    text_x = top_left_x + 3
                    text_y = center_y_px + text_height // 2 # try to place above the box

                    cv2.putText(overlay, text, (text_y, text_x), font, font_scale, text_color, thickness)
                    # objects = [] # uncomment to not use tracking

                im0 = np.hstack((im0, overlay))

            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                # Convert `im` from PyTorch tensor to NumPy array
                #im_numpy = im[0].permute(1, 2, 0).cpu().numpy()  # Convert CHW to HWC and move to CPU
                cv2.imshow(str(p), im0)
                key = cv2.waitKey(1)  # 1 millisecond
                if key == ord('v'):  # visualize + measure
                    # point_cloud_aligned is (H, W, 4) numpy array
                    o3d_pc_aligned = convert_point_cloud_to_open3d_and_filter_edges(
                        point_cloud_aligned,
                        num_bottom_lines=int(im_height * 0.30),
                        num_top_lines=int(im_height * 0.4),
                        num_left_columns=int(im_width* 0.2),
                        num_right_columns=int(im_width * 0.2),
                        z_min=-20.0,
                        z_max=-4.0,
                        x_max = 1.5
                    )
                    display_point_cloud_measure(o3d_pc_aligned)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                elif dataset.mode == "svo":
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get_camera_information().camera_configuration.fps
                            h, w = im0.shape[:2]
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)
        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    vid_writer[0].release()  # release previous video writer

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    """
    Parse command-line arguments for YOLOv5 detection, allowing custom inference options and model configurations.

    Args:
        --weights (str | list[str], optional): Model 0 path or Triton URL. Defaults to ROOT / 'yolov5s.pt'.
        --source (str, optional): File/dir/URL/glob/screen/0(webcam). Defaults to ROOT / 'data/images'.
        --data (str, optional): Dataset YAML path. Provides dataset configuration information.
        --imgsz (list[int], optional): Inference size (height, width). Defaults to [640].
        --conf-thres (float, optional): Confidence threshold. Defaults to 0.25.
        --iou-thres (float, optional): NMS IoU threshold. Defaults to 0.45.
        --max-det (int, optional): Maximum number of detections per image. Defaults to 1000.
        --device (str, optional): CUDA device, i.e., '0' or '0,1,2,3' or 'cpu'. Defaults to "".
        --view-img (bool, optional): Flag to display results. Defaults to False.
        --save-txt (bool, optional): Flag to save results to *.txt files. Defaults to False.
        --save-csv (bool, optional): Flag to save results in CSV format. Defaults to False.
        --save-conf (bool, optional): Flag to save confidences in labels saved via --save-txt. Defaults to False.
        --save-crop (bool, optional): Flag to save cropped prediction boxes. Defaults to False.
        --nosave (bool, optional): Flag to prevent saving images/videos. Defaults to False.
        --classes (list[int], optional): List of classes to filter results by, e.g., '--classes 0 2 3'. Defaults to None.
        --agnostic-nms (bool, optional): Flag for class-agnostic NMS. Defaults to False.
        --augment (bool, optional): Flag for augmented inference. Defaults to False.
        --visualize (bool, optional): Flag for visualizing features. Defaults to False.
        --update (bool, optional): Flag to update all models in the model directory. Defaults to False.
        --project (str, optional): Directory to save results. Defaults to ROOT / 'runs/detect'.
        --name (str, optional): Sub-directory name for saving results within --project. Defaults to 'exp'.
        --exist-ok (bool, optional): Flag to allow overwriting if the project/name already exists. Defaults to False.
        --line-thickness (int, optional): Thickness (in pixels) of bounding boxes. Defaults to 3.
        --hide-labels (bool, optional): Flag to hide labels in the output. Defaults to False.
        --hide-conf (bool, optional): Flag to hide confidences in the output. Defaults to False.
        --half (bool, optional): Flag to use FP16 half-precision inference. Defaults to False.
        --dnn (bool, optional): Flag to use OpenCV DNN for ONNX inference. Defaults to False.
        --vid-stride (int, optional): Video frame-rate stride, determining the number of frames to skip in between
            consecutive frames. Defaults to 1.

    Returns:
        argparse.Namespace: Parsed command-line arguments as an argparse.Namespace object.

    Example:
        ```python
        from ultralytics import YOLOv5
        args = YOLOv5.parse_opt()
        ```
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model 0 path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-format",
        type=int,
        default=0,
        help="whether to save boxes coordinates in YOLO format or Pascal-VOC format when save-txt is True, 0 for YOLO and 1 for Pascal-VOC",
    )
    parser.add_argument("--save-csv", action="store_true", help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true", help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true", help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    parser.add_argument("--tracking-overlay", default=False, action="store_true", help="2D tracking overlay")
    parser.add_argument("--save-speed", default=False, action="store_true", help="Controls saving of vehicle estimated speed")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """
    Executes YOLOv5 model inference based on provided command-line arguments, validating dependencies before running.

    Args:
        opt (argparse.Namespace): Command-line arguments for YOLOv5 detection. See function `parse_opt` for details.

    Returns:
        None

    Note:
        This function performs essential pre-execution checks and initiates the YOLOv5 detection process based on user-specified
        options. Refer to the usage guide and examples for more information about different sources and formats at:
        https://github.com/ultralytics/ultralytics

    Example usage:

    ```python
    if __name__ == "__main__":
        opt = parse_opt()
        main(opt)
    ```
    """
    check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
