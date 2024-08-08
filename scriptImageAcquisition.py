import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
from math import cos, sin, pi
import matplotlib.pyplot as plt
import matplotlib as mpl


def imageAcquisition(m, d, depth_images, seg_images, angles, poses):
    """
    Salva in vettori distinti le immagini depth, le immagini segmentate, il vettore di traslazione da terna human a
    terna TCP e gli angoli di rotazione della matrice di trasformazione omogenea riferita alla terna human della terna
    TCP.

    :param m: Mujoco model
    :param d: Mujoco data
    :param depth_images: vettore di immagini depth
    :param seg_images: vettore di immagini segmentate
    :param angles: vettore degli angoli di rotazione (da human a TCP)
    :param poses: vettore traslazione (da human a TCP)
    :return: I vettori di immagini depth, di immagini segmentate, angoli e posizioni
    """

    # Image Resolution
    width = m.cam_resolution[0][0]
    height = m.cam_resolution[0][1]
    # Fields of view (if width == height should be fovx == fovy)
    fovx = np.deg2rad(120)
    fovy = np.deg2rad(m.cam_fovy[0])
    # Focal Lenghts (if width == height should be fovx == fovy)
    fx = width / (2 * np.tan(fovx / 2))
    fy = height / (2 * np.tan(fovy / 2))
    # Camera center coordinates
    cx = width / 2
    cy = height / 2

    # DEPTH
    with mujoco.Renderer(m, height, width) as renderer:
        # mujoco.mj_step(m, d)
        renderer.enable_depth_rendering()
        renderer.update_scene(d, camera='azure_kinect')
        # depth is a float array, in meters.
        depth_plane = renderer.render()
        depth_plane[depth_plane > 5] = np.nan
        renderer.disable_depth_rendering()
        i_indices, j_indices = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')
        x_camera = (i_indices - cx) * depth_plane / fx
        y_camera = (j_indices - cy) * depth_plane / fy
        depth = np.sqrt(depth_plane ** 2 + x_camera ** 2 + y_camera ** 2)
        depth_images.append(depth)

        # plt.imshow(depth)
        # plt.show()

    # SEGMENTATION
    with mujoco.Renderer(m, height, width) as renderer:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(d, camera='azure_kinect')
        seg_frame = renderer.render()
        # # Display the contents of the first channel, which contains object
        # # IDs. The second channel, seg[:, :, 1], contains object types
        # geom_ids = seg_frame[:, :, 0]
        # # Infinity is mapped to -1. Now we remap it to 0
        # geom_ids = geom_ids.astype(np.float64) + 1
        # # Scale to [0, 1]
        # geom_ids = geom_ids / geom_ids.max()
        # seg_frame = 255 * geom_ids
        renderer.disable_segmentation_rendering()
        seg_images.append(seg_frame)

        # plt.imshow(seg_frame[:, :, 0], cmap='Dark2')
        # plt.show()

    # manipolazione angoli per Giorgio
    angoli_TCP = R.from_quat(np.array(d.mocap_quat[1]), scalar_first=True)
    angoli_TCP = angoli_TCP.as_euler('xyz', degrees=True)
    rot_x = -angoli_TCP[0]          # rotazione asse x di giorgio
    rot_y = angoli_TCP[1]           # rotazione asse y di giorgio
    rot_z = -angoli_TCP[2]          # rotazione asse z di giorgio
    # # vettore di traslazione TCP
    O_ws_TCP = d.mocap_pos[1]
    # # matrice rotazione
    R_TCP = R.from_rotvec([rot_x, rot_y, rot_z], degrees=True)
    R_TCP = R_TCP.as_matrix()
    # matrice omogenea
    A_ws_TCP = np.vstack((np.hstack((R_TCP, np.reshape(O_ws_TCP, (3, -1)))), [0, 0, 0, 1]))


    # orientamento terna Human
    angoli_H = R.from_quat(np.array(d.mocap_quat[4]), scalar_first=True)
    angoli_H = angoli_H.as_euler('xyz', degrees=True)
    rotH_x = -angoli_H[0]          # rotazione asse x di giorgio
    rotH_y = angoli_H[1]           # rotazione asse y di giorgio
    rotH_z = -angoli_H[2]          # rotazione asse z di giorgio
    # vettore traslazione Human
    O_ws_H = d.mocap_pos[4]
    # matrice rotazione
    R_H = R.from_rotvec([rotH_x, rotH_y, rotH_z], degrees=True)
    R_H = R_H.as_matrix()
    # matrice omogenea
    A_ws_H = np.vstack((np.hstack((R_H, np.reshape(O_ws_H, (3, -1)))), [0, 0, 0, 1]))

    A_H_ws = np.linalg.inv(A_ws_H)

    # matrice omogenea da terna uomo a TCP
    A_H_TCP = A_H_ws * A_ws_TCP

    angles_H_TCP = R.from_matrix(A_H_TCP[:3,:3])
    angles_H_TCP = angles_H_TCP.as_euler('xyz', degrees=True)
    angles.append(np.array(angles_H_TCP))
    poses.append(np.array(A_H_TCP[:3,3]))

    return depth_images, seg_images, angles, poses


if __name__ == "__main__":
    print("Questo script non deve essere eseguito autonomamente.")
