import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


def imageAcquisition(m, d, depth_images, angles, poses, plot):
    """
    Salva in vettori distinti le immagini depth, le immagini segmentate, il vettore di traslazione da terna human a
    terna TCP e gli angoli di rotazione della matrice di trasformazione omogenea riferita alla terna human della terna
    TCP. Alla fine del codice è possibile attivare la visualizzazione del plot

    :param m: Mujoco model.
    :param d: Mujoco data.
    :param depth_images: Vettore di immagini depth
    :param seg_images: vettore di immagini segmentate
    :param angles: vettore degli angoli di rotazione (da human a TCP)
    :param poses: vettore traslazione (da human a TCP)
    :param plot: variabile bool per attivare il plot (se TRUE).
    :return: I vettori di immagini depth, di immagini segmentate, angoli e posizioni.
    """

    # Image Resolution
    width = m.cam_resolution[0][0]
    height = m.cam_resolution[0][1]
    # Fields of view (if width == height should be fov_x == fov_y)
    fovx = np.deg2rad(120)
    fovy = np.deg2rad(m.cam_fovy[0])
    # Focal Lenghts (if width == height should be f_x == f_y)
    fx = width / (2 * np.tan(fovx / 2))
    fy = height / (2 * np.tan(fovy / 2))
    # Camera center coordinates
    cx = width / 2
    cy = height / 2

    # DEPTH
    with mujoco.Renderer(m, height, width) as renderer:
        renderer.enable_depth_rendering()
        renderer.update_scene(d, camera='azure_kinect')
        depth_plane = renderer.render()                 # depth is a float array, in meters
        depth_plane[depth_plane > 5] = 0
        renderer.disable_depth_rendering()
        i_indices, j_indices = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')
        x_camera = (i_indices - cx) * depth_plane / fx
        y_camera = (j_indices - cy) * depth_plane / fy
        depth_frame = np.sqrt(depth_plane ** 2 + x_camera ** 2 + y_camera ** 2)
        depth_images.append(depth_frame)

    # SEGMENTATION
    with mujoco.Renderer(m, height, width) as renderer:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(d, camera='azure_kinect')
        seg_frame = renderer.render()
        renderer.disable_segmentation_rendering()
        seg_frame = seg_frame[:, :, 0] + 1

    # Applicazione della maschera segmentata all'immagine depth
    depth_frame = depth_frame * seg_frame

    # manipolazione angoli per Giorgio
    angoli_TCP = R.from_quat(np.array(d.mocap_quat[1]), scalar_first=True).as_euler('xyz', degrees=True)
    rot_x = angoli_TCP[0]
    rot_y = angoli_TCP[1]
    rot_z = angoli_TCP[2]

    # matrice rotazione (mio sistema di riferimento)
    R_ws_TCP = R.from_euler('xyz', [rot_x, rot_y, rot_z], degrees=True).as_matrix()

    R_y = R.from_euler('xyz', [0, -180, 0], degrees=True).as_matrix()  # rotazione di 180° attorno all'asse Y
    R_TCP =  R_ws_TCP @ R_y        # matrice di rotazione espressa rispetto alla terna world ruotata di 180° attorno Y

    # vettore di traslazione TCP
    O_ws_TCP = d.mocap_pos[1]

    # matrice di trasformazione omogenea
    A_ws_TCP = np.vstack((np.hstack((R_TCP, np.reshape(O_ws_TCP, (3, -1)))), [0, 0, 0, 1]))

    # orientamento terna Human
    angoli_H = R.from_quat(np.array(d.mocap_quat[4]), scalar_first=True).as_euler('xyz', degrees=True)
    rotH_x = angoli_H[0]
    rotH_y = angoli_H[1]
    rotH_z = angoli_H[2]
    # matrice rotazione
    R_H = R.from_euler('xyz', [rotH_x, rotH_y, rotH_z], degrees=True).as_matrix()

    R_z = R.from_euler('xyz', [0, 0, -180], degrees=True).as_matrix()     # rotazione di -180° attorno all'asse Z
    R_H =  R_H @ R_y @ R_z                  # matrice di rotazione espresso rispetto alla terna world ruotata di 180 attorno Y e 180 attorno Z

    # vettore traslazione Human
    O_ws_H = d.mocap_pos[4]

    # matrice di trasformazione omogenea
    A_ws_H = np.vstack((np.hstack((R_H, np.reshape(O_ws_H, (3, -1)))), [0, 0, 0, 1]))
    A_H_ws = np.linalg.inv(A_ws_H)

    # matrice omogenea da terna uomo a TCP
    A_H_TCP = A_H_ws @ A_ws_TCP

    angles_H_TCP = R.from_matrix(A_H_TCP[:3,:3]).as_euler('xyz', degrees=True)
    angles.append(np.array(angles_H_TCP))
    poses.append(np.array(A_H_TCP[:3,3]))

    # PLOT
    if plot:
        _, ax = plt.subplots(ncols=2)
        ax[0].imshow(depth_frame)
        ax[1].imshow(seg_frame)
        plt.show()

    return depth_images, angles, poses


if __name__ == "__main__":
    print("Questo script non deve essere eseguito autonomamente.")
