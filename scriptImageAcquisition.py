import mujoco.viewer
import numpy as np
from math import cos, sin, pi
import matplotlib.pyplot as plt
import matplotlib as mpl


def imageAcquisition(m, d, yaw, pitch, roll, depth_images, seg_images, angles, poses):

    # Resolution
    height = 512
    width = 512

    # DEPTH
    with mujoco.Renderer(m, height, width) as renderer:
        renderer.enable_depth_rendering()
        renderer.update_scene(d, camera='azure_kinect')

        # depth is a float array, in meters.
        depth_frame = renderer.render()
        # Shift nearest values to the origin.
        depth_frame -= depth_frame.min()
        # Scale by 2 mean distances of near rays.
        depth_frame /= 2 * depth_frame[depth_frame <= 1].mean()
        # Scale to [0, 255]
        depth_frame = 255 * np.clip(depth_frame, 0, 1)

        depth_images.append(depth_frame)
        renderer.disable_depth_rendering()

        # plt.imshow(depth_frame, cmap='gray')
        # plt.show()

    # SEGMENTATION
    with mujoco.Renderer(m, height, width) as renderer:
        renderer.enable_segmentation_rendering()
        renderer.update_scene(d, camera='azure_kinect')

        seg_frame = renderer.render()
        # Display the contents of the first channel, which contains object
        # IDs. The second channel, seg[:, :, 1], contains object types
        geom_ids = seg_frame[:, :, 0]
        # Infinity is mapped to -1
        geom_ids = geom_ids.astype(np.float64) + 1
        # Scale to [0, 1]
        geom_ids = geom_ids / geom_ids.max()
        seg_frame = 255 * geom_ids

        seg_images.append(seg_frame)
        renderer.disable_segmentation_rendering()

        # plt.imshow(seg_frame, cmap='Dark2')
        # plt.show()

    rot_x = (180 - roll)    # rotazione asse x di giorgio
    rot_y = pitch           # rotazione asse y di giorgio
    rot_z = -yaw            # rotazione asse z di giorgio

    # vettore di traslazione (verticale)
    O_ws_TCP = d.mocap_pos[1]

    # matrice di rotazione
    # R_ws_TCP = np.array([[cos(phi) * cos(theta), cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi),
    #                       cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi)],
    #                      [sin(phi) * cos(theta), sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi),
    #                       sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi)],
    #                      [-sin(theta), cos(theta) * sin(psi), cos(theta) * cos(psi)]])

    angles.append(np.array([rot_x, rot_y, rot_z]))
    poses.append(np.array(O_ws_TCP))

    return depth_images, seg_images, angles, poses


if __name__ == "__main__":
    print("Questo script non deve essere eseguito autonomamente.")
