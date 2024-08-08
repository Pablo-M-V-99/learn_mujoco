from scriptTraiettoria import angleStep
from scriptImageAcquisition import imageAcquisition
from scipy.spatial.transform import Rotation as R
import numpy as np
import json

def firstRot(m, d, viewer, first_step, first_rot, sec_step, sec_rot, or0,
             tr, tR, depth_images, seg_images, angles, poses):
    """
    Prima rotazione e riallineamento. Rotazione in Yaw
    :param m: Mujoco model
    :param d: Mujoco data
    :param viewer: viewer per consenteri la sincronizzazione del viewer
    :param first_step: Valore dell'incremento della prima rotazione
    :param first_rot: Ampiezza della prima rotazione
    :param sec_step: Valore dell'incremento della seconda rotazione
    :param sec_rot: Ampiezza della prima rotazione
    :param or0: Orientamento all'inizio del movimento
    :param tr: durata del movimento
    :param tR: durata del riallineamento
    :param depth_images: vettore di immagini depth
    :param seg_images: vettore di immagini segmentate
    :param angles: vettore degli angoli di rotazione (da human a TCP)
    :param poses: vettore traslazione (da human a TCP)
    :return:
    """
    json_path = (f"/home/pablo/PycharmProjects/learn_mujoco/grigliaRadiale.json")
    with open(json_path, "r") as file:
        grigliaRad = json.load(file)

    for q in range(first_step, first_rot + first_step, first_step):

        yaw = R.from_quat(np.array(d.mocap_quat[1]), scalar_first=True).as_euler('xyz', degrees=True)[2]
        pitch = R.from_quat(np.array(d.mocap_quat[1]), scalar_first=True).as_euler('xyz', degrees=True)[1]
        roll = R.from_quat(np.array(d.mocap_quat[1]), scalar_first=True).as_euler('xyz', degrees=True)[0]

        nextOr = [roll, pitch, grigliaRad[f"rot_{q}"]]

        roll, pitch, yaw, or0 = angleStep(m, d, viewer, or0, nextOr, tr)

        depth_images, seg_images, angles, poses = imageAcquisition(m, d, depth_images, seg_images, angles, poses)

        depth_images, seg_images, angles, poses = secondRot(m, d, viewer, sec_step, sec_rot, or0, tr, tR,
                                                            depth_images, seg_images, angles, poses)
        depth_images, seg_images, angles, poses = secondRot(m, d, viewer, -sec_step, -sec_rot, or0, tr, tR,
                                                            depth_images, seg_images, angles, poses)

        # riallineamento
        if q == first_rot:
            nextOr = [roll, pitch, grigliaRad[f"rot_{0}"]]
            angleStep(m, d, viewer, or0, nextOr, tR)
    return depth_images, seg_images, angles, poses

def secondRot(m, d, viewer, sec_step, sec_rot, or0,
              tr, tR, depth_images, seg_images, angles, poses):
    """
    Seconda rotazione e riallineamento. Rotazione in Pitch

    :param m: Mujoco model
    :param d: Mujoco data
    :param viewer: viewer per consenteri la sincronizzazione del viewer
    :param sec_step: Valore dell'incremento della seconda rotazione
    :param sec_rot: Ampiezza della prima rotazione
    :param or0: Orientamento all'inizio del movimento
    :param tr: durata del movimento
    :param tR: durata del riallineamento
    :param depth_images: vettore di immagini depth
    :param seg_images: vettore di immagini segmentate
    :param angles: vettore degli angoli di rotazione (da human a TCP)
    :param poses: vettore traslazione (da human a TCP)
    :return: I vettori di immagini depth, di immagini segmentate, angoli e posizioni
    """

    json_path = (f"/home/pablo/PycharmProjects/learn_mujoco/grigliaRadiale.json")
    with open(json_path, "r") as file:
        grigliaRad = json.load(file)

    for q in range(sec_step, sec_rot + sec_step, sec_step):

        yaw = R.from_quat(np.array(d.mocap_quat[1]), scalar_first=True).as_euler('xyz', degrees=True)[2]
        pitch = R.from_quat(np.array(d.mocap_quat[1]), scalar_first=True).as_euler('xyz', degrees=True)[1]
        roll = R.from_quat(np.array(d.mocap_quat[1]), scalar_first=True).as_euler('xyz', degrees=True)[0]
        nextOr = [roll, grigliaRad[f"rot_{q}"], yaw]

        roll, pitch, yaw, or0 = angleStep(m, d, viewer, or0, nextOr, tr)

        depth_images, seg_images, angles, poses = imageAcquisition(m, d, depth_images, seg_images, angles, poses)

        # riallineamento
        if q == sec_rot:
            nextOr = [roll, grigliaRad[f"rot_{0}"], yaw]
            angleStep(m, d, viewer, or0, nextOr, tR)
    return depth_images, seg_images, angles, poses

if __name__ == "__main__":
    print("Questo script non deve essere eseguito autonomamente.")
