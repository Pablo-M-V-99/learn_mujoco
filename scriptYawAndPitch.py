from scriptTraiettoria import angleStep, moveStep
from scriptImageAcquisition import imageAcquisition
from scipy.spatial.transform import Rotation as R
import numpy as np


def firstRot(m, d, viewer, first_step, first_rot, sec_step, sec_rot, or0,
             tr, tR, depth_images, seg_images, angles, poses, list, list_s):
    """
    Rotazione e riallineamento. Rotazione in Yaw
    :param m: Mujoco model
    :param d: Mujoco data
    :param viewer: viewer per consenteri la sincronizzazione del viewer
    :param first_step: Valore dell'incremento della prima rotazione
    :param first_rot: Ampiezza della prima rotazione
    :param sec_step: Valore dell'incremento della seconda rotazione
    :param sec_rot: Ampiezza della seconda rotazione
    :param or0: Orientamento all'inizio del movimento
    :param tr: durata del movimento
    :param tR: durata del riallineamento
    :param depth_images: vettore di immagini depth
    :param seg_images: vettore di immagini segmentate
    :param angles: vettore degli angoli di rotazione (da human a TCP)
    :param poses: vettore traslazione (da human a TCP)
    :param list: lista di tutte le configurazioni
    :param list_s: lista delle configurazioni da campionnare
    :return:
    """

    for q in range(first_step, first_rot + first_step, first_step):
        yaw = R.from_quat(np.array(d.mocap_quat[1]), scalar_first=True).as_euler('xyz', degrees=True)[2]
        pitch = R.from_quat(np.array(d.mocap_quat[1]), scalar_first=True).as_euler('xyz', degrees=True)[1]
        roll = R.from_quat(np.array(d.mocap_quat[1]), scalar_first=True).as_euler('xyz', degrees=True)[0]
        nextOr = [roll, pitch, q]
        if list[0] in list_s:
            list_s.remove(list[0])
            angleStep(m, d, viewer, or0, nextOr, tr)
            or0 = nextOr
            print(f"yaw: {yaw}, pitch: {pitch}, roll: {roll}")
            # Acquisizione
            depth_images, seg_images, angles, poses = imageAcquisition(m, d, depth_images, seg_images, angles, poses)
        list.pop(0)

        or0, depth_images, seg_images, angles, poses = secondRot(m, d, viewer, sec_step, sec_rot, or0, tr, tR,
                                                            depth_images, seg_images, angles, poses, list, list_s)
        or0, depth_images, seg_images, angles, poses = secondRot(m, d, viewer, -sec_step, -sec_rot, or0, tr, tR,
                                                            depth_images, seg_images, angles, poses, list, list_s)

        # Riallineamento
        # if q == first_rot and not(np.array_equal(or0, np.zeros(3))):
        #     yaw = R.from_quat(np.array(d.mocap_quat[1]), scalar_first=True).as_euler('xyz', degrees=True)[2]
        #     pitch = R.from_quat(np.array(d.mocap_quat[1]), scalar_first=True).as_euler('xyz', degrees=True)[1]
        #     roll = R.from_quat(np.array(d.mocap_quat[1]), scalar_first=True).as_euler('xyz', degrees=True)[0]
        #     nextOr = [roll, pitch, 0]
        #     angleStep(m, d, viewer, or0, nextOr, tR)
    return or0, depth_images, seg_images, angles, poses

def secondRot(m, d, viewer, sec_step, sec_rot, or0,
              tr, tR, depth_images, seg_images, angles, poses, list, list_s):
    """
    Rotazione e riallineamento. Rotazione in Pitch(Y). Quando è chiamato al di fuori di firstRot effettua le
    rotazioni con yaw = 0

    :param m: Mujoco model
    :param d: Mujoco data
    :param viewer: viewer per consenteri la sincronizzazione del viewer
    :param sec_step: Valore dell'incremento della rotazione
    :param sec_rot: Ampiezza della rotazione
    :param or0: Orientamento all'inizio del movimento
    :param tr: durata del movimento
    :param tR: durata del riallineamento
    :param depth_images: vettore di immagini depth
    :param seg_images: vettore di immagini segmentate
    :param angles: vettore degli angoli di rotazione (da human a TCP)
    :param poses: vettore traslazione (da human a TCP)
    :param list: lista di tutte le configurazioni
    :param list_s: lista delle configurazioni da campionnare
    :return: I vettori di immagini depth, di immagini segmentate, angoli e posizioni
    """

    for q in range(sec_step, sec_rot + sec_step, sec_step):
        yaw = R.from_quat(np.array(d.mocap_quat[1]), scalar_first=True).as_euler('xyz', degrees=True)[2]
        pitch = R.from_quat(np.array(d.mocap_quat[1]), scalar_first=True).as_euler('xyz', degrees=True)[1]
        roll = R.from_quat(np.array(d.mocap_quat[1]), scalar_first=True).as_euler('xyz', degrees=True)[0]
        nextOr = [roll, q, yaw]
        if list[0] in list_s:
            list_s.remove(list[0])
            angleStep(m, d, viewer, or0, nextOr, tr)
            or0 = nextOr
            print(f"yaw: {or0[2]}, pitch: {or0[1]}, roll: {or0[0]}")
            # Acquisizione
            depth_images, seg_images, angles, poses = imageAcquisition(m, d, depth_images, seg_images, angles, poses)
        list.pop(0)

        # Riallineamento
        # if q == sec_rot and not(np.array_equal(or0, np.zeros(3))):
        #     yaw = R.from_quat(np.array(d.mocap_quat[1]), scalar_first=True).as_euler('xyz', degrees=True)[2]
        #     pitch = R.from_quat(np.array(d.mocap_quat[1]), scalar_first=True).as_euler('xyz', degrees=True)[1]
        #     roll = R.from_quat(np.array(d.mocap_quat[1]), scalar_first=True).as_euler('xyz', degrees=True)[0]
        #     nextOr = [roll, 0, yaw]
        #     angleStep(m, d, viewer, or0, nextOr, tR)
    return or0, depth_images, seg_images, angles, poses

def rotV3(m, d, viewer, first_step, first_rot, sec_step, sec_rot, or0, pos0, nextPose,
             tr, T, plot, depth_images, seg_images, angles, poses, list, list_s):
    """
    Rotazione e riallineamento. Rotazione in Yaw
    :param m: Mujoco model
    :param d: Mujoco data
    :param viewer: viewer per consenteri la sincronizzazione del viewer
    :param first_step: Valore dell'incremento della prima rotazione
    :param first_rot: Ampiezza della prima rotazione
    :param sec_step: Valore dell'incremento della seconda rotazione
    :param sec_rot: Ampiezza della seconda rotazione
    :param or0: Orientamento all'inizio del movimento
    :param pos0: Posizione all'inizio della traslazione
    :param nextPose: Posizione al terminine della traslazione
    :param tr: durata della rotazione
    :param T: durata della traslazione
    :param plot: variabile per attivare la visualizzazione dei plot
    :param depth_images: vettore di immagini depth
    :param seg_images: vettore di immagini segmentate
    :param angles: vettore degli angoli di rotazione (da human a TCP)
    :param poses: vettore traslazione (da human a TCP)
    :param list: lista di tutte le configurazioni
    :param list_s: lista delle configurazioni da campionnare
    :return:
    """

    for q in range(-first_rot, first_rot + first_step, first_step):
        for qq in range(-sec_rot, sec_rot + sec_step, sec_step):
            # yaw = R.from_quat(np.array(d.mocap_quat[1]), scalar_first=True).as_euler('xyz', degrees=True)[2]
            # pitch = R.from_quat(np.array(d.mocap_quat[1]), scalar_first=True).as_euler('xyz', degrees=True)[1]
            # roll = R.from_quat(np.array(d.mocap_quat[1]), scalar_first=True).as_euler('xyz', degrees=True)[0]
            nextOr = [0, qq, q]
            if list[0] in list_s:
                if not(np.array_equal(pos0, nextPose)):
                    moveStep(m, d, viewer, pos0, nextPose, T, 'TRANSLATE')
                    pos0 = nextPose
                    print("Mi sono mosso")
                list_s.remove(list[0])
                moveStep(m, d, viewer, or0, nextOr, tr, 'ROTATE')
                or0 = nextOr
                print(f"yaw: {or0[2]}, pitch: {or0[1]}, roll: {or0[0]}")
                # Acquisizione
                depth_images, seg_images, angles, poses = imageAcquisition(m, d, depth_images, seg_images, angles,
                                                                           poses, plot)
            list.pop(0)
    return or0,pos0, depth_images, seg_images, angles, poses



if __name__ == "__main__":
    print("Questo script non deve essere eseguito autonomamente.")
