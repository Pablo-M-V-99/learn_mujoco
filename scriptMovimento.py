from scriptControlloVelocità import speedCtrl
from scriptImageAcquisition import imageAcquisition
from scipy.spatial.transform import Rotation as R
import numpy as np
import mujoco.viewer
import datetime



def compTrajectory(pose0, nextPose, t, T):
    """Calcola il polinomio di grado 5 impiegato per la traiettoria

    :param pos0: posa all'istante iniziale
    :param nextPose: posa da raggiungere (all'istante finale)
    :param t: istante di tempo t
    :param T: durata del movimento
    :return: variabile al termine del movimento
    """

    a0 = pose0
    a1 = 0
    a2 = 0
    a3 = 20 * (nextPose - pose0) / (2 * (T ** 3))
    a4 = 30 * (pose0 - nextPose) / (2 * (T ** 4))
    a5 = 12 * (nextPose - pose0) / (2 * (T ** 5))
    q = a0 + a1*t + a2*(t ** 2) + a3*(t ** 3) + a4*(t ** 4) + a5*(t ** 5)
    return q


def moveToNext(m, d, viewer, pose0, nextPose, T, flag)-> bool:
    """
    Esegue il movimento per spostarsi alla configurazione successiva

    :param m: Mujoco model
    :param d: Mujoco data
    :param viewer: consente la sincronizzazione del viewer
    :param pose0: posa all'inizio del movimento
    :param nextPose: posa al termine del movimento
    :param T: durata del movimento
    :param flag: stabilisce se il corpo ruota o trasla
    :return: se True l'immagine verrà acquisita
    """

    t = 0
    timeStep = m.opt.timestep  # mujoco simulation timestep
    while t <= T:
        move1 = compTrajectory(pose0[0], nextPose[0], t, T)   # roll se ROTATE, x se TRANSLATE
        move2 = compTrajectory(pose0[1], nextPose[1], t, T)   # pitch se ROTATE, y se TRANSLATE
        move3 = compTrajectory(pose0[2], nextPose[2], t, T)   # yaw se ROTATE, z se TRANSLATE
        if t >= T - 2 * timeStep:   # memorizzazione ultima posa per il controllo velocità
            prevPose = np.array([d.body(f"flag_{ii}").xpos for ii in range(d.flexvert_xpos.shape[0])])
        if flag == 'ROTATE':
            orientation = R.from_euler('xyz', [move1, move2, move3], degrees=True)
            d.mocap_quat[1] = np.array(orientation.as_quat(scalar_first=True))
        if flag == 'TRANSLATE':
            d.mocap_pos[1] = np.array([move1, move2, move3])
        mujoco.mj_step(m, d)
        if viewer:
            viewer.sync()
        t += timeStep
    return speedCtrl(m, d, viewer, prevPose)


def move(m, d, viewer, first_step, first_rot, sec_step, sec_rot, or0, pos0, nextPose, tr, T, plot, depth_images, angles,
         poses, list, list_s):
    """
    Effettua traslazione e rotazione

    :param m: Mujoco model.
    :param d: Mujoco data.
    :param viewer: Consente la sincronizzazione del viewer.
    :param first_step: Valore dell'incremento della prima rotazione.
    :param first_rot: Ampiezza della prima rotazione.
    :param sec_step: Valore dell'incremento della seconda rotazione.
    :param sec_rot: Ampiezza della seconda rotazione.
    :param or0: Orientamento all'inizio del movimento.
    :param pos0: Posizione all'inizio della traslazione.
    :param nextPose: Posizione al termine della traslazione.
    :param tr: Durata della rotazione.
    :param T: Durata della traslazione.
    :param plot: Consente la visualizzazione dei plot.
    :param depth_images: Vettore di immagini depth
    :param angles: vettore degli angoli di rotazione (da human a TCP)
    :param poses: vettore traslazione (da human a TCP)
    :param list: lista di tutte le configurazioni
    :param list_s: lista delle configurazioni da campionare
    """

    good_pose = True
    for qq in range(-first_rot, first_rot + first_step, first_step):
        for q in range(-sec_rot, sec_rot + sec_step, sec_step):
            # yaw = R.from_quat(np.array(d.mocap_quat[1]), scalar_first=True).as_euler('xyz', degrees=True)[2]
            # pitch = R.from_quat(np.array(d.mocap_quat[1]), scalar_first=True).as_euler('xyz', degrees=True)[1]
            # roll = R.from_quat(np.array(d.mocap_quat[1]), scalar_first=True).as_euler('xyz', degrees=True)[0]
            nextOr = [0, q, qq]
            if list[0] in list_s:
                list_s.remove(list[0])

                if len(list_s) == 1000 or len(list_s) == 2000 or len(list_s) == 0:
                    # Genera il timestamp corrente
                    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    print(f"{current_time}: 1000 pose acquisite")

                if not (np.array_equal(pos0, nextPose)):
                    good_pose = moveToNext(m, d, viewer, pos0, nextPose, T, 'TRANSLATE')
                    pos0 = nextPose
                    # print("Ho traslato")

                if not (np.array_equal(or0, nextOr)):
                    good_pose = moveToNext(m, d, viewer, or0, nextOr, tr, 'ROTATE')
                    or0 = nextOr
                    # print(f"yaw: {or0[2]}, pitch: {or0[1]}, roll: {or0[0]}")

                # Acquisizione (solo se il check di velocità è negativo)
                if good_pose:
                    depth_images, angles, poses = imageAcquisition(m, d, depth_images, angles, poses, plot)
            list.pop(0)
    return or0, pos0, depth_images, angles, poses
