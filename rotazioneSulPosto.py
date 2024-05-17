import time
import mujoco.viewer
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from scriptGriglia import creazioneGrigliaRadiale as radialGrid
from scriptTraiettoria import move

# PARAMETRI
t0 = 0
t1 = 1  # durata movimento da un punto della griglia all'altro
T = t1 - t0
len_G = 0.1  # lunghezza griglia
wid_G = 0.1  # larghezza griglia
height_G = 0.02  # altezza griglia
dimCell = 0.01  # dimensiona cella
offX, offY, offZ = 0.2, 0.45, 0.2  # offset per l'allineamento della griglia
i, j, k = 0, 0, 0  # coordinate cella

# CREAZIONE GRIGLIA RADIALE
radialGrid()
json_path = (f"/home/pablo/PycharmProjects/learn_mujoco/grigliaRadiale.json")
with open(json_path, "r") as file:
    grigliaRad = json.load(file)

# CARICAMENTO MODELLO MUJOCO XML
xml_path = "/home/pablo/PycharmProjects/mujoco/model/plugin/elasticity/muoviBandiera.xml"

m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)

timeStep = m.opt.timestep  # mujoco simulation timestep

or0 = R.from_quat(np.array(d.mocap_quat[1]))  # np.array(d.mocap_quat[1]) è l'orientamento in quaternioni come
or0 = or0.as_euler('xyz', degrees=True)  # me lo dà mujoco -> [1 0 0 0]. pitch0.as_quat() è l'orientamento
# in quaternioni da scipy ->[1 0 0 0]. pitch0.as_euler() è
t = 0  # l'orientamento in angoli di Eulero da scipy -> [180 0 0] roll pitch yaw

yaw, pitch, roll = or0[0], or0[1], or0[2]  # orientamento (terna fissa) yaw(x) pitch(y) roll(z)

# Transitorio spawn bandiera
while t <= 20:
    t += timeStep
    mujoco.mj_step(m, d)

with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()

    while viewer.is_running() and i <= wid_G * 100 and j <= len_G * 100 and k <= height_G * 100:

        t = 0

        # ROTAZIONE IN PITCH
        # rotazione da 0 a 180
        for q in range(1, 180 + 1, 1):
            nextOr = [yaw, grigliaRad[f"rot_{q}"], roll]
            t = 0
            while t <= t1:
                pitch = move(or0[1], nextOr[1], t, t0, T)
                orientation = R.from_euler('xyz', [180 - roll, pitch, 180 + yaw], degrees=True)
                d.mocap_quat[1] = np.array(orientation.as_quat())
                mujoco.mj_step(m, d)
                viewer.sync()
                t += timeStep
            or0 = nextOr
        # ritorno a 0
        nextOr = [yaw, grigliaRad[f"rot_{0}"], roll]
        t = 0
        while t <= 10:
            pitch = move(or0[1], nextOr[1], t, t0, 10)
            orientation = R.from_euler('xyz', [180 - roll, pitch, 180 + yaw], degrees=True)
            d.mocap_quat[1] = np.array(orientation.as_quat())
            mujoco.mj_step(m, d)
            viewer.sync()
            t += timeStep
        # rotazione da 0 a -180
        for q in range(-1, -180 - 1, -1):
            or0 = nextOr
            nextOr = [yaw, grigliaRad[f"rot_{q}"], roll]
            t = 0
            while t <= t1:
                pitch = move(or0[1], nextOr[1], t, t0, T)
                orientation = R.from_euler('xyz', [180 - roll, pitch, 180 + yaw], degrees=True)
                d.mocap_quat[1] = np.array(orientation.as_quat())
                mujoco.mj_step(m, d)
                viewer.sync()
                t += timeStep
        # ritorno a 0
        or0 = nextOr
        nextOr = [yaw, grigliaRad[f"rot_{0}"], roll]
        t = 0
        while t <= 10:
            pitch = move(or0[1], nextOr[1], t, t0, 10)
            orientation = R.from_euler('xyz', [180 - roll, pitch, 180 + yaw], degrees=True)
            d.mocap_quat[1] = np.array(orientation.as_quat())
            mujoco.mj_step(m, d)
            viewer.sync()
            t += timeStep

        # ROTAZIONE IN ROLL
        # rotazione da 0 a 180
        for q in range(1, 180 + 1, 1):
            nextOr = [yaw, pitch, grigliaRad[f"rot_{q}"]]
            t = 0
            while t <= t1:
                roll = move(or0[2], nextOr[2], t, t0, T)
                orientation = R.from_euler('xyz', [180 - roll, pitch, 180 + yaw], degrees=True)
                d.mocap_quat[1] = np.array(orientation.as_quat())
                mujoco.mj_step(m, d)
                viewer.sync()
                t += timeStep
            or0 = nextOr
        # ritorno a 0
        nextOr = [yaw, pitch, grigliaRad[f"rot_{0}"]]
        t = 0
        while t <= 10:
            roll = move(or0[2], nextOr[2], t, t0, 10)
            orientation = R.from_euler('xyz', [180 - roll, pitch, 180 + yaw], degrees=True)
            d.mocap_quat[1] = np.array(orientation.as_quat())
            mujoco.mj_step(m, d)
            viewer.sync()
            t += timeStep
        # rotazione da 0 a -180
        for q in range(-1, -180 - 1, -1):
            or0 = nextOr
            nextOr = [yaw, pitch, grigliaRad[f"rot_{q}"]]
            t = 0
            while t <= t1:
                roll = move(or0[2], nextOr[2], t, t0, T)
                orientation = R.from_euler('xyz', [180 - roll, pitch, 180 + yaw], degrees=True)
                d.mocap_quat[1] = np.array(orientation.as_quat())
                mujoco.mj_step(m, d)
                viewer.sync()
                t += timeStep
        # ritorno a 0
        or0 = nextOr
        nextOr = [yaw, pitch, grigliaRad[f"rot_{0}"]]
        t = 0
        while t <= 10:
            roll = move(or0[2], nextOr[2], t, t0, 10)
            orientation = R.from_euler('xyz', [180 - roll, pitch, 180 + yaw], degrees=True)
            d.mocap_quat[1] = np.array(orientation.as_quat())
            mujoco.mj_step(m, d)
            viewer.sync()
            t += timeStep

        # ROTAZIONE IN YAW
        # rotazione da 0 a 180
        for q in range(1, 180 + 1, 1):
            nextOr = [grigliaRad[f"rot_{q}"] + 180, pitch, roll]
            t = 0
            while t <= t1:
                yaw = move(or0[0], nextOr[0], t, t0, T)
                orientation = R.from_euler('xyz', [180 - roll, pitch, 180 + yaw], degrees=True)
                d.mocap_quat[1] = np.array(orientation.as_quat())
                mujoco.mj_step(m, d)
                viewer.sync()
                t += timeStep
            or0 = nextOr
        # ritorno a 0
        nextOr = [grigliaRad[f"rot_{0}"] + 180, pitch, yaw]
        t = 0
        while t <= 10:
            yaw = move(or0[0], nextOr[0], t, t0, 10)
            orientation = R.from_euler('xyz', [180 - roll, pitch, 180 + yaw], degrees=True)
            d.mocap_quat[1] = np.array(orientation.as_quat())
            mujoco.mj_step(m, d)
            viewer.sync()
            t += timeStep
        # rotazione da 0 a -180
        for q in range(-1, -180 - 1, -1):
            or0 = nextOr
            nextOr = [grigliaRad[f"rot_{q}"] + 180, pitch, yaw]
            t = 0
            while t <= t1:
                yaw = move(or0[0], nextOr[0], t, t0, T)
                orientation = R.from_euler('xyz', [180 - roll, pitch, 180 + yaw], degrees=True)
                d.mocap_quat[1] = np.array(orientation.as_quat())
                mujoco.mj_step(m, d)
                viewer.sync()
                t += timeStep
        # ritorno a 0
        or0 = nextOr
        nextOr = [grigliaRad[f"rot_{0}"] + 180, pitch, yaw]
        t = 0
        while t <= 10:
            yaw = move(or0[0], nextOr[0], t, t0, 10)
            orientation = R.from_euler('xyz', [180 - roll, pitch, 180 + yaw], degrees=True)
            d.mocap_quat[1] = np.array(orientation.as_quat())
            mujoco.mj_step(m, d)
            viewer.sync()
            t += timeStep

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
