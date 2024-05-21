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
tr = 1          # durata riallineamento
T = t1 - t0
len_G = 0.1  # lunghezza griglia
wid_G = 0.1  # larghezza griglia
height_G = 0.02  # altezza griglia
dimCell = 0.01  # dimensiona cella
offX, offY, offZ = 0.2, 0.45, 0.2  # offset per l'allineamento della griglia
i, j, k = 0, 0, 0  # coordinate cella
pitch_rot, yaw_rot, roll_rot = 20, 20, 10
pitch_step, yaw_step, roll_step = 5, 5, 2


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

# np.array(d.mocap_quat[1]) è l'orientamento in quaternioni come me lo dà mujoco -> [1 0 0 0]. pitch0.as_quat() è
# l'orientamento in quaternioni da scipy ->[1 0 0 0]. pitch0.as_euler() è l'orientamento in angoli di Eulero da
# scipy -> [180 0 0] (nell'ordine yaw pitch roll)
or0 = R.from_quat(np.array(d.mocap_quat[1]))
or0 = or0.as_euler('xyz', degrees=True)
t = 0

roll, pitch, yaw = or0[0], or0[1], or0[2]  # orientamento (terna fissa) yaw(z) pitch(y) roll(x)
or0_init = or0
# Transitorio spawn bandiera
while t <= 20:
    t += timeStep
    mujoco.mj_step(m, d)

with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()

    while viewer.is_running() and i <= wid_G * 100 and j <= len_G * 100 and k <= height_G * 100:

        t = 0

        # ROTAZIONE IN YAW
        for p in range(yaw_step, yaw_rot + 1, yaw_step):
            nextOr = [roll, pitch, grigliaRad[f"rot_{p}"]]
            t = 0
            while t <= tr:                      # rotazione positiva
                yaw = move(or0[2], nextOr[2], t, t0, tr - t0)
                orientation = R.from_euler('xyz', [180 - yaw, -pitch, 180 - roll], degrees=True)
                d.mocap_quat[1] = np.array(orientation.as_quat())
                mujoco.mj_step(m, d)
                viewer.sync()
                t += timeStep
            or0 = nextOr
            if p == yaw_rot:
                nextOr = [roll, pitch, grigliaRad[f"rot_{0}"]]
                t = 0
                while t <= 10:                   # ritorno a 0
                    yaw = move(or0[2], nextOr[2], t, t0, 10)
                    orientation = R.from_euler('xyz', [180 - yaw, -pitch, 180 - roll], degrees=True)
                    d.mocap_quat[1] = np.array(orientation.as_quat())
                    mujoco.mj_step(m, d)
                    viewer.sync()
                    t += timeStep
        for p in range(-yaw_step, -yaw_rot - 1, -yaw_step):
            or0 = nextOr
            nextOr = [roll, pitch, grigliaRad[f"rot_{p}"]]
            t = 0
            while t <= tr:                      # rotazione negativa
                yaw = move(or0[2], nextOr[2], t, t0, tr - t0)
                orientation = R.from_euler('xyz', [180 - yaw, -pitch, 180 - roll], degrees=True)
                d.mocap_quat[1] = np.array(orientation.as_quat())
                mujoco.mj_step(m, d)
                viewer.sync()
                t += timeStep
            if p == -yaw_rot:
                or0 = nextOr
                nextOr = [roll, pitch, grigliaRad[f"rot_{0}"]]
                t = 0
                while t <= 10:                          # ritorno a 0
                    yaw = move(or0[2], nextOr[2], t, t0, 10)
                    orientation = R.from_euler('xyz', [180 - yaw, -pitch, 180 - roll], degrees=True)
                    d.mocap_quat[1] = np.array(orientation.as_quat())
                    mujoco.mj_step(m, d)
                    viewer.sync()
                    t += timeStep

        # ROTAZIONE IN PITCH
        for q in range(pitch_step, pitch_rot + 1, pitch_step):
            nextOr = [roll, grigliaRad[f"rot_{q}"], yaw]
            t = 0
            while t <= tr:          # rotazione positiva
                pitch = move(or0[1], nextOr[1], t, t0, tr - t0)
                orientation = R.from_euler('xyz', [180 - yaw, -pitch, 180 - roll], degrees=True)
                d.mocap_quat[1] = np.array(orientation.as_quat())
                mujoco.mj_step(m, d)
                viewer.sync()
                t += timeStep
            or0 = nextOr
            if q == pitch_rot:
                nextOr = [roll, grigliaRad[f"rot_{0}"], yaw]
                t = 0
                while t <= 10:                  # ritorno a 0
                    pitch = move(or0[1], nextOr[1], t, t0, 10)
                    orientation = R.from_euler('xyz', [180 - yaw, -pitch, 180 - roll], degrees=True)
                    d.mocap_quat[1] = np.array(orientation.as_quat())
                    mujoco.mj_step(m, d)
                    viewer.sync()
                    t += timeStep
        for q in range(-pitch_step, -pitch_rot - 1, -pitch_step):
            or0 = nextOr
            nextOr = [roll, grigliaRad[f"rot_{q}"], yaw]
            t = 0
            while t <= tr:                      # rotazione negativa
                pitch = move(or0[1], nextOr[1], t, t0, tr - t0)
                orientation = R.from_euler('xyz', [180 - yaw, -pitch, 180 + roll], degrees=True)
                d.mocap_quat[1] = np.array(orientation.as_quat())
                mujoco.mj_step(m, d)
                viewer.sync()
                t += timeStep
            if q == -pitch_rot:
                or0 = nextOr
                nextOr = [roll, grigliaRad[f"rot_{0}"], yaw]
                t = 0
                while t <= 10:                   # ritorno a 0
                    pitch = move(or0[1], nextOr[1], t, t0, 10)
                    orientation = R.from_euler('xyz', [180 - yaw, -pitch, 180 + roll], degrees=True)
                    d.mocap_quat[1] = np.array(orientation.as_quat())
                    mujoco.mj_step(m, d)
                    viewer.sync()
                    t += timeStep

        # ROTAZIONE IN ROLL
        for l in range(roll_step, roll_rot + 1, roll_step):
            nextOr = [grigliaRad[f"rot_{l}"], pitch, yaw]
            t = 0
            while t <= tr:                      # rotazione positiva
                roll = move(or0[0], nextOr[0] + 180, t, t0, tr - t0)
                orientation = R.from_euler('xyz', [180 - yaw, -pitch, roll - 180], degrees=True)
                d.mocap_quat[1] = np.array(orientation.as_quat())
                mujoco.mj_step(m, d)
                viewer.sync()
                t += timeStep
            or0 = nextOr
            or0[0] += 180
            if l == roll_rot:
                nextOr = [grigliaRad[f"rot_{0}"], -pitch, yaw]
                t = 0
                while t <= 10:                   # ritorno a 0
                    roll = move(or0[0], nextOr[0] + 180, t, t0, 10)
                    orientation = R.from_euler('xyz', [180 - yaw, -pitch, roll - 180], degrees=True)
                    d.mocap_quat[1] = np.array(orientation.as_quat())
                    mujoco.mj_step(m, d)
                    viewer.sync()
                    t += timeStep
        for l in range(-roll_step, -roll_rot - 1, -roll_step):
            or0 = nextOr
            or0[0] += 180
            nextOr = [grigliaRad[f"rot_{l}"], pitch, yaw]
            t = 0
            while t <= tr:                      # rotazione negativa
                roll = move(or0[0], nextOr[0] + 180, t, t0, tr - t0)
                orientation = R.from_euler('xyz', [180 - yaw, -pitch, roll - 180], degrees=True)
                d.mocap_quat[1] = np.array(orientation.as_quat())
                mujoco.mj_step(m, d)
                viewer.sync()
                t += timeStep
            if l == -roll_rot:
                or0 = nextOr
                or0[0] += 180
                nextOr = [grigliaRad[f"rot_{0}"], pitch, yaw]
                t = 0
                while t <= 10:                          # ritorno a 0
                    roll = move(or0[0], nextOr[0] + 180, t, t0, 10)
                    orientation = R.from_euler('xyz', [180 - yaw, -pitch, roll - 180], degrees=True)
                    d.mocap_quat[1] = np.array(orientation.as_quat())
                    mujoco.mj_step(m, d)
                    viewer.sync()
                    t += timeStep

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
