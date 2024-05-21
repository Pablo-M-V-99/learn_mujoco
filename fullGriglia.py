import time
import mujoco.viewer
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from scriptGriglia import creazioneGriglia as cartesianGrid
from scriptGriglia import creazioneGrigliaRadiale as radialGrid
from scriptTraiettoria import move

# PARAMETRI
t0 = 0          # tempo iniziale
t1 = 5          # durata movimento da un punto della griglia all'altro
tz = 10         # durata cambio piano
tr = 3          # durata riallineamento
T = t1 - t0
len_G = 0.02         # lunghezza griglia (righe)
wid_G = 0.02         # larghezza griglia (colonne)
height_G = 0.02      # altezza griglia (piani)
dimCell = 0.1      # dimensiona cella
offX, offY, offZ = 0, 0.7, 0.7     # offset per l'allineamento della griglia
i, j, k = 0, 0, 0     # coordinate cella
x, y, z = 0, 0, 0     # coordinate spaziali
pitch_rot, yaw_rot = 10, 10
pitch_step, yaw_step = 5, 5     # la rotazione deve essere divisibile per gli incrementi!


# CREAZIONE GRIGLIA CARTESIANA
cartesianGrid(len_G, wid_G, height_G, dimCell, offX, offY, offZ)
json_path = (f"/home/pablo/PycharmProjects/learn_mujoco/griglia_{int(wid_G * 100)}x"
             f"{int(len_G * 100)}x{int(height_G * 100)}.json")
with open(json_path, "r") as file:
    griglia = json.load(file)

# CREAZIONE GRIGLIA RADIALE
radialGrid()
json_path = (f"/home/pablo/PycharmProjects/learn_mujoco/grigliaRadiale.json")
with open(json_path, "r") as file:
    grigliaRad = json.load(file)

# CARICAMENTO MODELLO MUJOCO XML
xml_path = "/home/pablo/PycharmProjects/mujoco/model/plugin/elasticity/muoviBandiera.xml"

m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)

timeStep = m.opt.timestep               # mujoco simulation timestep
pos0 = np.array(d.mocap_pos[1])         # posa iniziale mocap body
nextPose = griglia[f"cella_{i}_{j}_{k}"]

# np.array(d.mocap_quat[1]) è l'orientamento in quaternioni come me lo dà mujoco -> [1 0 0 0]. pitch0.as_quat() è
# l'orientamento in quaternioni da scipy ->[1 0 0 0]. pitch0.as_euler() è l'orientamento in angoli di Eulero da
# scipy -> [180 0 0] (nell'ordine roll pitch yaw)
or0 = R.from_quat(np.array(d.mocap_quat[1]))
or0 = or0.as_euler('xyz', degrees=True)
t = 0

yaw, pitch, roll = or0[2], or0[1], or0[0]  # orientamento (terna fissa) yaw(z) pitch(y) roll(x)
or0_init = or0

# Transitorio spawn bandiera
while t <= 20:
    t += timeStep
    mujoco.mj_step(m, d)

with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()

    while viewer.is_running() and i <= wid_G * 100 and j <= len_G * 100 and k <= height_G * 100:

        # MOVIMENTO SU TUTTA LA GRIGLIA

        t = 0
        # Allineamento con la griglia
        while t <= 25 and i == 0 and j == 0 and k == 0:
            x = move(pos0[0], nextPose[0], t, t0, 25)
            y = move(pos0[1], nextPose[1], t, t0, 25)
            z = move(pos0[2], nextPose[2], t, t0, 25)
            d.mocap_pos[1] = np.array([x, y, z])
            mujoco.mj_step(m, d)
            viewer.sync()
            t += timeStep
            if int(t) == 25:
                i += 1

        # ROTAZIONE IN YAW
        for p in range(yaw_step, yaw_rot + 1, yaw_step):
            nextOr = [roll, pitch, grigliaRad[f"rot_{p}"]]
            t = 0
            while t <= tr:  # rotazione positiva
                yaw = move(or0[2], nextOr[2], t, t0, tr - t0)
                orientation = R.from_euler('xyz', [180 - yaw, -pitch, 180 - roll], degrees=True)
                d.mocap_quat[1] = np.array(orientation.as_quat())
                mujoco.mj_step(m, d)
                viewer.sync()
                t += timeStep
            # ROTAZIONE IN PITCH
            for q in range(pitch_step, pitch_rot + 1, pitch_step):
                or0 = nextOr
                nextOr = [roll, grigliaRad[f"rot_{q}"], yaw]
                t = 0
                while t <= tr:  # rotazione positiva
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
                    while t <= 10:  # ritorno a 0
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
                while t <= tr:  # rotazione negativa
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
                    while t <= 10:  # ritorno a 0
                        pitch = move(or0[1], nextOr[1], t, t0, 10)
                        orientation = R.from_euler('xyz', [180 - yaw, -pitch, 180 + roll], degrees=True)
                        d.mocap_quat[1] = np.array(orientation.as_quat())
                        mujoco.mj_step(m, d)
                        viewer.sync()
                        t += timeStep
            or0 = nextOr
            if p == yaw_rot:
                nextOr = [roll, pitch, grigliaRad[f"rot_{0}"]]
                t = 0
                while t <= 10:  # ritorno a 0
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
            while t <= tr:  # rotazione negativa
                yaw = move(or0[2], nextOr[2], t, t0, tr - t0)
                orientation = R.from_euler('xyz', [180 - yaw, -pitch, 180 - roll], degrees=True)
                d.mocap_quat[1] = np.array(orientation.as_quat())
                mujoco.mj_step(m, d)
                viewer.sync()
                t += timeStep
            # ROTAZIONE IN PITCH
            for q in range(pitch_step, pitch_rot + 1, pitch_step):
                or0 = nextOr
                nextOr = [roll, grigliaRad[f"rot_{q}"], yaw]
                t = 0
                while t <= tr:  # rotazione positiva
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
                    while t <= 10:  # ritorno a 0
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
                while t <= tr:  # rotazione negativa
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
                    while t <= 10:  # ritorno a 0
                        pitch = move(or0[1], nextOr[1], t, t0, 10)
                        orientation = R.from_euler('xyz', [180 - yaw, -pitch, 180 + roll], degrees=True)
                        d.mocap_quat[1] = np.array(orientation.as_quat())
                        mujoco.mj_step(m, d)
                        viewer.sync()
                        t += timeStep
            if p == -yaw_rot:
                or0 = nextOr
                nextOr = [roll, pitch, grigliaRad[f"rot_{0}"]]
                t = 0
                while t <= 10:  # ritorno a 0
                    yaw = move(or0[2], nextOr[2], t, t0, 10)
                    orientation = R.from_euler('xyz', [180 - yaw, -pitch, 180 - roll], degrees=True)
                    d.mocap_quat[1] = np.array(orientation.as_quat())
                    mujoco.mj_step(m, d)
                    viewer.sync()
                    t += timeStep
        or0 = or0_init

        # ROTAZIONE IN PITCH
        for q in range(pitch_step, pitch_rot + 1, pitch_step):
            nextOr = [roll, grigliaRad[f"rot_{q}"], yaw]
            t = 0
            while t <= tr:  # rotazione positiva
                pitch = move(or0[1], nextOr[1], t, t0, tr - t0)
                orientation = R.from_euler('xyz', [180 - yaw, -pitch, 180 - roll], degrees=True)
                d.mocap_quat[1] = np.array(orientation.as_quat())
                mujoco.mj_step(m, d)
                viewer.sync()
                t += timeStep
            # ROTAZIONE IN YAW
            for p in range(yaw_step, yaw_rot + 1, yaw_step):
                or0 = nextOr
                nextOr = [roll, pitch, grigliaRad[f"rot_{p}"]]
                t = 0
                while t <= tr:  # rotazione positiva
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
                    while t <= 10:  # ritorno a 0
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
                while t <= tr:  # rotazione negativa
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
                    while t <= 10:  # ritorno a 0
                        yaw = move(or0[2], nextOr[2], t, t0, 10)
                        orientation = R.from_euler('xyz', [180 - yaw, -pitch, 180 - roll], degrees=True)
                        d.mocap_quat[1] = np.array(orientation.as_quat())
                        mujoco.mj_step(m, d)
                        viewer.sync()
                        t += timeStep

            or0 = nextOr
            if q == pitch_rot:
                nextOr = [roll, grigliaRad[f"rot_{0}"], yaw]
                t = 0
                while t <= 10:  # ritorno a 0
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
            while t <= tr:  # rotazione negativa
                pitch = move(or0[1], nextOr[1], t, t0, tr - t0)
                orientation = R.from_euler('xyz', [180 - yaw, -pitch, 180 + roll], degrees=True)
                d.mocap_quat[1] = np.array(orientation.as_quat())
                mujoco.mj_step(m, d)
                viewer.sync()
                t += timeStep
            # ROTAZIONE IN YAW
            for p in range(yaw_step, yaw_rot + 1, yaw_step):
                nextOr = [roll, pitch, grigliaRad[f"rot_{p}"]]
                t = 0
                while t <= tr:  # rotazione positiva
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
                    while t <= 10:  # ritorno a 0
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
                while t <= tr:  # rotazione negativa
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
                    while t <= 10:  # ritorno a 0
                        yaw = move(or0[2], nextOr[2], t, t0, 10)
                        orientation = R.from_euler('xyz', [180 - yaw, -pitch, 180 - roll], degrees=True)
                        d.mocap_quat[1] = np.array(orientation.as_quat())
                        mujoco.mj_step(m, d)
                        viewer.sync()
                        t += timeStep
            if q == -pitch_rot:
                or0 = nextOr
                nextOr = [roll, grigliaRad[f"rot_{0}"], yaw]
                t = 0
                while t <= 10:  # ritorno a 0
                    pitch = move(or0[1], nextOr[1], t, t0, 10)
                    orientation = R.from_euler('xyz', [180 - yaw, -pitch, 180 + roll], degrees=True)
                    d.mocap_quat[1] = np.array(orientation.as_quat())
                    mujoco.mj_step(m, d)
                    viewer.sync()
                    t += timeStep
        or0 = or0_init

        # Traslazione
        pos0 = nextPose
        nextPose = griglia[f"cella_{i}_{j}_{k}"]
        t = 0
        while t <= T:
            x = move(pos0[0], nextPose[0], t, t0, T)
            y = move(pos0[1], nextPose[1], t, t0, T)
            z = move(pos0[2], nextPose[2], t, t0, T)
            d.mocap_pos[1] = np.array([x, y, z])
            mujoco.mj_step(m, d)
            viewer.sync()
            t += timeStep
        T = t1 - t0

        # Movimento lungo X, Y e Z
        if j % 2 == 0:                                  # riga pari
            if i == wid_G * 100 and j != len_G * 100:   # incrementa riga se non sono all'ultima riga
                j += 1
            else:
                if i != wid_G * 100:                    # incrementa colonna se non sono all'ultima colonna
                    i += 1
                else:                                   # inc. piano se all'ultima colonna e
                    k += 1
                    i = 0
                    j = 0
                    T = tz - t0
        else:                                           # riga dispari
            if i == 0 and j != len_G * 100:             # incrementa riga se non sono all'ultima riga
                j += 1
            else:
                if i != 0:                              # decrementa colonna se non sono alla prima colonna
                    i -= 1
                else:                                   # inc. piano se all'ultima colonna e
                    k += 1
                    i = 0
                    j = 0
                    T = tz - t0

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
