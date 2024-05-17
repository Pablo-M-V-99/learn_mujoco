
import time
import mujoco.viewer
import numpy as np
import json
import scriptGriglia
import scriptTraiettoria

# PARAMETRI
t0 = 0
t1 = 5          # durata movimento da un punto della griglia all'altro
T = t1 - t0
len_G = 0.3     # lunghezza griglia
wid_G = 0.4     # larghezza griglia
height_G = 0.4  # altezza griglia
dimCell = 0.01  # dimensiona cella
offX, offY, offZ = 0.2, 0.45, 0.2     # offset per l'allineamento della griglia
i, j, k = 0, 0, 0     # coordinate cella

scriptGriglia.creazioneGriglia(len_G, wid_G, height_G, dimCell, offX, offY, offZ)
json_path = (f"/home/pablo/PycharmProjects/learn_mujoco/griglia_{int(wid_G * 100)}x"
             f"{int(len_G * 100)}x{int(height_G * 100)}.json")
with open(json_path, "r") as file:
    griglia = json.load(file)

xml_path = "/home/pablo/PycharmProjects/mujoco/model/plugin/elasticity/muoviBandiera.xml"

m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)

timeStep = m.opt.timestep               # mujoco simulation timestep
pos0 = np.array(d.mocap_pos[1])         # posa iniziale mocap body
nextPose = griglia[f"cella_{i}_{j}_{k}"]
t = 0
while t <= 20:
    t += timeStep
    mujoco.mj_step(m, d)

with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()

    while viewer.is_running() and i <= wid_G * 100 and j <= len_G * 100:

        # MOVIMENTO SU GRIGLIA TRASVERSALE

        t = 0
        # Allineamento con la griglia
        while t <= 25 and i == 0 and j == 0 and k == 0:
            x = scriptTraiettoria.move(pos0[0], nextPose[0], t, t0, 25)
            y = scriptTraiettoria.move(pos0[1], nextPose[1], t, t0, 25)
            z = scriptTraiettoria.move(pos0[2], nextPose[2], t, t0, 25)
            d.mocap_pos[1] = np.array([x, y, z])
            mujoco.mj_step(m, d)
            viewer.sync()
            t += timeStep
            if t == 25:
                i += 1

        pos0 = nextPose
        nextPose = griglia[f"cella_{i}_{j}_{k}"]
        t = 0

        while t <= t1:
            x = scriptTraiettoria.move(pos0[0], nextPose[0], t, t0, T)
            d.mocap_pos[1] = np.array([x, y, z])
            mujoco.mj_step(m, d)
            viewer.sync()
            t += timeStep

        if j % 2 == 0:
            if i == wid_G * 100:
                j += 1
                pos0 = nextPose
                nextPose = griglia[f"cella_{i}_{j}_{k}"]
                t = 0

                while t <= t1:
                    y = scriptTraiettoria.move(pos0[1], nextPose[1], t, t0, T)
                    d.mocap_pos[1] = np.array([x, y, z])
                    mujoco.mj_step(m, d)
                    viewer.sync()
                    t += timeStep

            else:
                i += 1
        else:
            if i == 0:
                j += 1
                pos0 = nextPose
                nextPose = griglia[f"cella_{i}_{j}_{k}"]
                t = 0

                while t <= t1:
                    y = scriptTraiettoria.move(pos0[1], nextPose[1], t, t0, T)
                    d.mocap_pos[1] = np.array([x, y, z])
                    mujoco.mj_step(m, d)
                    viewer.sync()
                    t += timeStep
            else:
                i -= 1

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
