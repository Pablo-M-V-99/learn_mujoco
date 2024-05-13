import time
import mujoco.viewer
import numpy as np
import math
import json

pi = math.pi

# PARAMETRI
t0 = 0
t1 = 5      # durata movimento da un punto della griglia all'altro
T = t1 - t0
lunghezzaGriglia = 0.3
larghezzaGriglia = 0.4
i, j = 0, 0 # coordinate cella

json_path = "/home/pablo/PycharmProjects/learn_mujoco/griglia_40x30.json"
with open(json_path, "r") as file:
    griglia = json.load(file)

xml_path = "/home/pablo/PycharmProjects/mujoco/model/plugin/elasticity/muoviBandiera.xml"

m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)

timeStep = m.opt.timestep       # mujoco simulation timestep
pos0 = np.array(d.mocap_pos[1])         # posa iniziale mocap body
nextPose = griglia[f"cella_{i}_{j}"]

with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()
    # MOVIMENTO TRASLATORIO

    # for element in griglia
    # MAKE TRAJECTORY(ELEM TO ELEM+1)
    #     FOR POINT TRAJECTORY
    #         MOCAP POS = POINT
    #         STEP

    while viewer.is_running() and i <= larghezzaGriglia * 100 and j <= lunghezzaGriglia * 100 :

        t = 0
        # allineamento con la griglia
        while t <= 25 and i == 0 and j == 0:

            ax0 = pos0[0]
            ax1 = 0
            ax2 = 0
            ax3 = 20 * (nextPose[0] - pos0[0]) / (2 * ((25 - t0) ** 3))
            ax4 = 30 * (pos0[0] - nextPose[0]) / (2 * ((25 - t0) ** 4))
            ax5 = 12 * (nextPose[0] - pos0[0]) / (2 * ((25 - t0) ** 5))
            x = ax0 + ax1*(t - t0) + ax2*((t - t0) ** 2) + ax3*((t - t0) ** 3) + ax4*((t - t0) ** 4) + ax5*((t - t0) ** 5)

            ay0 = pos0[1]
            ay1 = 0
            ay2 = 0
            ay3 = 20 * (nextPose[1] - pos0[1]) / (2 * ((25 - t0) ** 3))
            ay4 = 30 * (pos0[1] - nextPose[1]) / (2 * ((25 - t0) ** 4))
            ay5 = 12 * (nextPose[1] - pos0[1]) / (2 * ((25 - t0) ** 5))
            y = ay0 + ay1*(t - t0) + ay2*((t - t0) ** 2) + ay3*((t - t0) ** 3) + ay4*((t - t0) ** 4) + ay5*((t - t0) ** 5)

            d.mocap_pos[1] = np.array([x, y, 1.5])
            mujoco.mj_step(m, d)
            viewer.sync()
            t += timeStep
            if t == 25:
                i += 1

        pos0 = nextPose
        nextPose = griglia[f"cella_{i}_{j}"]

        t = 0
        # movimento sull'asse X
        while t <= t1:

            ax0 = pos0[0]
            ax1 = 0
            ax2 = 0
            ax3 = 20 * (nextPose[0] - pos0[0]) / (2 * (T ** 3))
            ax4 = 30 * (pos0[0] - nextPose[0]) / (2 * (T ** 4))
            ax5 = 12 * (nextPose[0] - pos0[0]) / (2 * (T ** 5))

            x = ax0 + ax1*(t - t0) + ax2*((t - t0) ** 2) + ax3*((t - t0) ** 3) + ax4*((t - t0) ** 4) + ax5*((t - t0) ** 5)

            d.mocap_pos[1] = np.array([x, y, 1.5])

            mujoco.mj_step(m, d)
            viewer.sync()

            t += timeStep

        # TRAIETTORIA a S
        if j % 2 == 0:
            if i == larghezzaGriglia * 100:
                j += 1
                pos0 = nextPose
                nextPose = griglia[f"cella_{i}_{j}"]
                i -= 1

                t = 0
                # movimento sull'asse Y
                while t <= t1:
                    ay0 = pos0[1]
                    ay1 = 0
                    ay2 = 0
                    ay3 = 20 * (nextPose[1] - pos0[1]) / (2 * (T ** 3))
                    ay4 = 30 * (pos0[1] - nextPose[1]) / (2 * (T ** 4))
                    ay5 = 12 * (nextPose[1] - pos0[1]) / (2 * (T ** 5))

                    y = ay0 + ay1 * (t - t0) + ay2 * ((t - t0) ** 2) + ay3 * ((t - t0) ** 3) + ay4 * ((t - t0) ** 4) + ay5 * ((t - t0) ** 5)

                    d.mocap_pos[1] = np.array([x, y, 1.5])

                    mujoco.mj_step(m, d)
                    viewer.sync()

                    t += timeStep
            i += 1

        else:
            if i == 0:
                j += 1
                pos0 = nextPose
                nextPose = griglia[f"cella_{i}_{j}"]

                t = 0
                # movimento sull'asse Y
                while t <= t1:
                    ay0 = pos0[1]
                    ay1 = 0
                    ay2 = 0
                    ay3 = 20 * (nextPose[1] - pos0[1]) / (2 * (T ** 3))
                    ay4 = 30 * (-nextPose[1] + pos0[1]) / (2 * (T ** 4))
                    ay5 = 12 * (nextPose[1] - pos0[1]) / (2 * (T ** 5))

                    y = ay0 + ay1 * (t - t0) + ay2 * ((t - t0) ** 2) + ay3 * ((t - t0) ** 3) + ay4 * ((t - t0) ** 4) + ay5 * (
                                (t - t0) ** 5)

                    d.mocap_pos[1] = np.array([x, y, 1.5])

                    mujoco.mj_step(m, d)
                    viewer.sync()

                    t += timeStep
                i += 1
            i -= 1

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - start)

        if time_until_next_step > 0:
            time.sleep(time_until_next_step)