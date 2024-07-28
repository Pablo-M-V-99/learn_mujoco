import mujoco.viewer
import numpy as np
from scriptControlloVelocità import speedCtrl
from scipy.spatial.transform import Rotation as R


def move(pos0, nextPose, t, T):
    a0 = pos0
    a1 = 0
    a2 = 0
    a3 = 20 * (nextPose - pos0) / (2 * (T ** 3))
    a4 = 30 * (pos0 - nextPose) / (2 * (T ** 4))
    a5 = 12 * (nextPose - pos0) / (2 * (T ** 5))
    q = a0 + a1*t + a2*(t ** 2) + a3*(t ** 3) + a4*(t ** 4) + a5*(t ** 5)
    return q


def angleStep(m, d, viewer, or0, nextOr, tr, timeStep):
    t = 0
    while t <= tr:
        yaw = move(or0[2], nextOr[2], t, tr)
        pitch = move(or0[1], nextOr[1], t, tr)
        roll = move(or0[0], nextOr[0], t, tr)
        orientation = R.from_euler('xyz', [180 - yaw, -pitch, 180 - roll], degrees=True)
        if t >= tr - 2 * timeStep:  # memorizzazione ultima posa per il controllo velocità
            prevPose = np.array([d.body(f"flag_{ii}").xpos for ii in range(171)])
        d.mocap_quat[1] = np.array(orientation.as_quat())
        mujoco.mj_step(m, d)
        if viewer:
            viewer.sync()
        t += timeStep
    speedCtrl(prevPose, m, d, timeStep)
    return roll, pitch, yaw, nextOr


def posStep(m, d, viewer, pos0, nextPose, T, timeStep):
    t = 0
    while t <= T:
        x = move(pos0[0], nextPose[0], t, T)  # coordinate mocap_body
        y = move(pos0[1], nextPose[1], t, T)
        z = move(pos0[2], nextPose[2], t, T)
        if t >= T - 2 * timeStep:   # memorizzazione ultima posa per il controllo velocità
            prevPose = np.array([d.body(f"flag_{ii}").xpos for ii in range(171)])
        d.mocap_pos[1] = np.array([x, y, z])
        mujoco.mj_step(m, d)
        if viewer:
            viewer.sync()
        t += timeStep
    speedCtrl(prevPose, m, d, timeStep)


if __name__ == "__main__":
    print("Questo script non deve essere eseguito autonomamente.")
