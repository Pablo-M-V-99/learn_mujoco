import time
import mujoco.viewer
import numpy as np
import math
import json

pi = math.pi
deg_to_rad = pi / 180
rad_to_deg = 180 / pi
incremento = 0.1
alfa = 0
VIEWER = True

def sin(alfa):
    return math.sin(alfa*deg_to_rad)

def cos(alfa):
    return math.cos(alfa*deg_to_rad)

json_path = "/home/pablo/PycharmProjects/learn_mujoco/griglia_45x30.json"
with open(json_path, "r") as file:
    griglia = json.load(file)

xml_path = "/home/pablo/PycharmProjects/mujoco/model/plugin/elasticity/muoviBandiera.xml"

m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)

pos0 = np.array(d.mocap_pos[1])

with mujoco.viewer.launch_passive(m, d) as viewer:

    start = time.time()
    while viewer.is_running():
        step_start = time.time()

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.

        # for element in griglia
        # MAKE TRAJECTORY(ELEM TO ELEM+1)
        #     FOR POINT TRAJECTORY
        #         MOCAP POS = POINT
        #         STEP

        # MOVIMENTO TRASLATORIO
        mujoco.mj_step(m, d)

        if alfa <= 90 and alfa >= -90 :     # velocità nulla all'inversione
            alfa += incremento
            d.mocap_pos[1] = pos + np.array([0, 0, 0.5*sin(alfa)])
        else:
            incremento = -incremento
            alfa += incremento

        # ROTAZIONE
        # if alfa <= 180 and alfa >= -180 :
        #     alfa += incremento
        #     # d.mocap_quat[1] = [cos(alfa/2), sin(alfa/2), 0, 0]      # roll rotation INUTILE
        #     d.mocap_quat[1] = [cos(alfa/2), 0, sin(alfa/2), 0]      # pitch rotation  tra 180 e -180
        #     # d.mocap_quat[1] = [cos(alfa/2), 0, 0, sin(alfa/2)]      # yaw rotation
        # else:
        #     incremento = -incremento
        #     alfa += incremento

        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)