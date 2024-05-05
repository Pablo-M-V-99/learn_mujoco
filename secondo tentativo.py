import time
import mujoco
import mujoco.viewer
import numpy as np
import math

pi = math.pi
deg_to_rad = pi / 180
rad_to_deg = 180 / pi
incremento = 0.1

def sin(alfa):
    return math.sin(alfa*deg_to_rad)

def cos(alfa):
    return math.cos(alfa*deg_to_rad)

xml_path = "/home/pablo/PycharmProjects/mujoco/model/plugin/elasticity/muoviBandiera.xml"

m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)

alfa = 0

with mujoco.viewer.launch_passive(m, d) as viewer:

    start = time.time()
    while viewer.is_running():
        step_start = time.time()

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.

        mujoco.mj_step(m, d)


        # MOVIMENTO LEMBO SINISTRO
        if alfa <= 90 and alfa >= -90 :     # velocità nulla all'inversione
            alfa += incremento
            d.mocap_pos[1] = np.array([0, 0, sin(alfa)])
        else:
            incremento = -incremento
            alfa += incremento

        # ROTAZIONE
        # if alfa <= 180 and alfa >= -180 :
        #     alfa += incremento
        #     # d.mocap_quat[3] = [cos(alfa/2), sin(alfa/2), 0, 0]      # roll rotation INUTILE
        #     d.mocap_quat[3] = [cos(alfa/2), 0, sin(alfa/2), 0]      # pitch rotation  tra 180 e -180
        #     # d.mocap_quat[3] = [cos(alfa/2), 0, 0, sin(alfa/2)]      # yaw rotation
        # else:
        #     incremento = -incremento
        #     alfa += incremento

        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)