from scipy.spatial.transform import Rotation as R
import mujoco.viewer
import numpy as np

def checkVel(prev_body, m, d, timeStep, viewer)-> bool:
    n_body=171
    counter = 0
    max_counter = 10
    while True:

        lin_speed = [(d.body(f"flag_{i}").xpos - prev_body(f"flag_{i}").xpos) / timeStep for i in range(n_body)]
        # ang_speed = [(R.from_quat(d.body(f"flag_{i}").xquat).as_euler('xyz') -
        #              R.from_quat(prev_body(f"flag_{i}").xquat).as_euler('xyz')) / timeStep for i in range(n_body)]
        t = 0

        if np.max(np.linalg.norm(lin_speed, axis=1)) > 0.01 :
            while t <= 1:
                mujoco.mj_step(m, d)
                viewer.sync()
                t += timeStep
        else:
            return True

        if counter > max_counter:
            return False



if __name__ == "__main__":
    print("Questo script non deve essere eseguito autonomamente.")
