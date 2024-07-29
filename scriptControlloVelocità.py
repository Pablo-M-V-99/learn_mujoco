import mujoco.viewer
import numpy as np

def speedCtrl(prevPose, m, d, timeStep)-> bool:
    counter = 0
    max_counter = 10
    while True:

        currentPose = np.array([d.body(f"flag_{i}").xpos for i in range(d.flexvert_xpos.shape[0])])
        lin_speed = (currentPose - prevPose) / timeStep
        max_speed = np.max(np.linalg.norm(lin_speed, axis=1))
        # ang_speed = [(R.from_quat(d.body(f"flag_{i}").xquat).as_euler('xyz') -
        #              R.from_quat(prevPose(f"flag_{i}").xquat).as_euler('xyz')) / timeStep for i in range(n_body)]
        t = 0

        if max_speed > 0.01:
            counter += 1
            print(f"Sono stato fermo {counter/2} secondi")
            while t <= 0.5:
                mujoco.mj_step(m, d)
                t += timeStep
        else:
            return True

        if counter > max_counter:
            print(f"Il counter ha superato il valore soglia. Corpo non fermo")
            return False

if __name__ == "__main__":
    print("Questo script non deve essere eseguito autonomamente.")
