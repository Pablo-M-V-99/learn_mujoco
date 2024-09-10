import mujoco.viewer
import numpy as np

def speedCtrl(m, d, viewer, prevPose)-> bool:
    """
    Verifica che tutti i punti del corpo siano fermi o con velocità inferiori ad un valore soglia
    :param m: Mujoco model
    :param d: Mujoco Data
    :param viewer: consente la sincronizzazione del viewer
    :param prevPose: posa all'istante precedente
    :return:
    """
    counter = 0
    max_counter = 10
    speed_threshold = 0.02
    timeStep = m.opt.timestep  # mujoco simulation timestep

    while True:
        currentPose = np.array([d.body(f"flag_{i}").xpos for i in range(d.flexvert_xpos.shape[0])])
        lin_speed = (currentPose - prevPose) / timeStep
        max_speed = np.max(np.linalg.norm(lin_speed, axis=1))
        t = 0

        if max_speed > speed_threshold:
            counter += 1
            # print(f"Sono stato fermo {counter/2} secondi")
            while t <= 0.5:
                mujoco.mj_step(m, d)
                if viewer:
                    viewer.sync()
                t += timeStep
        else:
            return True

        if counter > max_counter:
            print(f"Il counter ha superato il valore soglia. Corpo non fermo")
            return False

if __name__ == "__main__":
    print("Questo script non deve essere eseguito autonomamente.")
