from scriptTraiettoria import angleStep


def firstRot(flag, m, d, viewer, first_step, first_rot, sec_step, sec_rot, or0,
             roll, pitch, yaw, grigliaRad, tr, tR, timeStep):

    for q in range(first_step, first_rot + first_step, first_step):

        if flag == 'YAW':
            nextOr = [roll, pitch, grigliaRad[f"rot_{q}"]]
            flag2 = 'PITCH'
        if flag == 'PITCH':
            nextOr = [roll, grigliaRad[f"rot_{q}"], yaw]
            flag2 = 'YAW'

        roll, pitch, yaw, or0 = angleStep(m, d, viewer, or0, nextOr, tr, timeStep)
        secondRot(flag2, m, d, viewer, sec_step, sec_rot, or0, roll, pitch, yaw, grigliaRad, tr, tR, timeStep)
        secondRot(flag2, m, d, viewer, -sec_step, -sec_rot, or0, roll, pitch, yaw, grigliaRad, tr, tR, timeStep)

        # riallineamento
        if q == first_rot:
            if flag == 'YAW':
                nextOr = [roll, pitch, grigliaRad[f"rot_{0}"]]
            if flag == 'PITCH':
                nextOr = [roll, grigliaRad[f"rot_{0}"], yaw]
            angleStep(m, d, viewer, or0, nextOr, tR, timeStep)

def secondRot(flag, m, d, viewer, sec_step, sec_rot, or0,
              roll, pitch, yaw, grigliaRad, tr, tR, timeStep):

    for q in range(sec_step, sec_rot + sec_step, sec_step):

        if flag == 'YAW':
            nextOr = [roll, pitch, grigliaRad[f"rot_{q}"]]
        if flag == 'PITCH':
            nextOr = [roll, grigliaRad[f"rot_{q}"], yaw]

        roll, pitch, yaw, or0 = angleStep(m, d, viewer, or0, nextOr, tr, timeStep)

        # riallineamento
        if q == sec_rot:
            if flag == 'YAW':
                nextOr = [roll, pitch, grigliaRad[f"rot_{0}"]]
            if flag == 'PITCH':
                nextOr = [roll, grigliaRad[f"rot_{0}"], yaw]
            angleStep(m, d, viewer, or0, nextOr, tR, timeStep)


if __name__ == "__main__":
    print("Questo script non deve essere eseguito autonomamente.")
