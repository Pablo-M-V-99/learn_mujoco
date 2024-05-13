def move(pos0, nextPose, t, t0, T):

    a0 = pos0
    a1 = 0
    a2 = 0
    a3 = 20 * (nextPose - pos0) / (2 * (T ** 3))
    a4 = 30 * (pos0 - nextPose) / (2 * (T ** 4))
    a5 = 12 * (nextPose - pos0) / (2 * (T ** 5))
    q = a0 + a1*(t - t0) + a2*((t - t0) ** 2) + a3*((t - t0) ** 3) + a4*((t - t0) ** 4) + a5*((t - t0) ** 5)
    return q


if __name__ == "__main__":
    print("Questo script non deve essere eseguito autonomamente.")
