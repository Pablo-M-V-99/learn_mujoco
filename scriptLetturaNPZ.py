import numpy as np
import random
import matplotlib.pyplot as plt


# Carica il file npz
data = np.load('datasets/big_dataset/depth_and_labels.npz')

# Visualizza le chiavi (nomi degli array salvati)ll
print(data.files)

poses = data['poses']
angles = data['angles']
labels = np.hstack((poses, angles))
depth_images = data['depth_images']

print(f"Il numero di immagini acquisite è {np.shape(depth_images)[0]}")
for k in range(np.shape(depth_images)[0]):
    if random.uniform(0, 1) < 0.005:
        plt.imshow(depth_images[k])
        plt.title(f'orientation:[{round(angles[k][0], 1)}, {round(angles[k][1], 1)}, {round(angles[k][2], 1)}] + '
                  f'position:[{round(poses[k][0],2)}, {round(poses[k][1],2)}, {round(poses[k][2],2)}]')
        plt.show()
