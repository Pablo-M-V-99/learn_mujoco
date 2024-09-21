import numpy as np
import random
import matplotlib.pyplot as plt


# Carica il file npz
data = np.load('datasets/dati_20240921_182929/depth_and_labels.npz')

# Visualizza le chiavi (nomi degli array salvati)
print(data.files)

poses = data['poses']
angles = data['angles']
depth_images = data['depth_images']

print(f"Il numero di immagini acquisite è {np.shape(depth_images)[0]}")
for k in range(np.shape(depth_images)[0]):
    if random.uniform(0, 1) < 0.2:
        plt.imshow(data['depth_images'][k])
        plt.show()
