import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt


# Carica il file npz
data = np.load('datasets/dati_20241012_175413/PP_depth_and_labels.npz')

# Visualizza le chiavi (nomi degli array salvati)
print(data.files)

depth_images = data['depth_images']
# labels = np.hstack((data['poses'], data['angles']))
labels = data['labels']

print(f"Il numero di immagini acquisite è {np.shape(depth_images)[0]}")
for k in range(np.shape(depth_images)[0]):
    if random.uniform(0, 1) < 0.2:
        plt.imshow(depth_images[k])
        # plt.title(f'orientation:[{round(angles[k][0], 1)}, {round(angles[k][1], 1)}, {round(angles[k][2], 1)}] + '
        #           f'position:[{round(poses[k][0],2)}, {round(poses[k][1],2)}, {round(poses[k][2],2)}]')
        plt.title(f'orientation:[{round(labels[k][3], 0)}, {round(labels[k][4], 0)}, {round(labels[k][5], 0)}] + '
                  f'position:[{round(labels[k][0], 2)}, {round(labels[k][1], 2)}, {round(labels[k][2], 2)}]')
        plt.show()
