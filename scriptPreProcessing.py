import numpy as np
import cv2
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime


# plot = True
plot = False
datasets_path = 'datasets'
depth_imgs = []
labels = []

top_side_mask = np.ones((512, 512))
top_side_mask[:180, :] = 0

bottom_side_mask = np.ones((512, 512))
bottom_side_mask[450:, :] = 0

crop_size = [[90, 512], [45, 512 - 45]]
# Genera il timestamp corrente
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")



for folder_name in os.listdir(datasets_path):
    folder_path = os.path.join(datasets_path, folder_name)
    # Controlla se è una directory
    if os.path.isdir(folder_path):
        file_path = os.path.join(folder_path, "depth_and_labels.npz")
        # Controlla se il file immagine_e_posa.npz esiste nella cartella
        if os.path.exists(file_path):
            # Carica il file .npz
            data = np.load(file_path)
            depth_imgs = data['depth_images']
            depth_raws = [np.zeros((224, 224))] * len(depth_imgs)
            labels = np.hstack((data['poses'], data['angles']))  # x y z rotX rotY rotZ

            for k in range(np.shape(depth_imgs)[0]):
                depth_raw = depth_imgs[k][crop_size[0][0]: crop_size[0][1], crop_size[1][0]: crop_size[1][1]]
                depth_raw = cv2.resize(depth_raw, (224, 224)).astype(np.float32)     #???????????????
                depth_raws[k] = depth_raw
                if labels[k, 5] < 0:
                    labels[k, 5] += 360
                if plot:
                    fig, axs = plt.subplots(1, 2)
                    axs[0].imshow(depth_imgs[k])
                    axs[1].imshow(depth_raws[k])
                    axs[0].set_title('original')
                    axs[1].set_title('cropped')
                    plt.tight_layout()
                    plt.suptitle(
                        f'position:[{round(labels[k][0], 2)}, {round(labels[k][1], 2)}, {round(labels[k][2], 2)}] + '
                        f'orientation:[{round(labels[k][3], 1)}, {round(labels[k][4], 1)}, {round(labels[k][5], 1)}]')
                    plt.show()

            np.savez(f"{folder_path}/PP_depth_and_labels.npz", depth_images = depth_raws, labels = labels)
            print(f"{current_time}: Completatato il preprocessing di {folder_path}")























