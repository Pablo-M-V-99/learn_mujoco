import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

plot = True
# plot = False

# Carica il file npz
data = np.load('datasets/dati_20241012_174901/PP_depth_and_labels.npz')
depth_images = data['depth_images']
img_size = depth_images.shape[1]
# cropped_imgs = np.empty((len(depth_images), img_size, img_size))
found = False


for img_idx in range(np.shape(depth_images)[0]):
    depth_img = depth_images[img_idx]
    cropped_img = np.empty((depth_img.shape[0], depth_img.shape[1]))

    # find upper corner
    for i in range(img_size):
        if not np.array_equal(depth_img[i, :], np.zeros(img_size)):
            for j in range(img_size):
                if depth_img[i, j] != 0:
                    bound_up = [i, j]
                    found = True
                    break
        if found:
            found = False
            break

    # find leftest point
    for i in range(img_size):
        if not np.array_equal(depth_img[:, i], np.zeros(img_size)):
            for j in range(img_size):
                if depth_img[j, i] != 0:
                    bound_Sx = [j, i]
                    found = True
                    break
        if found:
            found = False
            break

    # find rightest point
    for i in range(len(depth_img[img_size - 1]) - 1, -1, -1):
        if not np.array_equal(depth_img[:, i], np.zeros(img_size)):
            for j in range(img_size):
                if depth_img[j, i] != 0:
                    bound_Dx = [j, i]
                    found = True
                    break
        if found:
            found = False
            break

    cropped_img = depth_img[bound_up[0]:, :]
    cropped_img = cropped_img[:, bound_Sx[1]:]
    cropped_img = cropped_img[:, :(bound_Dx[1] - bound_Sx[1] + 1)]


    if plot:
        _, ax = plt.subplots(ncols=2)
        ax[0].imshow(cropped_img)
        ax[1].imshow(depth_img)

        # plt.plot(cropped_img)
        plt.show()

