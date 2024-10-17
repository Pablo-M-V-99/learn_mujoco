import random
import numpy as np
from skimage.util import random_noise
from skimage.transform import rotate, AffineTransform, warp
import cv2
import json
from torch.utils.data import Dataset
import torch
import os
import sklearn


class DeformationDataset(Dataset):
    """ Dataset immagini depth """

    # init is run once when instantiating the Dataset object
    def __init__(self,
                 datasets_path: str,
                 training: bool,
                 classification: bool,
                 max_translation_y: int = None,
                 max_translation_x: int = None,
                 max_rotation: float = None,
                 flip_left_right: bool = False,
                 gradient_thresholding: bool = False,
                 ):
        """
        Arguments:
            datasets_path: path to the folder containing the images of a certain simulation
            training: if training is performed
            classification: if we are going to perform classification or regression
            max_translation_y: maximum translation y [px]
            max_translation_x: maximum translation x [px]
            max_rotation: maximum rotation [degrees]
            flip_left_right: flip left right [True/False]
            gradient_thresholding: threshold for gradient [True/False]
        """

        super().__init__()
        self.datasets_path = datasets_path
        self.training = training
        self.classification = classification
        self.evaluation = False
        self.max_translation_y = max_translation_y
        self.max_translation_x = max_translation_x
        self.max_rotation = max_rotation
        self.flip_left_right = flip_left_right
        self.class_bins_trasl = np.array([-0.15, -0.049, 0.049, 0.15])
        self.class_bins_rot = np.array([-15, -4.9, 4.9, 15])
        self.deformation_classes = ['- Big', '- Small', 'None', '+ Small', '+ Big']
        self.axis = 5
        self.depth_imgs = []
        self.labels = []
        self.datasets_size = []
        self.n_datasets = 0
        idx0 = 0
        idx1 = -1
        # Per ogni cartella dentro la cartella "Datasets"
        for folder_name in os.listdir(self.datasets_path):
            folder_path = os.path.join(self.datasets_path, folder_name)
            # Controlla se è una directory
            if os.path.isdir(folder_path):
                # file_path = os.path.join(folder_path, "depth_and_labels.npz")
                file_path = os.path.join(folder_path, "PP_depth_and_labels.npz")
                # Controlla se il file PP_depth_and_labels.npz esiste nella cartella
                if os.path.exists(file_path):
                    # Carica il file .npz
                    data = np.load(file_path)
                    self.depth_imgs.append(data['depth_images'])
                    # self.labels.append(np.hstack((data['poses'], data['angles'])))
                    self.labels.append(data['labels'])
                    self.n_datasets += 1
                    idx1 += np.shape(data['depth_images'])[0]
                    self.datasets_size.append([idx0 , idx1])
                    idx0 = idx1 + 1

    # len returns the number of samples in the dataset
    def __len__(self):
        return self.datasets_size[len(self.datasets_size) - 1][1]

    # getitem loads and returns a sample from the dataset at the given index
    def __getitem__(self, index):

        i = 0
        while True:
            if index >= self.datasets_size[i][0]:
                if index <= self.datasets_size[i][1]:
                    img1 = np.copy(self.depth_imgs[i][index - self.datasets_size[i][0]])
                    lab1 = self.labels[i][index - self.datasets_size[i][0]]
                    bottom = self.datasets_size[i][0]
                    top = self.datasets_size[i][1]
                    random_index = random.randint(bottom, top)
                    img2 = self.depth_imgs[i][random_index - self.datasets_size[i][0]]
                    lab2 = self.labels[i][random_index - self.datasets_size[i][0]]
                    break
            i += 1

        if self.training:
            # Effetto riflesso
            if np.random.rand() > 0.5:
                k_size = 5

                grad_x_1 = cv2.Sobel(img1, cv2.CV_32F, 1, 0, ksize=k_size)
                grad_y_1 = cv2.Sobel(img1, cv2.CV_32F, 0, 1, ksize=k_size)
                grad_magnitude_1 = np.sqrt(grad_x_1 ** 2 + grad_y_1 ** 2)

                grad_x_2 = cv2.Sobel(img2, cv2.CV_32F, 1, 0, ksize=k_size)
                grad_y_2 = cv2.Sobel(img2, cv2.CV_32F, 0, 1, ksize=k_size)
                grad_magnitude_2 = np.sqrt(grad_x_2 ** 2 + grad_y_2 ** 2)

                # Define the threshold
                threshold = np.random.uniform(0.5, 1)
                img1 *= grad_magnitude_1 < threshold
                img2 *= grad_magnitude_2 < threshold

            # Traslazione dell'immagine
            if self.max_translation_x > 0 and self.max_translation_y > 0:
                transformation = AffineTransform(translation=(np.random.randint(low=-self.max_translation_x,
                                                                                high=self.max_translation_x + 1),
                                                              np.random.randint(low=-self.max_translation_y,
                                                                                high=self.max_translation_y + 1)))
                img1 = warp(img1, transformation)
                img2 = warp(img2, transformation)
            # Rotazione dell'immagine
            if self.max_rotation > 0:
                random_rot = np.random.uniform(low=-1, high=1) * self.max_rotation
                img1 = rotate(img1, angle=random_rot)
                img2 = rotate(img2, angle=random_rot)

            # Ribaltamento dell'immagine
            if self.flip_left_right:
                if np.random.rand() > 0.5:
                    img1 = np.ascontiguousarray(np.fliplr(img1))
                    img2 = np.ascontiguousarray(np.fliplr(img2))

                    lab1[0] *= -1   # traslazione X
                    lab1[4] *= -1   # rotazione Y
                    lab1[5] *= -1   # rotazione Z

                    lab2[0] *= -1
                    lab2[4] *= -1
                    lab2[5] *= -1

            # Applicazione di random noise
            if np.random.rand() > 0.5:
                img1 *= np.ones_like(img1) * random_noise(img1, mode='pepper', clip=False)
                img2 *= np.ones_like(img2) * random_noise(img2, mode='pepper', clip=False)

            # rimozione dei valori incoerenti causati dall'augmentation
            img1[img1 < 0] = 0
            img2[img2 < 0] = 0

        img1 = torch.tensor(np.expand_dims(img1, axis=0), dtype=torch.float32)
        img2 = torch.tensor(np.expand_dims(img2, axis=0), dtype=torch.float32)

        # rimozione di rotX
        lab1 = np.delete(lab1, [3])
        lab2 = np.delete(lab2, [3])

        delta_label = lab1 - lab2

        label = torch.tensor([np.digitize(d_label, bins=self.class_bins_trasl) for d_label in delta_label[:-2]] +
                             [np.digitize(d_label, bins=self.class_bins_rot) for d_label in delta_label[3:]])


        if self.evaluation:
            return img1, img2, label, delta_label
        else:
            if self.classification:
                return img1, img2, label
            else:
                delta_label = torch.tensor(delta_label, dtype= torch.float32)
                return img1, img2, delta_label


