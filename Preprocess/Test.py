import os 
from scipy import io

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


MR = "C:/Users/PHOENIX/Desktop/PseudoCT/Data/Train/MR"
CT = "C:/Users/PHOENIX/Desktop/PseudoCT/Data/Train/CT"


images = os.listdir(MR)
labels = os.listdir(CT)
for i in range(len(images)):

    # Load Data
    image = io.loadmat(os.path.join(MR, images[i]))['MR'].astype('float32')
    label = io.loadmat(os.path.join(CT, labels[i]))['CT'].astype('float32')

    image_max = np.unravel_index(np.argmax(image), image.shape)
    label_max = np.unravel_index(np.argmax(label), label.shape)

    image_min = np.unravel_index(np.argmin(image), image.shape)
    label_min = np.unravel_index(np.argmin(label), label.shape)


    print()
    print(images[i])
    print(image.max(), '\t', image.min())
    print(image_max, '\t', image_min)
    print()
    print(labels[i])
    print(label.max(), '\t', label.min())
    print(label_max, '\t', label_min)
    print()
    print('===========================================================================')