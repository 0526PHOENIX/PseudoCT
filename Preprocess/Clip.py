import os 
from scipy import io

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


MR = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Nii/Train/MR"
CT = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Nii/Train/CT"
BG = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Nii/Train/BG"


images = os.listdir(MR)
labels = os.listdir(CT)
backgrounds = os.listdir(BG)
for i in range(len(images)):
    
    # image = io.loadmat(os.path.join(MR, images[i]))['MR'].astype('float32')
    # label = io.loadmat(os.path.join(CT, labels[i]))['CT'].astype('float32')

    image = nib.load(os.path.join(MR, images[i])).get_fdata().astype('float32')
    label = nib.load(os.path.join(CT, labels[i])).get_fdata().astype('float32')

    background = nib.load(os.path.join(BG, backgrounds[i])).get_fdata()

    upper = -1
    lower = -1
    for j in range(background.shape[0]):

        mask = background[j, :, :]

        ratio = mask.sum() / (mask.shape[0] * mask.shape[1])

        if (ratio > 0.05) and (upper == -1):

            upper = j

        if (ratio < 0.05) and (upper != -1) and (lower == -1):

            lower = j

    image = image[upper: lower, :, :]
    label = label[upper: lower, :, :]

    # Save Nifti Data
    image = nib.Nifti1Image(image, np.eye(4))
    nib.save(image, os.path.join(MR, images[i]))

    label = nib.Nifti1Image(label, np.eye(4))
    nib.save(label, os.path.join(CT, labels[i]))

    """
    Check Progress
    """
    print()
    print(i + 1, 'Done')
    print(upper, lower)
    print()
    print('===========================================================================')