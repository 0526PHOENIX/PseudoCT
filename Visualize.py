import os 
from scipy import io

import numpy as np
import nibabel as nib


MR_RAW = "C:\\Users\\PHOENIX\\Desktop\\PseudoCT\\Data_Raw\\Train\\MR"
CT_RAW = "C:\\Users\\PHOENIX\\Desktop\\PseudoCT\\Data_Raw\\Train\\CT"

MR = "C:\\Users\\PHOENIX\\Desktop\\PseudoCT\\Data_Check\\Train\\MR"
CT = "C:\\Users\\PHOENIX\\Desktop\\PseudoCT\\Data_Check\\Train\\CT"


images = os.listdir(MR_RAW)
labels = os.listdir(CT_RAW)
for i in range(len(images)):

    """
    MR 
    """
    # load MR data
    image = io.loadmat(os.path.join(MR_RAW, images[i]))['MR'].astype('float32')

    image = nib.Nifti1Image(image, np.eye(4))
    nib.save(image, os.path.join(MR, images[i].strip('.mat') + '.nii'))                     # Nifti file

    """
    CT
    """
    # load CT data
    label = io.loadmat(os.path.join(CT_RAW, labels[i]))['CT'].astype('float32')

    label = nib.Nifti1Image(label, np.eye(4))
    nib.save(label, os.path.join(CT, labels[i].strip('.mat') + '.nii'))                     # Nifti file

    """
    Check Progress
    """
    print(i + 1)
    print('MR.mat shape:', image.shape)
    print('CT.mat shape:', image.shape)