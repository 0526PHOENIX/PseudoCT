import os 
from scipy import io

import numpy as np
import nibabel as nib


MR = "C:/Users/PHOENIX/Desktop/PseudoCT/Data/Train/MR"
CT = "C:/Users/PHOENIX/Desktop/PseudoCT/Data/Train/CT"

MR_NII = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Nii/Train/MR"
CT_NII = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Nii/Train/CT"


images = os.listdir(MR)
labels = os.listdir(CT)
for i in range(len(images)):

    image = io.loadmat(os.path.join(MR, images[i]))['MR'].astype('float32')
    label = io.loadmat(os.path.join(CT, labels[i]))['CT'].astype('float32')

    image = nib.Nifti1Image(image, np.eye(4))
    nib.save(image, os.path.join(MR_NII, images[i].strip('.mat') + '.nii'))

    label = nib.Nifti1Image(label, np.eye(4))
    nib.save(label, os.path.join(CT_NII, labels[i].strip('.mat') + '.nii'))