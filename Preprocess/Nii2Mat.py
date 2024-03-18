import os 
from scipy import io

import numpy as np
import nibabel as nib


MR = "C:/Users/PHOENIX/Desktop/PseudoCT/Data/Train/MR"
CT = "C:/Users/PHOENIX/Desktop/PseudoCT/Data/Train/CT"

MR_NII = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Nii/Train/MR"
CT_NII = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Nii/Train/CT"


images = os.listdir(MR_NII)
labels = os.listdir(CT_NII)
for i in range(len(images)):

    image = nib.load(os.path.join(MR_NII, images[i])).get_fdata().astype('float32')
    label = nib.load(os.path.join(CT_NII, labels[i])).get_fdata().astype('float32')

    io.savemat(os.path.join(MR, images[i].strip('.nii') + '.mat'), {'MR': image})
    io.savemat(os.path.join(CT, labels[i].strip('.nii') + '.mat'), {'CT': label})