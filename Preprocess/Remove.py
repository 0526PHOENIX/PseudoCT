import os 
from scipy import io

import numpy as np
import nibabel as nib

import torch
import torch.nn.functional as F


MR_RAW = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Raw/Train/MR"
CT_RAW = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Raw/Train/CT"

MR = "C:/Users/PHOENIX/Desktop/PseudoCT/Data/Train/MR"
CT = "C:/Users/PHOENIX/Desktop/PseudoCT/Data/Train/CT"

BG = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Nii/Train/BG"

MR_NII = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Nii/Train/MR"
CT_NII = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Nii/Train/CT"


images = os.listdir(MR_RAW)
labels = os.listdir(CT_RAW)
backgrounds = os.listdir(BG)
for i in range(len(images)):

    # Load Data and Background
    image = io.loadmat(os.path.join(MR_RAW, images[i]))['MR'].astype('float32')
    label = io.loadmat(os.path.join(CT_RAW, labels[i]))['CT'].astype('float32')

    background = nib.load(os.path.join(BG, backgrounds[i])).get_fdata()

    # Convert to Troch Tensor
    image = torch.from_numpy(image).to(torch.float32)
    label = torch.from_numpy(label).to(torch.float32)

    # Trilinear Interpolation: (192, 192, 192)
    image = F.interpolate(image[None, None, ...], size = (192, 192, 192), mode = 'trilinear')[0, 0, ...]
    label = F.interpolate(label[None, None, ...], size = (192, 192, 192), mode = 'trilinear')[0, 0, ...]

    # Convert to Numpy Array + Transpose: (C * H * W)
    image = image.numpy().transpose(2, 0, 1)
    label = label.numpy().transpose(2, 0, 1)

    # Prepare Mask
    mask = (background != 0)

    # Apply Mask
    image = np.where(mask, image, 0)
    label = np.where(mask, label, -1000)

    # Save Matlab Data
    io.savemat(os.path.join(MR, images[i]), {'MR': image})
    io.savemat(os.path.join(CT, labels[i]), {'CT': label})

    # Save Nifti Data
    image = nib.Nifti1Image(image, np.eye(4))
    nib.save(image, os.path.join(MR_NII, images[i].strip('.mat') + '.nii'))

    label = nib.Nifti1Image(label, np.eye(4))
    nib.save(label, os.path.join(CT_NII, labels[i].strip('.mat') + '.nii'))

    # Check Progress
    print()
    print(i + 1, 'Done')
    print()
    print('===========================================================================')