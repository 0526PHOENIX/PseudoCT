import os 
from scipy import io

import numpy as np
import nibabel as nib

import torch
import torch.nn.functional as F


MR_RAW = "C:\\Users\\PHOENIX\\Desktop\\PseudoCT\\Data_Raw\\Train\\MR"
CT_RAW = "C:\\Users\\PHOENIX\\Desktop\\PseudoCT\\Data_Raw\\Train\\CT"

MR = "C:\\Users\\PHOENIX\\Desktop\\PseudoCT\\Data\\Train\\MR"
CT = "C:\\Users\\PHOENIX\\Desktop\\PseudoCT\\Data\\Train\\CT"

MR_NII = "C:\\Users\\PHOENIX\\Desktop\\PseudoCT\\Data_Nifti\\Train\\MR"
CT_NII = "C:\\Users\\PHOENIX\\Desktop\\PseudoCT\\Data_Nifti\\Train\\CT"


images = os.listdir(MR_RAW)
labels = os.listdir(CT_RAW)
for i in range(len(images)):

    """
    MR 
    """
    # load MR data
    image = io.loadmat(os.path.join(MR_RAW, images[i]))['MR'].astype('float32')
    image = torch.from_numpy(image).to(torch.float32)

    # interpolate to (192, 192, 192)
    image = F.interpolate(image[None, None, ...], size = (192, 192, 192), mode = 'trilinear')[0, 0, ...]

    # save MR data 
    image = image.numpy()

    image_nii = nib.Nifti1Image(image, np.eye(4))
    nib.save(image_nii, os.path.join(MR_NII, images[i].strip('.mat') + '.nii'))                     # Nifti file

    image = image.transpose(2, 0, 1)
    io.savemat(os.path.join(MR, images[i]), {'MR': image})                                          # Matlab file

    """
    CT
    """
    # load CT data
    label = io.loadmat(os.path.join(CT_RAW, labels[i]))['CT'].astype('float32')
    label = torch.from_numpy(label).to(torch.float32)

    # interpolate to (192, 192, 192)
    label = F.interpolate(label[None, None, ...], size = (192, 192, 192), mode = 'trilinear')[0, 0, ...]

    # save CT data
    label = label.numpy()

    label_nii = nib.Nifti1Image(label, np.eye(4))
    nib.save(label_nii, os.path.join(CT_NII, labels[i].strip('.mat') + '.nii'))                     # Nifti file

    label = label.transpose(2, 0, 1)
    io.savemat(os.path.join(CT, labels[i]), {'CT': label})                                          # Matlab file

    """
    Check Progress
    """
    print(i + 1)
    print('MR.mat shape:', image.shape)
    print('CT.mat shape:', image.shape)