import os 
from scipy import io
from scipy import ndimage
from scipy.ndimage.morphology import binary_dilation, binary_erosion

import numpy as np
import nibabel as nib

import torch
import torch.nn.functional as F


MR_RAW = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Raw/Train/MR"
CT_RAW = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Raw/Train/CT"

MR = "C:/Users/PHOENIX/Desktop/PseudoCT/Data/Train/MR"
CT = "C:/Users/PHOENIX/Desktop/PseudoCT/Data/Train/CT"

MR_NII = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Nii/Train/MR"
CT_NII = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Nii/Train/CT"


images = os.listdir(MR_RAW)
labels = os.listdir(CT_RAW)
for i in range(len(images)):

    """
    Begining
    """
    print()
    print(i + 1, 'Begin')
    print()

    """
    Load MR & CT
    """
    image = io.loadmat(os.path.join(MR_RAW, images[i]))['MR'].astype('float32')
    label = io.loadmat(os.path.join(CT_RAW, labels[i]))['CT'].astype('float32')

    """
    Find Threshold
    """
    # Flatten MR Data
    flat = image.flatten()

    # Ascending Order
    sorted = np.sort(flat)

    # Cumulative Distribution
    dis = np.cumsum(sorted)
    dis = dis / dis[-1]

    # Find Threshold
    if (i + 1) in [5, 7, 8, 9, 10]:

        # Specific Case
        index = np.where(dis <= 0.200)[0][-1]
        value = sorted[index]

        print('Threshold(20.0%):', value)

    else:

        # General Case
        index = np.where(dis <= 0.125)[0][-1]
        value = sorted[index]

        print('Threshold(12.5%):', value)

    """
    Prepare Binary Mask
    """
    # Thresholding
    binary = (image > value)

    # Get Connective Component
    component, feature = ndimage.label(binary)

    # Compute Size of Each Component
    size = ndimage.sum(binary, component, range(1, feature + 1))

    # Find Largest Component
    largest_component = np.argmax(size) + 1

    mask = (component == largest_component)

    # Fill Holes in Mask
    mask = binary_dilation(mask, np.ones((25, 25, 25)))
    mask = binary_erosion(mask, np.ones((25, 25, 25)))

    """
    Apply Mask 
    """
    # MR Air: 0
    image = np.where(mask, image, 0)
    # CT Air: -1000
    label = np.where(mask, label, -1000)

    """
    Interpolation
    """
    # Convert to Troch Tensor
    image = torch.from_numpy(image).to(torch.float32)
    label = torch.from_numpy(label).to(torch.float32)

    # Trilinear Interpolation: (192, 192, 192)
    image = F.interpolate(image[None, None, ...], size = (192, 192, 192), mode = 'trilinear')[0, 0, ...]
    label = F.interpolate(label[None, None, ...], size = (192, 192, 192), mode = 'trilinear')[0, 0, ...]

    """
    Save MR & CT
    """
    # Convert to Numpy Array
    image = image.numpy()
    label = label.numpy()

    # Save Nifti Data
    image_nii = nib.Nifti1Image(image, np.eye(4))
    nib.save(image_nii, os.path.join(MR_NII, images[i].strip('.mat') + '.nii'))

    label_nii = nib.Nifti1Image(label, np.eye(4))
    nib.save(label_nii, os.path.join(CT_NII, labels[i].strip('.mat') + '.nii'))

    # Save Matlab Data + Transpose: (C * H * W)
    image = image.transpose(2, 0, 1)
    io.savemat(os.path.join(MR, images[i]), {'MR': image})

    label = label.transpose(2, 0, 1)
    io.savemat(os.path.join(CT, labels[i]), {'CT': label})

    """
    Check Progress
    """
    print()
    print(i + 1, 'Done')
    print()
    print('===========================================================================')