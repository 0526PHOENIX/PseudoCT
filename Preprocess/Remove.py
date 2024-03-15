import os 
from scipy import io
from scipy import ndimage
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation, binary_erosion

import numpy as np
import nibabel as nib


MR = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Raw/Train/MR"
CT = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Raw/Train/CT"

MR_TEMP = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Temp/Train/MR"
CT_TEMP = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Temp/Train/CT"
TG_TEMP = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Temp/Train/TG"


images = os.listdir(MR)
labels = os.listdir(CT)

for i in range(len(images)):

    """
    Load MR & CT
    """
    image = io.loadmat(os.path.join(MR, images[i]))['MR'].astype('float32')
    label = io.loadmat(os.path.join(CT, labels[i]))['CT'].astype('float32')

    """
    Prepare Binary Mask
    """
    # Thresholding
    binary = image > 100

    # Get Connective Component
    component, feature = ndimage.label(binary)

    # Compute Size of Each Component
    size = ndimage.sum(binary, component, range(1, feature + 1))

    # Find Largest Component
    largest_component = np.argmax(size) + 1

    mask = component == largest_component

    # Fill Holes in Mask
    mask = binary_fill_holes(mask)
    
    mask = binary_dilation(mask, np.ones((25, 25, 25)))
    mask = binary_erosion(mask, np.ones((25, 25, 25)))

    """
    Apply Mask 
    """
    image = np.where(mask, image, 0)

    label = np.where(mask, label, -1000)

    target = np.where(mask, 1, 0)


    """
    Save MR & CT
    """
    image = nib.Nifti1Image(image, np.eye(4))
    nib.save(image, os.path.join(MR_TEMP, images[i].strip('.mat') + '.nii'))

    label = nib.Nifti1Image(label, np.eye(4))
    nib.save(label, os.path.join(CT_TEMP, labels[i].strip('.mat') + '.nii'))

    target = nib.Nifti1Image(target, np.eye(4))
    nib.save(target, os.path.join(TG_TEMP, str(i) + '.nii'))

    """
    Check Progress
    """
    print(images[i], 'Done')
    print(labels[i], 'Done')
    print()
