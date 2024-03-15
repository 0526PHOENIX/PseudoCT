import os 
from scipy import io
from scipy import ndimage
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation, binary_erosion

import numpy as np
import nibabel as nib


MR = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Raw/Train/MR/MR06.mat"
CT = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Raw/Train/CT/CT06.mat"
MR_TEMP = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Temp/Train/MR/MR01.temp.nii"
CT_TEMP = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Temp/Train/CT/CT01.temp.nii"

BINARY = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Temp/Train/binary.nii"
RESULT = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Temp/Train/mask.nii"
TEST = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Temp/Train/test.nii"


"""
Load MR & CT
"""
image = io.loadmat(MR)['MR'].astype('float32')
label = io.loadmat(CT)['CT'].astype('float32')

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
# mask = binary_fill_holes(mask)

# mask = binary_dilation(mask, np.ones((25, 25, 25)))
# mask = binary_erosion(mask, np.ones((25, 25, 25)))

"""
Apply Mask 
"""
image = np.where(mask, image, 0)

label = np.where(mask, label, -1000)

result = np.where(mask, 1, 0)

"""
Save MR & CT
"""
# image = nib.Nifti1Image(image, np.eye(4))
# nib.save(image, MR_TEMP)

# label = nib.Nifti1Image(label, np.eye(4))
# nib.save(CT_TEMP, CT_TEMP)

# binary = nib.Nifti1Image(binary.astype(int), np.eye(4))
# nib.save(binary, BINARY)

# result = nib.Nifti1Image(result, np.eye(4))
# nib.save(result, RESULT)