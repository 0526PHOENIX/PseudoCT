import os 
from scipy import io
from scipy import ndimage
from scipy.ndimage.morphology import binary_dilation, binary_erosion

import cv2
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


MR = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Raw/Train/MR"

TEMP = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_nii/Train/MR/MR14.nii"


# images = os.listdir(MR)
# for i in range(len(images)):
    
#     image = io.loadmat(os.path.join(MR, images[i]))['MR'].astype('float32')

#     flat = image.flatten()

#     sorted = np.sort(flat)

#     dis = np.cumsum(sorted)
#     dis = dis / dis[-1]

#     index = np.where(dis <= 0.125)[0][-1]
#     value = sorted[index]

#     print(images[i], value)
#     print()


temp = nib.load(TEMP).get_fdata()

print(temp.min(), temp.max())

# mask = temp != -1000
# temp = np.where(mask, 1, 0)

# plt.imshow(temp, cmap = 'gray')
# plt.show()

# print(temp.sum() / (temp.shape[0] * temp.shape[1]))
