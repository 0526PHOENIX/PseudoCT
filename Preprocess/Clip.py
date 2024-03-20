import os 
from scipy import io
from scipy import ndimage
from scipy.ndimage.morphology import binary_dilation, binary_erosion

import numpy as np
import nibabel as nib


MR = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Nii/Train/CT"


images = os.listdir(MR)
for i in range(len(images)):

    image = nib.load(os.path.join(MR, images[i])).get_fdata().astype('float32')

    flat = image.flatten()

    sorted = np.sort(flat)

    dis = np.cumsum(sorted)
    dis = dis / dis[-1]

    index = np.where(dis <= 0.995)[0][-1]
    value = sorted[index]

    print(value)