"""
====================================================================================================
Package
====================================================================================================
"""
import os 
from scipy import io
from scipy import ndimage
from scipy.ndimage.morphology import binary_dilation, binary_erosion

import numpy as np
import nibabel as nib

import torch
import torch.nn.functional as F


"""
====================================================================================================
Data Path
====================================================================================================
"""
MR_RAW = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Raw/Train/MR"
CT_RAW = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Raw/Train/CT"

MR = "C:/Users/PHOENIX/Desktop/PseudoCT/Data/Train/MR"
CT = "C:/Users/PHOENIX/Desktop/PseudoCT/Data/Train/CT"

MR_NII = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Nii/Train/MR"
CT_NII = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Nii/Train/CT"
TG_NII = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Nii/Train/TG"


"""
====================================================================================================
Preprocess
====================================================================================================
"""
class Preprocess():

    """
    ================================================================================================
    Critical Parameters
    ================================================================================================
    """
    def __init__(self):
        
        self.images = os.listdir(MR_RAW)
        self.labels = os.listdir(CT_RAW)

        self.target = os.listdir(TG_NII)

        if len(self.images) != len(self.labels):
            raise ValueError('\n', 'Unequal Amount of images and labels.', '\n')
        
        self.len = len(self.images)

        self.threshold = [5, 7, 8, 9, 10]
        self.direction = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    """
    ================================================================================================
    Main Process: Interpolation + Transpose + Remove Background + Rotate
    ================================================================================================
    """
    def main(self):

        for i in range(self.len):

            """
            Begining
            """
            print()
            print(i + 1, 'Begin')
            print()

            """
            Load MR & CT
            """
            image = io.loadmat(os.path.join(MR_RAW, self.images[i]))['MR'].astype('float32')
            label = io.loadmat(os.path.join(CT_RAW, self.labels[i]))['CT'].astype('float32')

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
            Transpose
            """
            # Convert to Numpy Array + Transpose: (C * H * W)
            image = image.numpy().transpose(2, 0, 1)
            label = label.numpy().transpose(2, 0, 1)

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
            if (i + 1) in self.threshold:

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
            # Background
            mask = np.where(mask, 1, 0)

            """
            Rotate: 11~20
            """
            if (i + 1) in self.direction:

                image = np.rot90(image, k = 1, axes = (1, 2))
                label = np.rot90(label, k = 1, axes = (1, 2))
                mask = np.rot90(mask, k = 1, axes = (1, 2))

            """
            Save Data
            """
            # Save Matlab Data
            io.savemat(os.path.join(MR, self.images[i]), {'MR': image})
            io.savemat(os.path.join(CT, self.labels[i]), {'CT': label})

            # Save Nifti Data
            image = nib.Nifti1Image(image, np.eye(4))
            nib.save(image, os.path.join(MR_NII, self.images[i].strip('.mat') + '.nii'))

            label = nib.Nifti1Image(label, np.eye(4))
            nib.save(label, os.path.join(CT_NII, self.labels[i].strip('.mat') + '.nii'))

            mask = nib.Nifti1Image(mask, np.eye(4))
            if i < 9:
                nib.save(mask, os.path.join(TG_NII, 'TG0' + str(i + 1) + '.nii'))
            else:
                nib.save(mask, os.path.join(TG_NII, 'TG' + str(i + 1) + '.nii'))

            """
            Check Progress
            """
            print()
            print(i + 1, 'Done')
            print()
            print('===============================================================================')

    """
    ================================================================================================
    Remove Background
    ================================================================================================
    """
    def remove_background(self):

        for i in range(self.len):

            """
            Load MR & CT
            """
            image = io.loadmat(os.path.join(MR_RAW, self.images[i]))['MR'].astype('float32')
            label = io.loadmat(os.path.join(CT_RAW, self.labels[i]))['CT'].astype('float32')

            mask = nib.load(os.path.join(TG_NII, self.target[i])).get_fdata()
            mask = (mask != 0)

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
            Transpose
            """
            # Convert to Numpy Array + Transpose: (C * H * W)
            image = image.numpy().transpose(2, 0, 1)
            label = label.numpy().transpose(2, 0, 1)

            """
            Rotate: 11~20
            """
            if (i > 9) and (i < 20):

                image = np.rot90(image, k = 1, axes = (1, 2))
                label = np.rot90(label, k = 1, axes = (1, 2))

            """
            Apply Mask 
            """
            # MR Air: 0
            image = np.where(mask, image, 0)
            # CT Air: -1000
            label = np.where(mask, label, -1000)

            """
            Save Data
            """
            # Save Matlab Data
            io.savemat(os.path.join(MR, self.images[i]), {'MR': image})
            io.savemat(os.path.join(CT, self.labels[i]), {'CT': label})

            # Save Nifti Data
            image = nib.Nifti1Image(image, np.eye(4))
            nib.save(image, os.path.join(MR_NII, self.images[i].strip('.mat') + '.nii'))

            label = nib.Nifti1Image(label, np.eye(4))
            nib.save(label, os.path.join(CT_NII, self.labels[i].strip('.mat') + '.nii'))

            """
            Check Progress
            """
            print()
            print(i + 1, 'Done')
            print()
            print('===============================================================================')


    """
    ================================================================================================
    Remove Blank Slice
    ================================================================================================
    """
    def remove_blank(self):

        for i in range(self.len):

            # Load Data and Backgrond
            image = io.loadmat(os.path.join(MR, self.images[i]))['MR'].astype('float32')
            label = io.loadmat(os.path.join(CT, self.labels[i]))['CT'].astype('float32')

            masks = nib.load(os.path.join(TG_NII, self.target[i])).get_fdata()

            # Find Blank Slice Index
            upper = -1
            lower = -1
            for j in range(masks.shape[0]):

                mask = masks[j, :, :]

                ratio = mask.sum() / (mask.shape[0] * mask.shape[1])

                if (ratio > 0.075) and (upper == -1):

                    upper = j

                if (ratio < 0.075) and (upper != -1) and (lower == -1):

                    lower = j

            # Remove Blank Slice
            image = image[upper: lower, :, :]
            label = label[upper: lower, :, :]

            # Save Matlab Data
            io.savemat(os.path.join(MR, self.images[i]), {'MR': image})
            io.savemat(os.path.join(CT, self.labels[i]), {'CT': label})

            # Save Nifti Data
            image = nib.Nifti1Image(image, np.eye(4))
            nib.save(image, os.path.join(MR_NII, self.images[i].strip('.mat') + '.nii'))

            label = nib.Nifti1Image(label, np.eye(4))
            nib.save(label, os.path.join(CT_NII, self.labels[i].strip('.mat') + '.nii'))

            # Check Progress
            print()
            print(i + 1, 'Done')
            print(upper, lower)
            print()
            print('===============================================================================')

    """
    ================================================================================================
    Clip Intensity
    ================================================================================================
    """
    def clip(self):
    
        sum = 0
        for i in range(self.len):

            # Load Data
            image = io.loadmat(os.path.join(MR, self.images[i]))['MR'].astype('float32')
            label = io.loadmat(os.path.join(CT, self.labels[i]))['CT'].astype('float32')
            
            # Summarize MR Max Value
            if i != 13:
                sum += image.max()

            # Clip CT Intensity
            label = np.clip(label, -1000, 3000)

            # Save Matlab CT Data
            io.savemat(os.path.join(CT, self.labels[i]), {'CT': label})

            # Save Nifti CT Data
            label = nib.Nifti1Image(label, np.eye(4))
            nib.save(label, os.path.join(CT_NII, self.labels[i].strip('.mat') + '.nii'))

            # Check Progress
            print()
            print(i + 1, 'Done')
            print()
            print('===============================================================================')

        # Load MR14
        image = io.loadmat(os.path.join(MR, self.images[13]))['MR'].astype('float32')

        # Clip
        image = np.clip(image, 0, (sum / 21))

        # Check
        print()
        print(self.images[13])
        print(image.max(), '\t', image.min())
        print()

        # Save Matlab MR14 Data
        io.savemat(os.path.join(MR, self.images[13]), {'MR': image})

        # Save Nifti MR14 Data
        image = nib.Nifti1Image(image, np.eye(4))
        nib.save(image, os.path.join(MR_NII, self.images[13].strip('.mat') + '.nii'))

    """
    ================================================================================================
    Check Statistics
    ================================================================================================
    """
    def check(self):

        for i in range(self.len):

            image = io.loadmat(os.path.join(MR, self.images[i]))['MR'].astype('float32')
            label = io.loadmat(os.path.join(CT, self.labels[i]))['CT'].astype('float32')

            image_max = str(np.unravel_index(np.argmax(image), image.shape))
            label_max = str(np.unravel_index(np.argmax(label), label.shape))

            image_min = str(np.unravel_index(np.argmin(image), image.shape))
            label_min = str(np.unravel_index(np.argmin(label), label.shape))

            space = "{: <15.2f}\t{: <15.2f}"
            print()
            print(self.images[i], image.shape)
            print(space.format(image.max(), image.min()))
            # print(space.format(image_max, image_min))
            # print(space.format(image.mean(), image.std()))
            print()
            print(self.labels[i], label.shape)
            print(space.format(label.max(), label.min()))
            # print(space.format(label_max, label_min))
            # print(space.format(label.mean(), label.std()))
            print()
            print('===========================================================================')

    """
    ================================================================================================
    Convert .mat to .nii
    ================================================================================================
    """
    def mat2nii(self):

        for i in range(self.len):

            # Load Matlab Data
            image = io.loadmat(os.path.join(MR, self.images[i]))['MR'].astype('float32')
            label = io.loadmat(os.path.join(CT, self.labels[i]))['CT'].astype('float32')

            # Save Nifti Data
            image = nib.Nifti1Image(image, np.eye(4))
            nib.save(image, os.path.join(MR_NII, self.images[i].strip('.mat') + '.nii'))

            label = nib.Nifti1Image(label, np.eye(4))
            nib.save(label, os.path.join(CT_NII, self.labels[i].strip('.mat') + '.nii'))

            # Check Progress
            print()
            print(i + 1, 'Done')
            print()
            print('===============================================================================')
    
    """
    ================================================================================================
    Convert .nii to .mat
    ================================================================================================
    """
    def nii2mat(self):

        for i in range(self.len):

            # Load Nifti Data
            image = nib.load(os.path.join(MR_NII, self.images[i])).get_fdata().astype('float32')
            label = nib.load(os.path.join(CT_NII, self.labels[i])).get_fdata().astype('float32')

            # Save Matlab Data
            io.savemat(os.path.join(MR, self.images[i].strip('.nii') + '.mat'), {'MR': image})
            io.savemat(os.path.join(CT, self.labels[i].strip('.nii') + '.mat'), {'CT': label})

            # Check Progress
            print()
            print(i + 1, 'Done')
            print()
            print('===============================================================================')


"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':
    
    pre = Preprocess()

    # pre.main()

    # pre.mat2nii()
    # pre.nii2mat()

    # pre.remove_blank()

    # pre.clip()

    pre.check()