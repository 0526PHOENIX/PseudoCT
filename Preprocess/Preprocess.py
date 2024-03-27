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
Global Constant
====================================================================================================
"""
MR_RAW = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Raw/Train/MR"
CT_RAW = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Raw/Train/CT"

MR = "C:/Users/PHOENIX/Desktop/PseudoCT/Data/Train/MR"
CT = "C:/Users/PHOENIX/Desktop/PseudoCT/Data/Train/CT"

MR_NII = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Nii/Train/MR"
CT_NII = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Nii/Train/CT"
TG_NII = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Nii/Train/TG"

MR_2D = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_2D/Train/MR"
CT_2D = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_2D/Train/CT"
TG_2D = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_2D/Train/TG"

MR_CHECK = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Check/Train/MR"
CT_CHECK = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_Check/Train/CT"

PATH_LIST = [MR, CT, MR_2D, CT_2D, TG_2D, MR_NII, CT_NII, TG_NII, MR_CHECK, CT_CHECK]

TRAIN = True


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

        # Check the Path
        for path in PATH_LIST:
            if not os.path.exists(path):
                os.makedirs(path)

        # Get File Name
        self.images = os.listdir(MR_RAW)
        self.labels = os.listdir(CT_RAW)

        self.target = os.listdir(TG_NII)

        # Check File Number
        if len(self.images) != len(self.labels):
            raise ValueError('\n', 'Unequal Amount of images and labels.', '\n')
        
        self.len = len(self.images)

        # Problem Case
        self.threshold = [4]
        self.direction = []

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
            print(self.images[i], self.labels[i])
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
            image = torch.from_numpy(image)
            label = torch.from_numpy(label)

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
            image = torch.from_numpy(image)
            label = torch.from_numpy(label)

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

        # Deal with MR14
        if TRAIN:

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
    Remove Blank Slice
    ================================================================================================
    """
    def slice(self, width = 3):

        for i in range(self.len):

            # Load Data and Backgrond
            image = io.loadmat(os.path.join(MR, self.images[i]))['MR'].astype('float32')
            label = io.loadmat(os.path.join(CT, self.labels[i]))['CT'].astype('float32')

            masks = nib.load(os.path.join(TG_NII, self.target[i])).get_fdata()

            # Find Blank Slice Index
            lower = -1
            upper = -1
            for j in range(masks.shape[0]):

                mask = masks[j, :, :]

                ratio = mask.sum() / (mask.shape[0] * mask.shape[1])

                if (ratio > 0.075) and (lower == -1):

                    lower = j

                if (ratio < 0.075) and (lower != -1) and (upper == -1):

                    upper = j

            # Slicing
            for j in range(lower + width, upper - width):

                mr = image[j - width : j + width + 1, :, :]
                ct = label[j : j + 1, :, :]
                tg = masks[j : j + 1, :, :]

                io.savemat(os.path.join(MR_2D, self.images[i].strip('.mat') + '_' + str(j) + '.mat'), {'MR': mr})
                io.savemat(os.path.join(CT_2D, self.labels[i].strip('.mat') + '_' + str(j) + '.mat'), {'CT': ct})
                
                if i < 9:
                    io.savemat(os.path.join(TG_2D, 'TG0' + str(i + 1) + '_' + str(j) + '.mat'), {'TG': tg})
                else:
                    io.savemat(os.path.join(TG_2D, 'TG' + str(i + 1) + '_' + str(j) + '.mat'), {'TG': tg})

            # Check Progress
            print()
            print(i + 1, 'Done')
            print(lower, upper)
            print()
            print('===============================================================================')

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
    ================================================================================================
    Visualize Raw Data
    ================================================================================================
    """
    def visualize(self):

        for i in range(self.len):

            # Load Matlab Data
            image = io.loadmat(os.path.join(MR_RAW, self.images[i]))['MR'].astype('float32')
            label = io.loadmat(os.path.join(CT_RAW, self.labels[i]))['CT'].astype('float32')

            # Save Nifti Data
            image = nib.Nifti1Image(image, np.eye(4))
            nib.save(image, os.path.join(MR_CHECK, self.images[i].strip('.mat') + '.nii'))

            label = nib.Nifti1Image(label, np.eye(4))
            nib.save(label, os.path.join(CT_CHECK, self.labels[i].strip('.mat') + '.nii'))

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

    # pre.visualize()

    # pre.main()

    # pre.mat2nii()
    # pre.nii2mat()

    # pre.clip()

    pre.slice()

    # pre.check()