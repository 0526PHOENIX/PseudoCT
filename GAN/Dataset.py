"""
====================================================================================================
Package
====================================================================================================
"""
import os
import random
from scipy import io
from matplotlib import pyplot as plt

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset


"""
====================================================================================================
Training & Validation
====================================================================================================
"""
class Training_2D(Dataset):

    def __init__(self, root = "", is_val = False, val_stride = 10, slice = 7):

        # file path setting
        self.root = root
        self.images_path = os.path.join(self.root, 'MR')
        self.labels_path = os.path.join(self.root, 'CT')

        # load image path into list
        self.images = []
        for series in sorted(os.listdir(self.images_path)):
            self.images.append(os.path.join(self.images_path, series))

        # load label path into list
        self.labels = []
        for series in sorted(os.listdir(self.labels_path)):
            self.labels.append(os.path.join(self.labels_path, series))

        # split training and validation
        if is_val:
            self.images = self.images[::val_stride]
            self.labels = self.labels[::val_stride]
        else:
            del self.images[::val_stride]
            del self.labels[::val_stride]

        # shuffle data
        samples = list(zip(self.images, self.labels))
        random.shuffle(samples)
        (self.images, self.labels) = zip(*samples)

        # check data quantity
        if len(self.images) != len(self.labels):
            raise ValueError('Unequal amount of images and labels.')
        
        # 2D slice parameter
        self.slice = slice                                                  # channels per slice
        self.width = self.slice // 2                                        # width
        self.num_slices = 192 - ((self.width + 1) * 2)                      # slices per series
        self.num_series = len(self.images)                                  # number of series
        
    def __len__(self):
        
        return (self.num_series * self.num_slices)

    def __getitem__(self, index):
        
        # index
        series_index = index // self.num_slices                             # which series
        slices_index = index % self.num_slices + self.width + 1             # which slice

        # get image data
        image = io.loadmat(self.images[series_index])['MR'].astype('float32')
        image = torch.from_numpy(image).to(torch.float32)
        image = image[slices_index - self.width : slices_index + self.width + 1, :, :]
        
        # get label data
        label = io.loadmat(self.labels[series_index])['CT'].astype('float32')
        label = torch.from_numpy(label).to(torch.float32)
        label = label[slices_index, :, :].unsqueeze(0)

        return (image, label)


"""
====================================================================================================
Testing
====================================================================================================
"""
class Testing_2D(Dataset):

    def __init__(self, root = ""):

        # file path setting
        self.root = root
        self.images_path = os.path.join(self.root, 'MR')
        self.labels_path = os.path.join(self.root, 'CT')

        # load image path into list
        self.images = []
        for series in sorted(os.listdir(self.images_path)):
            self.images.append(os.path.join(self.images_path, series))

        # load label path into list
        self.labels = []
        for series in sorted(os.listdir(self.labels_path)):
            self.labels.append(os.path.join(self.labels_path, series))

        # shuffle data
        samples = list(zip(self.images, self.labels))
        random.shuffle(samples)
        (self.images, self.labels) = zip(*samples)

        # check data quantity
        if len(self.images) != len(self.labels):
            raise ValueError('Unequal amount of images and labels.')
        
        # 2D slice parameter
        self.slice = slice                                                  # channels per slice
        self.width = self.slice // 2                                        # width
        self.num_slices = 262 - ((self.width + 1) * 2)                      # slices per series
        self.num_series = len(self.images)                                  # number of series

    def __len__(self):

        return (self.num_series * self.num_slices)

    def __getitem__(self, index):

        # index
        series_index = index // self.num_slices                             # which series
        slices_index = index % self.num_slices + self.width + 1             # which slice

        # get image data
        image = io.loadmat(self.images[series_index])['MR'].astype('float32')
        image = torch.from_numpy(image).to(torch.float32)
        image = image[slices_index - self.width : slices_index + self.width + 1, :, :]
        
        # get label data
        label = io.loadmat(self.labels[series_index])['CT'].astype('float32')
        label = torch.from_numpy(label).to(torch.float32)
        label = label[slices_index, :, :].unsqueeze(0)

        return (image, label)
    

"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':

    filepath = "C:\\Users\\PHOENIX\\Desktop\\PseudoCT\\Data\\Train"

    train_2D = Training_2D(filepath, False, 10, 7)
    print()
    print(len(train_2D))
    image, label = train_2D[155]
    print(type(image), image.device, image.shape)
    print(type(label), label.device, label.shape)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image[3, :, :], cmap = 'gray')
    plt.subplot(1, 2, 2)
    plt.imshow(label[0, :, :], cmap = 'gray')
    plt.show()