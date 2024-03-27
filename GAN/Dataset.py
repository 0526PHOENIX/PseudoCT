"""
====================================================================================================
Package
====================================================================================================
"""
import os
import random
import numpy as np
from scipy import io
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset


"""
====================================================================================================
Training & Validation
====================================================================================================
"""
class Training_2D(Dataset):

    def __init__(self, root = "", is_val = False, val_stride = 10):

        # Filepath
        self.root = root
        self.images_path = os.path.join(self.root, 'MR')
        self.labels_path = os.path.join(self.root, 'CT')
        self.masks_path = os.path.join(self.root, 'TG')

        # MR File List
        self.images = []
        for series in sorted(os.listdir(self.images_path)):
            self.images.append(os.path.join(self.images_path, series))

        # CT File List
        self.labels = []
        for series in sorted(os.listdir(self.labels_path)):
            self.labels.append(os.path.join(self.labels_path, series))

        # TG File List
        self.masks = []
        for series in sorted(os.listdir(self.masks_path)):
            self.masks.append(os.path.join(self.masks_path, series))

        # Split Training and Validation Dataset
        if is_val:
            self.images = self.images[::val_stride]
            self.labels = self.labels[::val_stride]
            self.masks = self.masks[::val_stride]
        else:
            del self.images[::val_stride]
            del self.labels[::val_stride]
            del self.masks[::val_stride]

        # Check Data Quantity
        if len(self.images) != len(self.labels):
            raise ValueError('Unequal Amount of Images and Labels.')
        
    def __len__(self):
        
        return len(self.images)

    def __getitem__(self, index):

        # Load MR Data: (7, 192, 192)
        image = io.loadmat(self.images[index])['MR'].astype('float32')
        image = torch.from_numpy(image)
        
        # Load CT Data: (1, 192, 192)
        label = io.loadmat(self.labels[index])['CT'].astype('float32')
        label = torch.from_numpy(label)

        # Load TG Data: (1, 192, 192)
        mask = io.loadmat(self.masks[index])['TG'].astype('bool')
        mask = torch.from_numpy(mask)

        return (image, label, mask)
    
"""
====================================================================================================
Testing
====================================================================================================
"""
class Testing_2D(Dataset):

    def __init__(self, root = ""):

        # Filepath
        self.root = root
        self.images_path = os.path.join(self.root, 'MR')
        self.labels_path = os.path.join(self.root, 'CT')
        self.masks_path = os.path.join(self.root, 'TG')

        # MR File List
        self.images = []
        for series in sorted(os.listdir(self.images_path)):
            self.images.append(os.path.join(self.images_path, series))

        # CT File List
        self.labels = []
        for series in sorted(os.listdir(self.labels_path)):
            self.labels.append(os.path.join(self.labels_path, series))

        # TG File List
        self.masks = []
        for series in sorted(os.listdir(self.masks_path)):
            self.masks.append(os.path.join(self.masks_path, series))

        # Check Data Quantity
        if len(self.images) != len(self.labels):
            raise ValueError('Unequal Amount of Images and Labels.')
        
    def __len__(self):
        
        return len(self.images)

    def __getitem__(self, index):

        # Load MR Data: (7, 192, 192)
        image = io.loadmat(self.images[index])['MR'].astype('float32')
        image = torch.from_numpy(image)
        
        # Load CT Data: (1, 192, 192)
        label = io.loadmat(self.labels[index])['CT'].astype('float32')
        label = torch.from_numpy(label)

        # Load TG Data: (1, 192, 192)
        mask = io.loadmat(self.masks[index])['TG'].astype('bool')
        mask = torch.from_numpy(mask)

        return (image, label, mask)

"""
====================================================================================================
Training & Validation
====================================================================================================
"""
class Training_3D(Dataset):

    def __init__(self, root = "", is_val = False, val_stride = 10, slice = 7):

        # Filepath
        self.root = root
        self.images_path = os.path.join(self.root, 'MR')
        self.labels_path = os.path.join(self.root, 'CT')

        # MR File List
        self.images = []
        for series in sorted(os.listdir(self.images_path)):
            self.images.append(os.path.join(self.images_path, series))

        # CT File List
        self.labels = []
        for series in sorted(os.listdir(self.labels_path)):
            self.labels.append(os.path.join(self.labels_path, series))

        # Split Training and Validation Dataset
        if is_val:
            self.images = self.images[::val_stride]
            self.labels = self.labels[::val_stride]
        else:
            del self.images[::val_stride]
            del self.labels[::val_stride]

        # Check Data Quantity
        if len(self.images) != len(self.labels):
            raise ValueError('Unequal Amount of Images and Labels.')
        
        # 2D Slice Information
        self.slice = slice                                                  # Channels per Slice
        self.width = self.slice // 2                                        # Width
        self.num_slices = 192 - ((self.width + 1) * 2)                      # Slices per Series
        self.num_series = len(self.images)                                  # Number of Series
        
    def __len__(self):
        
        return (self.num_series * self.num_slices)

    def __getitem__(self, index):
        
        # 2D Slice Index
        series_index = index // self.num_slices                             # Which Series
        slices_index = index % self.num_slices + self.width + 1             # Which Slice

        # Load MR Data: (7, 192, 192)
        image = io.loadmat(self.images[series_index])['MR'].astype('float32')
        image = torch.from_numpy(image)
        image = image[slices_index - self.width : slices_index + self.width + 1, :, :]
        
        # Load CT Data: (1, 192, 192)
        label = io.loadmat(self.labels[series_index])['CT'].astype('float32')
        label = torch.from_numpy(label)
        label = label[slices_index : slices_index + 1, :, :]

        return (image, label)


"""
====================================================================================================
Testing
====================================================================================================
"""
class Testing_3D(Dataset):

    def __init__(self, root = "", slice = 7):

        # Filepath
        self.root = root
        self.images_path = os.path.join(self.root, 'MR')
        self.labels_path = os.path.join(self.root, 'CT')

        # MR File List
        self.images = []
        for series in sorted(os.listdir(self.images_path)):
            self.images.append(os.path.join(self.images_path, series))

        # CT File List
        self.labels = []
        for series in sorted(os.listdir(self.labels_path)):
            self.labels.append(os.path.join(self.labels_path, series))

        # Check Data Quantity
        if len(self.images) != len(self.labels):
            raise ValueError('Unequal Amount of Images and Labels.')
        
        # 2D Slice Information
        self.slice = slice                                                  # Channels per Slice
        self.width = self.slice // 2                                        # Width
        self.num_slices = 192 - ((self.width + 1) * 2)                      # Slices per Series
        self.num_series = len(self.images)                                  # Number of Series

    def __len__(self):

        return (self.num_series * self.num_slices)

    def __getitem__(self, index):

        # 2D Slice Index
        series_index = index // self.num_slices                             # Which Series
        slices_index = index % self.num_slices + self.width + 1             # Which Slice

        # Load MR Data: (7, 192, 192)
        image = io.loadmat(self.images[series_index])['MR'].astype('float32')
        image = torch.from_numpy(image)
        image = image[slices_index - self.width : slices_index + self.width + 1, :, :]
        
        # Load CT Data: (1, 192, 192)
        label = io.loadmat(self.labels[series_index])['CT'].astype('float32')
        label = torch.from_numpy(label)
        label = label[slices_index : slices_index + 1, :, :]

        return (image, label)
    

"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':

    filepath = "/home/ccy/PseudoCT/Data_2D/Train"

    train_2D = Training_2D(filepath, False, 10)

    for i in range(5):
        
        index = random.randint(0, len(train_2D))
        
        image, label, mask = train_2D[index]

        print(label.max(), label.min())
        label = torch.where(mask, label, 5000)
        print(label.max(), label.min())

        # plt.figure()
        # plt.subplot(1, 3, 1)
        # plt.imshow(image[3, :, :], cmap = 'gray')
        # plt.subplot(1, 3, 2)
        # plt.imshow(label[0, :, :], cmap = 'gray')
        # plt.subplot(1, 3, 3)
        # plt.imshow(mask[0, :, :], cmap = 'gray')
        # plt.show()