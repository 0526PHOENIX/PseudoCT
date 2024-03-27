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
        
    def __len__(self):
        
        return len(self.images)

    def __getitem__(self, index):

        # Load MR Data: (7, 192, 192)
        image = io.loadmat(self.images[index])['MR'].astype('float32')
        image = torch.from_numpy(image)
        
        # Load CT Data: (1, 192, 192)
        label = io.loadmat(self.labels[index])['CT'].astype('float32')
        label = torch.from_numpy(label)

        return (image, label)
    
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
        
    def __len__(self):
        
        return len(self.images)

    def __getitem__(self, index):

        # Load MR Data: (7, 192, 192)
        image = io.loadmat(self.images[index])['MR'].astype('float32')
        image = torch.from_numpy(image)
        
        # Load CT Data: (1, 192, 192)
        label = io.loadmat(self.labels[index])['CT'].astype('float32')
        label = torch.from_numpy(label)

        return (image, label)

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

    filepath = "C:/Users/PHOENIX/Desktop/PseudoCT/Data_2D/Train"

    train_2D = Training_2D(filepath, False, 10)

    for i in range(5):
        
        index = random.randint(0, len(train_2D))
        
        image, label = train_2D[index]

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(image[3, :, :], cmap = 'gray')
        plt.subplot(1, 2, 2)
        plt.imshow(label[0, :, :], cmap = 'gray')
        plt.show()