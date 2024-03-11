"""
====================================================================================================
Package
====================================================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


"""
====================================================================================================
Residual Block
====================================================================================================
"""
class Res(nn.Module):

    def __init__(self, filters):

        super().__init__()

        # (Normalization -> Activation -> Convolution) * 2
        self.res_block = nn.Sequential(nn.BatchNorm2d(filters), 
                                       nn.LeakyReLU(0.01),
                                       nn.Conv2d(filters, filters, kernel_size = 3, padding = 1, bias = False),
                                       nn.BatchNorm2d(filters),
                                       nn.LeakyReLU(0.01),
                                       nn.Conv2d(filters, filters, kernel_size = 3, padding = 1, bias = False))

    def forward(self, img_in):

        img_out = self.res_block(img_in)

        # Jump Connection
        return img_in + img_out


"""
====================================================================================================
Initialization Block
====================================================================================================
"""
class Init(nn.Module):

    def __init__(self, slice, filters):

        super().__init__()

        # Convolution -> Dropout -> Residual Block
        self.conv = nn.Conv2d(slice, filters, kernel_size = 3, padding = 1, bias = False)
        self.drop = nn.Dropout2d(0.2)
        self.res = Res(filters)

    def forward(self, img_in):

        img_out = self.conv(img_in)
        img_out = self.drop(img_out)
        img_out = self.res(img_out)

        return img_out


"""
====================================================================================================
Final Block
====================================================================================================
"""
class Final(nn.Module):

    def __init__(self, filters):

        super().__init__()

        self.final_block = nn.Sequential(nn.Conv2d(filters, 1, kernel_size = 1, bias = False),
                                         nn.Sigmoid())
    
    def forward(self, img_in):

        img_out = self.final_block(img_in)

        return img_out


"""
====================================================================================================
Unet
====================================================================================================
"""


"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':

    model = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained = True, scale = 0.5)
    
    num_1 = model.inc.double_conv[3].out_channels
    model.inc = Init(7, num_1)

    num_2 = model.outc.conv.in_channels
    model.outc = Final(num_2)
    
    # print(model)
    
    print(summary(model, input_size = (7, 192, 192), batch_size = 2, device = 'cpu'))
    
    
    