"""
====================================================================================================
Package
====================================================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from Unet import Unet, Pretrain


"""
====================================================================================================
Generator: Unet
====================================================================================================
"""
def Generator(pretrain = True, slice = 7):

    if pretrain:
        model = Pretrain(slice = slice)

    else:
        model = Unet(slice = slice)
    
    return model


"""
====================================================================================================
Initialization Block
====================================================================================================
"""
class Init(nn.Module):

    def __init__(self, filters):

        super().__init__()

        self.init_block = nn.Sequential(nn.Conv2d(1, filters, kernel_size = 4, stride = 2, padding = 1),
                                        nn.LeakyReLU(0.01))

    def forward(self, img_in):

        img_out = self.init_block(img_in)

        return img_out


"""
====================================================================================================
Discriminator Block
====================================================================================================
"""
class Dis(nn.Module):

    def __init__(self, filters):

        super().__init__()

        self.dis_block = nn.Sequential(nn.Conv2d(filters // 2, filters, kernel_size = 4, stride = 2, padding = 1),
                                       nn.BatchNorm2d(filters),
                                       nn.LeakyReLU(0.01))

    def forward(self, img_in):

        img_out = self.dis_block(img_in)

        return img_out


"""
====================================================================================================
Final Block
====================================================================================================
"""
class Final(nn.Module):

    def __init__(self, filters):

        super().__init__()

        self.final_block = nn.Sequential(nn.ZeroPad2d((1, 0, 1, 0)),
                                         nn.Conv2d(filters, 1, kernel_size = 4, padding = 1, bias = False))

    def forward(self, img_in):

        img_out = self.final_block(img_in)

        return img_out
    

"""
====================================================================================================
Discriminator
====================================================================================================
"""
class Discriminator(nn.Module):

    def __init__(self):

        super().__init__()

        self.filters = [64, 128, 256, 512]

        self.init = Init(self.filters[0])

        self.dis1 = Dis(self.filters[1])
        self.dis2 = Dis(self.filters[2])
        self.dis3 = Dis(self.filters[3])

        self.final = Final(self.filters[3])

    def forward(self, img_in):

        init = self.init(img_in)

        dis1 = self.dis1(init)
        dis2 = self.dis2(dis1)
        dis3 = self.dis3(dis2)

        final = self.final(dis3)

        return final


"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':

    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print('\n' + 'Training on Device: ' + str(device) + '\n')
    
    # model = Generator(pretrain = True, slice = 7).to(device = device)
    # print(summary(model, input_size = (7, 192, 192), batch_size = 2))

    model = Discriminator().to(device = device)
    print(summary(model, input_size = (1, 192, 192), batch_size = 2))