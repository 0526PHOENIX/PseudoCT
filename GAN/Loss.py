"""
====================================================================================================
Package
====================================================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss, L1Loss
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""
====================================================================================================
Adversarial Loss
====================================================================================================
"""
def get_adv_loss(predicts, labels):

    adv = MSELoss().to(device)(predicts, labels)

    return adv


"""
====================================================================================================
Pixel-Wise Loss
====================================================================================================
"""
def get_pix_loss(predicts, labels):

    pix = L1Loss().to(device)(predicts, labels)

    return pix


"""
====================================================================================================
PSNR
====================================================================================================
"""
def get_psnr(predicts, labels):

    psnr = PeakSignalNoiseRatio().to(device)(predicts, labels)

    return psnr

"""
====================================================================================================
SSIM
====================================================================================================
"""
def get_ssim(predicts, labels):

    ssim = StructuralSimilarityIndexMeasure().to(device)(predicts, labels)

    return ssim

"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':

    image = torch.rand((16, 1, 512, 512))
    label = torch.rand((16, 1, 512, 512))

    print()
    adv = get_adv_loss(image, label)
    print(adv, adv.size())

    print()
    pix = get_pix_loss(image, label)
    print(pix, pix.size())

    print()
    psnr = get_psnr(image, label)
    print(psnr, psnr.size())

    print()
    ssim = get_ssim(image, label)
    print(ssim, ssim.size())
