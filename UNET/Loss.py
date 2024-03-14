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
Adversarial Loss: MSE Loss
====================================================================================================
"""
def get_adv_loss(predicts, labels):

    return MSELoss().to(device)(predicts, labels)


"""
====================================================================================================
Pixelwise Loss: L1 Loss
Alternative: L2
====================================================================================================
"""
def get_pix_loss(predicts, labels):

    return L1Loss().to(device)(predicts, labels)


"""
====================================================================================================
Gradient Difference Loss
====================================================================================================
"""
def get_gdl_loss(predicts, labels):

    # First Derivative of Predicts
    grad_predicts_x = torch.abs(predicts[:, :, 1:, :] - predicts[:, :, :-1, :])
    grad_predicts_y = torch.abs(predicts[:, :, :, 1:] - predicts[:, :, :, :-1])

    # First Derivative of Labels
    grad_labels_x = torch.abs(labels[:, :, 1:, :] - labels[:, :, :-1, :])
    grad_labels_y = torch.abs(labels[:, :, :, 1:] - labels[:, :, :, :-1])

    # Gradient Difference
    gdl_x = MSELoss().to(device)(grad_predicts_x, grad_labels_x)
    gdl_y = MSELoss().to(device)(grad_predicts_y, grad_labels_y)

    return gdl_x + gdl_y


"""
====================================================================================================
MAE: L1 Loss
====================================================================================================
"""
def get_mae(predicts, labels):

    return L1Loss().to(device)(predicts, labels).item()


"""
====================================================================================================
PSNR
====================================================================================================
"""
def get_psnr(predicts, labels):

    return PeakSignalNoiseRatio().to(device)(predicts, labels).item()


"""
====================================================================================================
SSIM
====================================================================================================
"""
def get_ssim(predicts, labels):

    return StructuralSimilarityIndexMeasure().to(device)(predicts, labels).item()

"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':

    image = torch.rand((16, 1, 192, 192))
    label = torch.rand((16, 1, 192, 192))

    adv = get_adv_loss(image, label)
    print(adv, adv.size())

    pix = get_pix_loss(image, label)
    print(pix, pix.size())

    gdl = get_gdl_loss(image, label)
    print(gdl, gdl.size())

    mae = get_mae(image, label)
    print(mae)

    psnr = get_psnr(image, label)
    print(psnr)

    ssim = get_ssim(image, label)
    print(ssim)