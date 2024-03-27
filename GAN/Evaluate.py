"""
====================================================================================================
Package
====================================================================================================
"""
import os
import math
import datetime
import numpy as np
from scipy import io
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Gan import Generator
from Loss import get_adv_loss, get_pix_loss, get_gdl_loss
from Loss import get_mae, get_psnr, get_ssim
from Dataset import Training_2D, Training_3D, Testing_2D, Testing_3D


"""
====================================================================================================
Global Constant
====================================================================================================
"""
STRIDE = 5
BATCH = 16

METRICS = 4
METRICS_GEN = 0
METRICS_MAE = 1
METRICS_PSNR = 2
METRICS_SSIM = 3

PRETRAIN = True

LAMBDA_1 = 2
LAMBDA_2 = 3
LAMBDA_3 = 1

VAL_PATH = "/home/ccy/PseudoCT/Data_2D/Train"
TEST_PATH = "/home/ccy/PseudoCT/Data_2D/Test"

FILE_PATH = "/home/ccy/PseudoCT/Data/Train"

MODEL_PATH = "/home/ccy/PseudoCT/GAN/Result/Model/2024-03-27_12-48.best.pt"
RESULTS_PATH = "/home/ccy/PseudoCT/GAN/Result"


"""
====================================================================================================
Evaluate
====================================================================================================
"""
class Evaluate():

    """
    ================================================================================================
    Initialize Critical Parameters
    ================================================================================================
    """
    def __init__(self):

        # Evaluating Device: CPU(cpu) or GPU(cuda)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('\n' + 'Training on: ' + str(self.device) + '\n')

        # Evaluating Timestamp
        self.time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        print('\n' + 'Start From: ' + self.time + '\n')

        # Model
        self.initialization()

    """
    ================================================================================================
    Model
    ================================================================================================
    """
    def initialization(self):

        self.gen = Generator(pretrain = PRETRAIN, slice = 7).to(self.device)

        print('\n' + 'Model Initialized' + '\n')

    """
    ================================================================================================
    TensorBorad
    ================================================================================================
    """  
    def init_tensorboard(self):

        # Metrics Filepath
        log_dir = os.path.join(RESULTS_PATH, 'Metrics', self.time)

        # Tensorboard Writer
        self.val_writer = SummaryWriter(log_dir + '/Val')
        self.test_writer = SummaryWriter(log_dir + '/Test')

        print('\n' + 'TensorBoard Initialized' + '\n')
    
    """
    ================================================================================================
    Initialize Testing Data Loader
    ================================================================================================
    """
    def init_dl(self):

        # Validation
        val_ds = Training_2D(root = VAL_PATH, is_val = True, val_stride = STRIDE)
        val_dl = DataLoader(val_ds, batch_size = BATCH, shuffle = True, drop_last = False)

        test_ds = Testing_2D(root = TEST_PATH)
        test_dl = DataLoader(test_ds, batch_size = BATCH, drop_last = False)

        return val_dl, test_dl

    """
    ================================================================================================
    Load Model Parameter and Hyperparameter
    ================================================================================================
    """
    def load_model(self):

        if os.path.isfile(MODEL_PATH):

            # Get Checkpoint Information
            checkpoint = torch.load(MODEL_PATH)
            print('\n' + 'Checkpoint Loaded' + '\n')

            # Model: Generator and Discriminator
            self.gen.load_state_dict(checkpoint['gen_state'])
            print('\n' + 'Model Loaded' + '\n')

            # Tensorboard
            self.init_tensorboard()

    """
    ================================================================================================
    Main Evaluating Function
    ================================================================================================
    """
    def main(self):

        # Data Loader
        val_dl, test_dl = self.init_dl()

        # Get Checkpoint
        self.load_model()

        # Validate Model
        print('\n' + 'Validation: ')
        metrics_val = self.validation(val_dl)
        self.save_metrics('val', metrics_val)
        self.save_images('val', val_dl)

        # # Evaluate Model
        # print('\n' + 'Testing: ')
        # metrics_test = self.testing(test_dl)
        # self.save_metrics(metrics_test)
        # self.save_images(test_dl)

    """
    ================================================================================================
    Validation Loop
    ================================================================================================
    """
    def validation(self, val_dl):

        with torch.no_grad():

            # Model: Validation State
            self.gen.eval()

            # Buffer for Matrics
            metrics = torch.zeros(METRICS, len(val_dl), device = self.device)
        
            progress = tqdm(enumerate(val_dl), total = len(val_dl), leave = True,
                            bar_format = '{l_bar}{bar:15}{r_bar}{bar:-10b}')
            for batch_index, batch_tuple in progress:

                """
                ========================================================================================
                Prepare Data
                ========================================================================================
                """
                # Get MT and rCT
                # real1: MR; real2: rCT
                (images_t, labels_t) = batch_tuple
                real1_g = images_t.to(self.device)
                real2_g = labels_t.to(self.device)

                # Z-Score Normalization
                real1_g -= real1_g.mean()
                real1_g /= real1_g.std()
                # Linear Sacling to [-1, 1]
                real1_g -= real1_g.min()
                real1_g /= real1_g.max()
                real1_g = (real1_g * 2) - 1

                # Linear Sacling to [-1, 1]
                real2_g -= -1000
                real2_g /= 4000
                real2_g = (real2_g * 2) - 1

                # Ground Truth
                valid = torch.ones(real1_g.size(0), 1, 12, 12, requires_grad = False, device = self.device)
                fake = torch.zeros(real1_g.size(0), 1, 12, 12, requires_grad = False, device = self.device)

                # Get sCT from Generator
                # fake2: sCT
                fake2_g = self.gen(real1_g)

                """
                ========================================================================================
                Generator
                ========================================================================================
                """
                # Adversarial loss
                loss_adv = get_adv_loss(self.dis(fake2_g), valid)  

                # Pixelwise Loss
                loss_pix = get_pix_loss(fake2_g, real2_g)      

                # Gradient Difference loss
                loss_gdl = get_gdl_loss(fake2_g, real2_g)           

                # Total Loss
                loss_gen = LAMBDA_1 * loss_adv + LAMBDA_2 * loss_pix + LAMBDA_3 * loss_gdl

                """
                ========================================================================================
                Metrics
                ========================================================================================
                """
                # Reconstruction
                real2_g = ((real2_g + 1) * 2000) - 1000
                fake2_g = ((fake2_g + 1) * 2000) - 1000

                # MAE
                mae = get_mae(fake2_g, real2_g)

                # PSNR
                psnr = get_psnr(fake2_g, real2_g)

                # SSIM
                ssim = get_ssim(fake2_g, real2_g)

                # Save Metrics
                metrics[METRICS_GEN, batch_index] = loss_gen.item()
                metrics[METRICS_MAE, batch_index] = mae
                metrics[METRICS_PSNR, batch_index] = psnr
                metrics[METRICS_SSIM, batch_index] = ssim

                progress.set_description('Evaluating Validation Set')
                progress.set_postfix(loss_gen = loss_gen.item(), mae = mae)

        return metrics.to('cpu')


    """
    ================================================================================================
    Save Metrics for Whole Epoch
    ================================================================================================
    """ 
    def save_metrics(self, mode, metrics_t):

        # Get Writer
        writer = getattr(self, mode + '_writer')
        
        # Torch Tensor to Numpy Array
        metrics_a = metrics_t.detach().numpy().mean(axis = 1)

        # Create Dictionary
        metrics_dict = {}
        metrics_dict['Loss/Generator'] = metrics_a[METRICS_GEN]
        metrics_dict['Metrics/MAE'] = metrics_a[METRICS_MAE]
        metrics_dict['Metrics/PSNR'] = metrics_a[METRICS_PSNR]
        metrics_dict['Metrics/SSIM'] = metrics_a[METRICS_SSIM]

        # Save Metrics
        for key, value in metrics_dict.items():

            writer.add_scalar(key, value.item())
        
        # Refresh tensorboard Writer
        writer.flush()

    """
    ================================================================================================
    Save Image
    ================================================================================================
    """ 
    def save_images(self, mode, dataloader):

        # Model: Validation State
        self.gen.eval()
    
        # Get Writer
        writer = getattr(self, mode + '_writer')

        # Get MT and rCT
        # real1: MR; real2: rCT
        (real1_t, real2_t) = dataloader.dataset[90]
        real1_g = real1_t.to(self.device).unsqueeze(0)
        real2_g = real2_t.to(self.device).unsqueeze(0)

        # Z-Score Normalization
        real1_g -= real1_g.mean()
        real1_g /= real1_g.std()
        # Linear Sacling to [-1, 1]
        real1_g -= real1_g.min()
        real1_g /= real1_g.max()
        real1_g = (real1_g * 2) - 1

        # Linear Sacling to [-1, 1]
        real2_g -= -1000
        real2_g /= 4000
        real2_g = (real2_g * 2) - 1

        # Get sCT from Generator
        # fake2: sCT
        fake2_g = self.gen(real1_g)

        # Torch Tensor to Numpy Array
        real2_a = real2_g.to('cpu').detach().numpy()[0, :, :, :]
        fake2_a = fake2_g.to('cpu').detach().numpy()[0, :, :, :]
        
        # Color Map: Cool-Warm
        colormap = plt.get_cmap('coolwarm')

        # Difference Map
        diff = np.abs(real2_a - fake2_a)
        diff -= diff.min()
        diff /= diff.max()
        diff = colormap(diff[0])
        diff = diff[..., :3]

        # Linear Sacling to [0, 1]
        real1_t -= real1_t.min()
        real1_t /= real1_t.max()

        real2_a = (real2_a + 1) / 2
        fake2_a = (fake2_a + 1) / 2

        # Save Image
        writer.add_image(mode + '/MR', real1_t[3:4, :, :], dataformats = 'CHW')
        writer.add_image(mode + '/rCT', real2_a, dataformats = 'CHW')
        writer.add_image(mode + '/sCT', fake2_a, dataformats = 'CHW')
        writer.add_image(mode + '/Diff', diff, dataformats = 'HWC')

        # Refresh Tensorboard Writer
        writer.flush()

    """
    ================================================================================================
    Inference
    ================================================================================================
    """
    def inference(self, number = 1, width = 3):

        print('\n' + 'Inferencing: ')

        # Model: validation State
        self.gen.eval()

        # Get MR Series
        if number < 9:
            image = io.loadmat(os.path.join(FILE_PATH, 'MR' + '0' + str(number) + '.mat'))['MR'].astype('float32')
        else:
            image = io.loadmat(os.path.join(FILE_PATH, 'MR' + str(number) + '.mat'))['MR'].astype('float32')

        buffer = []
        for i in range(width, image.shape[0] - width):

            # Get MR and CT Slice
            # real1: MR; real2: rCT
            real1_t = image[i - width : i + width + 1, :, :]

            real1_g = real1_t.to(self.device).unsqueeze(0)

            # Z-Score Normalization
            real1_g -= real1_g.mean()
            real1_g /= real1_g.std()
            # Linear Sacling to [-1, 1]
            real1_g -= real1_g.min()
            real1_g /= real1_g.max()
            real1_g = (real1_g * 2) - 1

            # Get sCT from Generator
            # fake2: sCT
            fake2_g = self.gen(real1_g)

            # Reconstruction
            fake2_g = ((fake2_g + 1) * 2000) - 1000

            # Stack
            buffer.append(fake2_g[0, 0, :, :])

        # Get CT Series from Stack
        result = np.stack(buffer, axis = 0)

        # Save CT Series
        if number < 9:
            io.savemat(os.path.join(RESULTS_PATH, 'CT' + '0' + str(number) + '.mat'), {'CT': result})
        else:
            io.savemat(os.path.join(RESULTS_PATH, 'CT' + str(number) + '.mat'), {'CT': result})


"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':

    eva = Evaluate()
    eva.main()
    eva.inference(number = 3)
    