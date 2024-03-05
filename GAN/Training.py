"""
====================================================================================================
Package
====================================================================================================
"""
import os
import math
import datetime
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from GAN import Generator, Discriminator
from Loss import get_adv_loss, get_pix_loss, get_gdl_loss
from Loss import get_psnr, get_ssim
from Dataset import Training_2D


"""
====================================================================================================
Global Constant
====================================================================================================
"""
MAX = 10000000
STRIDE = 5
BATCH = 32
EPOCH = 10

METRICS = 4
METRICS_GEN = 0
METRICS_DIS = 1
METRICS_PSNR = 2
METRICS_SSIM = 3

DATA_PATH = "C:/Users/PHOENIX/Desktop/PseudoCT/Fake/Train"
MODEL_PATH = ""
RESULTS_PATH = "C:/Users/PHOENIX/Desktop/PseudoCT/GAN/Result"


"""
====================================================================================================
Training
====================================================================================================
"""
class Training():

    """
    ================================================================================================
    Critical Parameters
    ================================================================================================
    """
    def __init__(self):
        
        # Training Device: CPU(cpu) or GPU(cuda)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('\n' + 'Training on: ' + str(self.device) + '\n')

        # Training Timestamp
        self.time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        print('\n' + 'Start From: ' + self.time + '\n')

        # Model and Optimizer
        self.initialization()

        # Begin Point
        self.begin = 1

    """
    ================================================================================================
    Model and Optimizer
    ================================================================================================
    """
    def initialization(self):

        # Model: Generator and Discriminator
        self.gen = Generator().to(self.device)
        self.dis = Discriminator().to(self.device)

        print('\n' + 'Model Initialized' + '\n')
        
        # Optimizer: Adam
        self.optimizer_gen = Adam(self.gen.parameters())
        self.optimizer_dis = Adam(self.dis.parameters())

        print('\n' + 'Optimizer Initialized' + '\n')

    """
    ================================================================================================
    TensorBorad
    ================================================================================================
    """
    def init_tensorboard(self):

        # Metrics Filepath
        log_dir = os.path.join(RESULTS_PATH, 'Metrics', self.time)

        # Tensorboard Writer
        self.train_writer = SummaryWriter(log_dir + '_train')
        self.val_writer = SummaryWriter(log_dir + '_val')

        print('\n' + 'TensorBoard Initialized' + '\n')

    """
    ================================================================================================
    Data Loader
    ================================================================================================
    """
    def init_dl(self):

        # Training
        train_ds = Training_2D(root = DATA_PATH, is_val = False, val_stride = STRIDE)
        train_dl = DataLoader(train_ds, batch_size = BATCH, drop_last = False)

        # Validation
        val_ds = Training_2D(root = DATA_PATH, is_val = True, val_stride = STRIDE)
        val_dl = DataLoader(val_ds, batch_size = BATCH, drop_last = False)

        return train_dl, val_dl
    
    """
    ================================================================================================
    Load Model Parameter and Hyperparameter
    ================================================================================================
    """
    def load_model(self):

        # Check Filepath
        if os.path.isfile(MODEL_PATH):

            # Get Checkpoint Information
            checkpoint = torch.load(MODEL_PATH)
            print('\n' + 'Checkpoint Loaded' + '\n')

            # Model: Generator and Discriminator
            self.gen.load_state_dict(checkpoint['gen_state'])
            self.dis.load_state_dict(checkpoint['dis_state'])
            print('\n' + 'Model Loaded' + '\n')
            
            # Optimizer: Adam
            self.optimizer_gen.load_state_dict(checkpoint['optimizer_gen_state'])
            self.optimizer_dis.load_state_dict(checkpoint['optimizer_dis_state'])
            print('\n' + 'Optimizer Loaded' + '\n')

            # Training Timestamp
            self.time = checkpoint['time']
            print('\n' + 'Continued From: ' +  self.time + '\n')

            # Begin Point
            if checkpoint['epoch'] < EPOCH:
                self.begin = checkpoint['epoch'] + 1
            else:
                self.begin = 1
            print('\n' + 'Start From Epoch: ' + str(self.begin) + '\n')

            # Tensorboard Writer
            log_dir = os.path.join(RESULTS_PATH, 'Metrics', checkpoint['time'])
            self.train_writer = SummaryWriter(log_dir + '_train')
            self.val_writer = SummaryWriter(log_dir + '_val')

            return checkpoint['score']
        
        else:
            
            # Tensorboard
            self.init_tensorboard()
        
            return MAX

    """
    ================================================================================================
    Main Training Function
    ================================================================================================
    """
    def main(self):

        # Data Loader
        train_dl, val_dl = self.init_dl()

        # Get Checkpoint
        best_score = self.load_model()

        # Main Training and Validation Loop
        count = 0
        for epoch_index in range(self.begin, EPOCH + 1):
            
            """
            ========================================================================================
            Training
            ========================================================================================
            """
            # Get Training Metrics
            print('Training: ')
            metrics_train = self.training(epoch_index, train_dl)

            # Save Training Metrics
            self.save_metrics(epoch_index, 'train', metrics_train)
            self.save_images(epoch_index, 'train', train_dl)

            # Validation: Stride = 10
            if epoch_index == 1 or epoch_index % STRIDE == 0:

                """
                ====================================================================================
                Validation
                ====================================================================================
                """
                # Get Validation Metrics
                print('============================================================')
                print('Validation: ')
                metrics_val = self.validation(epoch_index, val_dl)

                # Save Validation Metrics
                score = self.save_metrics(epoch_index, 'val', metrics_val)
                self.save_images(epoch_index, 'val', val_dl)

                # Save Model
                if not math.isnan(score):
                    best_score = min(best_score, score)
                self.save_model(epoch_index, score, (best_score == score))

                print('============================================================')

                # Early Stop
                if score == best_score:
                    count = 0
                elif count < 5:
                    count += 1
                elif count == 5:
                    print('\n' + 'Early Stop' + '\n')
                    break
        
        # Close Tensorboard Writer
        self.train_writer.close()
        self.val_writer.close()

    """
    ================================================================================================
    Training Loop
    ================================================================================================
    """
    def training(self, epoch_index, train_dl):
        
        # Model: Training State
        self.gen.train()
        self.dis.train() 

        # Buffer for Metrics
        metrics = torch.zeros(METRICS, len(train_dl), device = self.device)

        # Progress Bar
        space = "{:3}{:3}{:3}"
        progress = tqdm(enumerate(train_dl), total = len(train_dl), leave = True,
                        bar_format = '{l_bar}{bar:15}{r_bar}{bar:-10b}')
        for batch_index, batch_tuple in progress:

            """
            ========================================================================================
            Prepare Data
            ========================================================================================
            """
            # Get MT and rCT
            # real1: MR; real2: rCT
            (real1_t, real2_t) = batch_tuple
            real1_g = real1_t.to(self.device)
            real2_g = real2_t.to(self.device)

            # Save Min and Max for Reconstruction
            self.max1 = real1_g.max()
            self.min1 = real1_g.min()
            self.max2 = real2_g.max()
            self.min2 = real2_g.min()

            # Min-Max Normalization
            real1_g -= self.min1
            real1_g /= self.max1 + 1e-6
            real2_g -= self.min2
            real2_g /= self.max2 + 1e-6

            # Ground Truth
            valid = torch.ones(BATCH, 1, 12, 12, requires_grad = False, device = self.device)
            fake = torch.zeros(BATCH, 1, 12, 12, requires_grad = False, device = self.device)

            # Get sCT from Generator
            # fake2: sCT
            fake2_g = self.gen(real1_g)

            """
            ========================================================================================
            Generator
            ========================================================================================
            """
            # Refresh Optimizer's Gradient
            self.optimizer_gen.zero_grad()

            # Pixelwise Loss
            loss_pix = get_pix_loss(fake2_g, real2_g)

            # Adversarial loss
            loss_adv = get_adv_loss(self.dis(fake2_g, real1_g), valid)

            # Gradient Difference loss
            loss_gdl = get_gdl_loss(fake2_g, real2_g)           

            # Total Loss
            loss_gen = loss_pix + loss_adv + loss_gdl

            # Update Generator's Parameters
            loss_gen.backward()
            self.optimizer_gen.step()

            """
            ========================================================================================
            Discriminator
            ========================================================================================
            """
            # Refresh Optimizer's Gradient
            self.optimizer_dis.zero_grad()

            # Real Loss
            loss_real = get_adv_loss(self.dis(real2_g, real1_g), valid)

            # Fake Loss
            loss_fake = get_adv_loss(self.dis(fake2_g.detach(), real1_g), fake)

            # Total Loss
            loss_dis = (loss_real + loss_fake) / 2

            # Update Discriminator's Parameters
            loss_dis.backward()
            self.optimizer_dis.step()

            """
            ========================================================================================
            Metrics
            ========================================================================================
            """
            # PSNR
            psnr = get_psnr(fake2_g, real2_g)

            # SSIM
            ssim = get_ssim(fake2_g, real2_g)

            # Save Metrics
            metrics[METRICS_GEN, batch_index] = loss_gen.item()
            metrics[METRICS_DIS, batch_index] = loss_dis.item()
            metrics[METRICS_PSNR, batch_index] = psnr.item()
            metrics[METRICS_SSIM, batch_index] = ssim.item()

            # Progress Bar Information
            progress.set_description('Epoch [' + space.format(epoch_index, ' / ', EPOCH) + ']')
            progress.set_postfix(loss_gen = loss_gen.item(), loss_dis = loss_dis.item())

        return metrics.to('cpu')

    """
    ================================================================================================
    Validation Loop
    ================================================================================================
    """
    def validation(self, epoch_index, val_dl):

        with torch.no_grad():

            # Model: Validation State
            self.gen.eval()
            self.dis.eval() 

            # Buffer for Metrics
            metrics = torch.zeros(METRICS, len(val_dl), device = self.device)
        
            # Progress Bar
            space = "{:3}{:3}{:3}"
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
                (real1_t, real2_t) = batch_tuple
                real1_g = real1_t.to(self.device)
                real2_g = real2_t.to(self.device)

                # Save Min and Max for Reconstruction
                self.max1 = real1_g.max()
                self.min1 = real1_g.min()
                self.max2 = real2_g.max()
                self.min2 = real2_g.min()

                # Min-Max Normalization
                real1_g -= self.min1
                real1_g /= self.max1 + 1e-6
                real2_g -= self.min2
                real2_g /= self.max2 + 1e-6

                # Ground Truth
                valid = torch.ones(BATCH, 1, 12, 12, requires_grad = False, device = self.device)
                fake = torch.zeros(BATCH, 1, 12, 12, requires_grad = False, device = self.device)

                # Get sCT from Generator
                # fake2: sCT
                fake2_g = self.gen(real1_g)

                """
                ========================================================================================
                Generator
                ========================================================================================
                """
                # Pixelwise Loss
                loss_pix = get_pix_loss(fake2_g, real2_g)

                # Adversarial loss
                loss_adv = get_adv_loss(self.dis(fake2_g, real1_g), valid)        

                # Gradient Difference loss
                loss_gdl = get_gdl_loss(fake2_g, real2_g)           

                # Total Loss
                loss_gen = loss_pix + loss_adv + loss_gdl
        
                """
                ========================================================================================
                Discriminator
                ========================================================================================
                """
                # Real Loss
                loss_real2 = get_adv_loss(self.dis(real2_g, real1_g), valid)

                # Fake Loss
                loss_fake2 = get_adv_loss(self.dis(fake2_g.detach(), real1_g), fake)

                # Total Loss
                loss_dis = (loss_real2 + loss_fake2) / 2

                """
                ========================================================================================
                Metrics
                ========================================================================================
                """
                # PSNR
                psnr = get_psnr(fake2_g, real2_g)

                # SSIM
                ssim = get_ssim(fake2_g, real2_g)

                # Save Metrics
                metrics[METRICS_GEN, batch_index] = loss_gen.item()
                metrics[METRICS_DIS, batch_index] = loss_dis.item()
                metrics[METRICS_PSNR, batch_index] = psnr.item()
                metrics[METRICS_SSIM, batch_index] = ssim.item()
                
                # Progress Bar Information
                progress.set_description('Epoch [' + space.format(epoch_index, ' / ', EPOCH) + ']')
                progress.set_postfix(loss_gen = loss_gen.item(), loss_dis = loss_dis.item())

            return metrics.to('cpu')
    
    """
    ================================================================================================
    Save Metrics for Whole Epoch
    ================================================================================================
    """ 
    def save_metrics(self, epoch_index, mode, metrics_t):

        # Torch Tensor to Numpy Array
        metrics_a = metrics_t.detach().numpy().mean(axis = 1)

        # Create Dictionary
        metrics_dict = {}
        metrics_dict['Loss/Generator'] = metrics_a[METRICS_GEN]
        metrics_dict['Loss/Discriminator'] = metrics_a[METRICS_DIS]
        metrics_dict['Metrics/PSNR'] = metrics_a[METRICS_PSNR]
        metrics_dict['Metrics/SSIM'] = metrics_a[METRICS_SSIM]

        # Save Metrics
        writer = getattr(self, mode + '_writer')
        for key, value in metrics_dict.items():
            
            writer.add_scalar(key, value.item(), epoch_index)
        
        # Refresh Tensorboard Writer
        writer.flush()

        return metrics_dict['Loss/Generator']

    """
    ================================================================================================
    Save Image
    ================================================================================================
    """ 
    def save_images(self, epoch_index, mode, dataloader):

        # Model: Validation State
        self.gen.eval()

        # Get MT and rCT
        # real1: MR; real2: rCT
        (real1_t, real2_t) = dataloader.dataset[90]
        real1_g = real1_t.to(self.device).unsqueeze(0)
        real2_g = real2_t.to(self.device).unsqueeze(0)

        # Get sCT from Generator
        # fake2: sCT
        fake2_g = self.gen(real1_g)

        # Min-Max Normalization
        real1_g -= real1_g.min()
        real1_g /= real1_g.max() + 1e-6
        real2_g -= real2_g.min()
        real2_g /= real2_g.max() + 1e-6

        # Torch Tensor to Numpy Array
        real1_a = real1_g.to('cpu').detach().numpy()[:, 3, :, :]
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

        # Save Image
        writer = getattr(self, mode + '_writer')
        writer.add_image(mode + '/MR', real1_a, epoch_index, dataformats = 'CHW')
        writer.add_image(mode + '/rCT', real2_a, epoch_index, dataformats = 'CHW')
        writer.add_image(mode + '/sCT', fake2_a, epoch_index, dataformats = 'CHW')
        writer.add_image(mode + '/Diff', diff, epoch_index, dataformats = 'HWC')

        # Refresh Tensorboard Writer
        writer.flush()

    """
    ================================================================================================
    Save Model
    ================================================================================================
    """ 
    def save_model(self, epoch_index, score, is_best):

        # Time, Model State, Optimizer State
        # Ending Epoch, Best Score
        state = {
            'time': self.time,
            'gen_state': self.gen.state_dict(),
            'gen_name': type(self.gen).__name__,
            'dis_state': self.dis.state_dict(),
            'dis_name': type(self.dis).__name__,
            'optimizer_gen_state': self.optimizer_gen.state_dict(),
            'optimizer_gen_name': type(self.optimizer_gen).__name__,
            'optimizer_dis_state': self.optimizer_dis.state_dict(),
            'optimizer_dis_name': type(self.optimizer_dis).__name__,
            'epoch': epoch_index,
            'score': score,
        }

        # Save Model
        model_path = os.path.join(RESULTS_PATH, 'Model', self.time + '.pt')
        torch.save(state, model_path)

        # Save Best Model
        if is_best:
            best_path = os.path.join(RESULTS_PATH, 'Model', self.time + '.best.pt')
            torch.save(state, best_path)


"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':

    Training().main()