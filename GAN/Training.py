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

from Gan import Generator, Discriminator
from Loss import get_adv_loss, get_pix_loss, get_gdl_loss
from Loss import get_mae, get_psnr, get_ssim
from Dataset import Training_2D, Training_3D


"""
====================================================================================================
Global Constant
====================================================================================================
"""
MAX = 10000000
STRIDE = 5
BATCH = 16
EPOCH = 400
LR_GEN = 1e-4
LR_DIS = 1e-6

PRETRAIN = True

METRICS = 5
METRICS_GEN = 0
METRICS_DIS = 1
METRICS_MAE = 2
METRICS_PSNR = 3
METRICS_SSIM = 4

LAMBDA_1 = 2
LAMBDA_2 = 3
LAMBDA_3 = 1

"""
C:/Users/PHOENIX/Desktop/
/home/ccy/
"""
DATA_PATH = "/home/ccy/PseudoCT/Data_2D/Train"
MODEL_PATH = "/home/ccy/PseudoCT/GAN/Result/Model/2024-03-27_12-48.pt"
RESULTS_PATH = "/home/ccy/PseudoCT/GAN/Result"


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
        self.gen = Generator(pretrain = PRETRAIN, slice = 7).to(self.device)
        self.dis = Discriminator().to(self.device)

        print('\n' + 'Model Initialized' + '\n')
        
        # Optimizer: Adam
        self.optimizer_gen = Adam(self.gen.parameters(), lr = LR_GEN)
        self.optimizer_dis = Adam(self.dis.parameters(), lr = LR_DIS)

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
        self.train_writer = SummaryWriter(log_dir + '/Train')
        self.val_writer = SummaryWriter(log_dir + '/Val')

        print('\n' + 'TensorBoard Initialized' + '\n')

    """
    ================================================================================================
    Data Loader
    ================================================================================================
    """
    def init_dl(self):

        # Training
        train_ds = Training_2D(root = DATA_PATH, is_val = False, val_stride = STRIDE)
        train_dl = DataLoader(train_ds, batch_size = BATCH, shuffle = True, drop_last = False)

        # Validation
        val_ds = Training_2D(root = DATA_PATH, is_val = True, val_stride = STRIDE)
        val_dl = DataLoader(val_ds, batch_size = BATCH, shuffle = True, drop_last = False)

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
            self.train_writer = SummaryWriter(log_dir + '/Train')
            self.val_writer = SummaryWriter(log_dir + '/Val')

            return checkpoint['score']
        
        else:
            
            # Tensorboard
            self.init_tensorboard()

            # Save Hyperparameters
            self.save_hyper()
        
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
            score = self.save_metrics(epoch_index, 'train', metrics_train)
            self.save_images(epoch_index, 'train', train_dl)
            self.save_model(epoch_index, score)

            # Validation: Stride = 10
            if epoch_index == 1 or epoch_index % STRIDE == 0:

                """
                ====================================================================================
                Validation
                ====================================================================================
                """
                # Get Validation Metrics
                print('===========================================================================')
                print('Validation: ')
                metrics_val = self.validation(epoch_index, val_dl)

                # Save Validation Metrics
                score = self.save_metrics(epoch_index, 'val', metrics_val)
                self.save_images(epoch_index, 'val', val_dl)

                # Save Model
                if not math.isnan(score):
                    best_score = min(best_score, score)
                self.save_model(epoch_index, score, (best_score == score))

                print('===========================================================================')

                # # Early Stop
                # if score == best_score:
                #     count = 0
                # elif count < 5:
                #     count += 1
                # elif count == 5:
                #     print('\n' + 'Early Stop' + '\n')
                #     break
        
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
            # Refresh Optimizer's Gradient
            self.optimizer_gen.zero_grad()

            # Adversarial loss
            loss_adv = get_adv_loss(self.dis(fake2_g), valid)

            # Pixelwise Loss
            loss_pix = get_pix_loss(fake2_g, real2_g)

            # Gradient Difference loss
            loss_gdl = get_gdl_loss(fake2_g, real2_g)           

            # Total Loss
            loss_gen = LAMBDA_1 * loss_adv + LAMBDA_2 * loss_pix + LAMBDA_3 * loss_gdl

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
            loss_real = get_adv_loss(self.dis(real2_g), valid)

            # Fake Loss
            loss_fake = get_adv_loss(self.dis(real1_g[:, 3:4, :, :]), fake)

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
            metrics[METRICS_DIS, batch_index] = loss_dis.item()
            metrics[METRICS_MAE, batch_index] = mae
            metrics[METRICS_PSNR, batch_index] = psnr
            metrics[METRICS_SSIM, batch_index] = ssim

            # Progress Bar Information
            progress.set_description('Epoch [' + space.format(epoch_index, ' / ', EPOCH) + ']')
            progress.set_postfix(loss_gen = loss_gen.item(), loss_dis = loss_dis.item(), mae = mae)

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
                Discriminator
                ========================================================================================
                """
                # Real Loss
                loss_real = get_adv_loss(self.dis(real2_g), valid)

                # Fake Loss
                loss_fake = get_adv_loss(self.dis(real1_g[:, 3:4, :, :]), fake)

                # Total Loss
                loss_dis = (loss_real + loss_fake) / 2

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
                metrics[METRICS_DIS, batch_index] = loss_dis.item()
                metrics[METRICS_MAE, batch_index] = mae
                metrics[METRICS_PSNR, batch_index] = psnr
                metrics[METRICS_SSIM, batch_index] = ssim
                
                # Progress Bar Information
                progress.set_description('Epoch [' + space.format(epoch_index, ' / ', EPOCH) + ']')
                progress.set_postfix(loss_gen = loss_gen.item(), loss_dis = loss_dis.item(), mae = mae)

            return metrics.to('cpu')
    
    """
    Save Hyperparameter: Batch Size, Epoch, Learning Rate
    """
    def save_hyper(self):

        path = os.path.join(RESULTS_PATH, 'Metrics', self.time, 'Hyper.txt')

        with open(path, 'w') as f:

            print('Model:', 'Pix2Pix', file = f)
            print('Batch Size:', BATCH, file = f)
            print('Epoch:', EPOCH, file = f)
            print('Gen Learning Rate:', LR_GEN, file = f)
            print('Dis Learning Rate:', LR_DIS, file = f)
            print('Pretrain:', PRETRAIN, file = f)
            print('Adv Loss Lamda:', LAMBDA_1, file = f)
            print('Pix Loss Lamda:', LAMBDA_2, file = f)
            print('GDL Loss Lamda:', LAMBDA_3, file = f)

        print('\n' + 'Hyperparameter Saved' + '\n')

    """
    ================================================================================================
    Save Metrics for Whole Epoch
    ================================================================================================
    """ 
    def save_metrics(self, epoch_index, mode, metrics_t):

        # Get Writer
        writer = getattr(self, mode + '_writer')

        # Torch Tensor to Numpy Array
        metrics_a = metrics_t.detach().numpy().mean(axis = 1)

        # Create Dictionary
        metrics_dict = {}
        metrics_dict['Loss/Generator'] = metrics_a[METRICS_GEN]
        metrics_dict['Loss/Discriminator'] = metrics_a[METRICS_DIS]
        metrics_dict['Metrics/MAE'] = metrics_a[METRICS_MAE]
        metrics_dict['Metrics/PSNR'] = metrics_a[METRICS_PSNR]
        metrics_dict['Metrics/SSIM'] = metrics_a[METRICS_SSIM]

        # Save Metrics
        for key, value in metrics_dict.items():
            
            writer.add_scalar(key, value.item(), epoch_index)
        
        # Refresh Tensorboard Writer
        writer.flush()

        return metrics_dict['Metrics/MAE']

    """
    ================================================================================================
    Save Image
    ================================================================================================
    """ 
    def save_images(self, epoch_index, mode, dataloader):

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
        writer.add_image(mode + '/MR', real1_t[3:4, :, :], epoch_index, dataformats = 'CHW')
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
    def save_model(self, epoch_index, score, is_best = True):

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