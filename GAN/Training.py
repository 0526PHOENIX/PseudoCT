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

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from GAN import Generator, Discriminator
from Loss import get_adv_loss, get_pix_loss
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

DATA_PATH = "/home/ccy/PseudoCT/Fake/Train"
MODEL_PATH = ""
RESULTS_PATH = "/home/ccy/PseudoCT/GAN/Result"


"""
====================================================================================================
Training
====================================================================================================
"""
class Training():

    """
    ================================================================================================
    Initialize Critical Parameters
    ================================================================================================
    """
    def __init__(self):
        
        # training device
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        print('\n' + 'Training on: ' + str(self.device) + '\n')

        # time and tensorboard writer
        self.time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        print('\n' + 'Start From: ' + self.time + '\n')

        self.train_writer = None
        self.val_writer = None

        # model and optimizer
        self.init_model()
        self.init_optimizer()

        # begin epoch
        self.begin = 1

    """
    ================================================================================================
    Initialize Model
    ================================================================================================
    """
    def init_model(self):

        self.gen = Generator().to(self.device)
        self.dis = Discriminator().to(self.device)

        print('\n' + 'Model Initialized' + '\n')
    
    """
    ================================================================================================
    Initialize Optimizer
    ================================================================================================
    """
    def init_optimizer(self):

        self.optimizer_gen = Adam(self.gen.parameters(), lr = 1e-4)
        self.optimizer_dis = Adam(self.dis.parameters(), lr = 1e-4)

        print('\n' + 'Optimizer Initialized' + '\n')

    """
    ================================================================================================
    Initialize TensorBorad
    ================================================================================================
    """
    def init_tensorboard(self):

        if (self.train_writer == None) or (self.val_writer == None):

            # metrics path
            log_dir = os.path.join(RESULTS_PATH, 'Metrics', self.time)

            # training and validation tensorboard writer
            self.train_writer = SummaryWriter(log_dir + '_train')
            self.val_writer = SummaryWriter(log_dir + '_val')

            print('\n' + 'TensorBoard Initialized' + '\n')

    """
    ================================================================================================
    Initialize Data Loader
    ================================================================================================
    """
    def init_dl(self):

        train_ds = Training_2D(root = DATA_PATH, is_val = False, val_stride = STRIDE)
        train_dl = DataLoader(train_ds, batch_size = BATCH, drop_last = False)

        val_ds = Training_2D(root = DATA_PATH, is_val = True, val_stride = STRIDE)
        val_dl = DataLoader(val_ds, batch_size = BATCH, drop_last = False)

        return train_dl, val_dl
    
    """
    ================================================================================================
    Load Model Parameter and Hyperparameter
    ================================================================================================
    """
    def load_model(self):

        if os.path.isfile(MODEL_PATH):

            # get checkpoint
            checkpoint = torch.load(MODEL_PATH)
            print('\n' + 'Checkpoint Loaded' + '\n')

            # load model
            self.gen.load_state_dict(checkpoint['gen_state'])
            self.dis.load_state_dict(checkpoint['dis_state'])
            print('\n' + 'Model Loaded' + '\n')
            
            # load optimizer
            self.optimizer_gen.load_state_dict(checkpoint['optimizer_gen_state'])
            self.optimizer_dis.load_state_dict(checkpoint['optimizer_dis_state'])
            print('\n' + 'Optimizer Loaded' + '\n')

            # set time
            self.time = checkpoint['time']
            print('\n' + 'Continued From: ' +  self.time + '\n')

            # set epoch
            if checkpoint['epoch'] < EPOCH:
                self.begin = checkpoint['epoch'] + 1
            else:
                self.begin = 1
            print('\n' + 'Start From Epoch: ' + str(self.begin) + '\n')

            # set tensorboard
            log_dir = os.path.join(RESULTS_PATH, 'Metrics', checkpoint['time'])
            self.train_writer = SummaryWriter(log_dir + '_train')
            self.val_writer = SummaryWriter(log_dir + '_val')

            return checkpoint['score']
        
        else:

            self.init_tensorboard()
        
            return MAX

    """
    ================================================================================================
    Main Training Function
    ================================================================================================
    """
    def main(self):

        # data loader
        train_dl, val_dl = self.init_dl()

        # load model parameter and get checkpoint
        best_score = self.load_model()

        # main loop
        count = 0
        for epoch_index in range(self.begin, EPOCH + 1):
            
            # get and save metrics
            print('Training: ')
            metrics_train = self.training(epoch_index, train_dl)

            self.save_metrics(epoch_index, 'train', metrics_train)
            self.save_images(epoch_index, 'train', train_dl)

            # check performance for every 10 epochs
            if epoch_index == 1 or epoch_index % STRIDE == 0:
                
                # get and save metrics
                print('\n' + 'Validation: ')
                metrics_val = self.validation(epoch_index, val_dl)

                score = self.save_metrics(epoch_index, 'val', metrics_val)
                self.save_images(epoch_index, 'val', val_dl)

                # save model
                if not math.isnan(score):
                    best_score = min(best_score, score)
                self.save_model(epoch_index, score, (best_score == score))

                # early stop
                if score == best_score:
                    count = 0
                elif count < 5:
                    count += 1
                elif count == 5:
                    print('\n' + 'early stop' + '\n')
                    break

                print()
        
        self.train_writer.close()
        self.val_writer.close()

    """
    ================================================================================================
    Training Loop
    ================================================================================================
    """
    def training(self, epoch_index, train_dl):
        
        # training state
        self.gen.train()
        self.dis.train() 

        # create buffer for matrics
        metrics = torch.zeros(METRICS, len(train_dl), device = self.device)

        space = "{:3}{:3}{:3}"
        progress = tqdm(enumerate(train_dl), total = len(train_dl), leave = True,
                        bar_format = '{l_bar}{bar:15}{r_bar}{bar:-10b}')
        for batch_index, batch_tuple in progress:

            # get samples
            (real1_t, real2_t) = batch_tuple
            real1_g = real1_t.to(self.device)
            real2_g = real2_t.to(self.device)

            # save max & min
            self.max1 = real1_g.max()
            self.min1 = real1_g.min()
            self.max2 = real2_g.max()
            self.min2 = real2_g.min()

            # normalization
            real1_g = (real1_g - self.min1) / (self.max1 - self.min1 + 1e-6)
            real2_g = (real2_g - self.min2) / (self.max2 - self.min2 + 1e-6)

            # ground truth
            valid = torch.ones(real2_g.size(0), 1, 12, 12, requires_grad = False, device = self.device)
            fake = torch.zeros(real2_g.size(0), 1, 12, 12, requires_grad = False, device = self.device)

            # get output of model
            fake2_g = self.gen(real1_g)

            """
            ========================================================================================
            Train Generator
            ========================================================================================
            """
            # refresh gradient
            self.optimizer_gen.zero_grad()

            # get pixelwise loss
            loss_pix = get_pix_loss(fake2_g, real2_g)

            # get adversarial loss
            loss_adv = get_adv_loss(self.dis(fake2_g, real1_g), valid)        

            # total loss
            loss_gen = loss_pix + loss_adv

            # update parameters
            loss_gen.backward()
            self.optimizer_gen.step()

            """
            ========================================================================================
            Train Discriminator
            ========================================================================================
            """
            # refresh gradient
            self.optimizer_dis.zero_grad()

            # real loss
            loss_real = get_adv_loss(self.dis(real2_g, real1_g), valid)

            # fake loss
            loss_fake = get_adv_loss(self.dis(fake2_g.detach(), real1_g), fake)

            # total loss
            loss_dis = (loss_real + loss_fake) / 2

            # update parameters
            loss_dis.backward()
            self.optimizer_dis.step()

            """
            ========================================================================================
            Get and Save Metrics
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

            # validation state
            self.gen.eval()
            self.dis.eval() 

            # create buffer for matrics
            metrics = torch.zeros(METRICS, len(val_dl), device = self.device)
        
            space = "{:3}{:3}{:3}"
            progress = tqdm(enumerate(val_dl), total = len(val_dl), leave = True,
                            bar_format = '{l_bar}{bar:15}{r_bar}{bar:-10b}')
            for batch_index, batch_tuple in progress:

                # get samples
                (real1_t, real2_t) = batch_tuple
                real1_g = real1_t.to(self.device)
                real2_g = real2_t.to(self.device)

                # save max & min
                self.max1 = real1_g.max()
                self.min1 = real1_g.min()
                self.max2 = real2_g.max()
                self.min2 = real2_g.min()

                # normalization
                real1_g = (real1_g - self.min1) / (self.max1 - self.min1)
                real2_g = (real2_g - self.min2) / (self.max2 - self.min2)

                # ground truth
                valid = torch.ones(real2_g.size(0), 1, 12, 12, requires_grad = False, device = self.device)
                fake = torch.zeros(real2_g.size(0), 1, 12, 12, requires_grad = False, device = self.device)

                # get output of model
                fake2_g = self.gen(real1_g)

                """
                ========================================================================================
                Validate Generator
                ========================================================================================
                """
                # get pixelwise loss
                loss_pix = get_pix_loss(fake2_g, real2_g)

                # get adversarial loss
                loss_adv = get_adv_loss(self.dis(fake2_g, real1_g), valid)        

                # total loss
                loss_gen = loss_pix + loss_adv
        
                """
                ========================================================================================
                Validate Discriminator
                ========================================================================================
                """
                # real loss
                loss_real2 = get_adv_loss(self.dis(real2_g, real1_g), valid)

                # fake loss
                loss_fake2 = get_adv_loss(self.dis(fake2_g.detach(), real1_g), fake)

                # total loss
                loss_dis = (loss_real2 + loss_fake2) / 2

                """
                ========================================================================================
                Get and Save Metrics
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

                progress.set_description('Epoch [' + space.format(epoch_index, ' / ', EPOCH) + ']')
                progress.set_postfix(loss_gen = loss_gen.item(), loss_dis = loss_dis.item())

            return metrics.to('cpu')
    
    """
    ================================================================================================
    Save Metrics for Whole Epoch
    ================================================================================================
    """ 
    def save_metrics(self, epoch_index, mode, metrics_t):

        # copy metrics
        metrics_a = metrics_t.detach().numpy().mean(axis = 1)

        # create a dictionary to save metrics
        metrics_dict = {}
        metrics_dict['Loss/Gen'] = metrics_a[METRICS_GEN]
        metrics_dict['Loss/Dis'] = metrics_a[METRICS_DIS]
        metrics_dict['Metrics/PSNR'] = metrics_a[METRICS_PSNR]
        metrics_dict['Metrics/SSIM'] = metrics_a[METRICS_SSIM]

        # save metrics to tensorboard writer
        writer = getattr(self, mode + '_writer')
        for key, value in metrics_dict.items():

            writer.add_scalar(key, value.item(), epoch_index)
        
        # refresh tensorboard writer
        writer.flush()

        return metrics_dict['Loss/Gen']

    """
    ================================================================================================
    Save Some Image to Checking
    ================================================================================================
    """ 
    def save_images(self, epoch_index, mode, dataloader):

        # validation state
        self.gen.eval()

        # get random image index and load sample
        (real1_t, real2_t) = dataloader.dataset[90]
        real1_g = real1_t.to(self.device).unsqueeze(0)
        real2_g = real2_t.to(self.device).unsqueeze(0)

        # get predict
        fake2_g = self.gen(real1_g)

        real1_a = real1_g.to('cpu').detach().numpy()[:, 3, :, :]
        real2_a = real2_g.to('cpu').detach().numpy()[0]
        fake2_a = fake2_g.to('cpu').detach().numpy()[0]
        
        real1_a -= real1_a.min()
        real1_a /= real1_a.max()
        
        real2_a -= real2_a.min()
        real2_a /= real2_a.max()
        
        print(real1_a.min(), real1_a.max())
        print(real2_a.min(), real2_a.max())
        print(fake2_a.min(), fake2_a.max())

        # save image to tensorboard writer
        writer = getattr(self, mode + '_writer')
        writer.add_image(mode + '/MR', real1_a, epoch_index, dataformats = 'CHW')
        writer.add_image(mode + '/rCT', real2_a, epoch_index, dataformats = 'CHW')
        writer.add_image(mode + '/sCT', fake2_a, epoch_index, dataformats = 'CHW')

        # refresh tensorboard writer
        writer.flush()

    """
    ================================================================================================
    Save Model
    ================================================================================================
    """ 
    def save_model(self, epoch_index, score, is_best):

        # prepare model state dict
        gen = self.gen
        dis = self.dis

        opt_gen = self.optimizer_gen
        opt_dis = self.optimizer_dis

        state = {
            'time': self.time,
            'gen_state': gen.state_dict(),
            'gen_name': type(gen).__name__,
            'dis_state': dis.state_dict(),
            'dis_name': type(dis).__name__,
            'optimizer_gen_state': opt_gen.state_dict(),
            'optimizer_gen_name': type(opt_gen).__name__,
            'optimizer_dis_state': opt_dis.state_dict(),
            'optimizer_dis_name': type(opt_dis).__name__,
            'epoch': epoch_index,
            'score': score,
        }

        # save model
        model_path = os.path.join(RESULTS_PATH, 'Model', self.time + '.pt')
        torch.save(state, model_path)

        if is_best:

            # save best model
            best_path = os.path.join(RESULTS_PATH, 'Model', self.time + '.best.pt')
            torch.save(state, best_path)


"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':

    Training().main()