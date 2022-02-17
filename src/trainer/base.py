import os
import math
import time, datetime

import cv2
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.autograd as autograd
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from ..util.dnd_submission.bundle_submissions import bundle_submissions_srgb
from ..util.dnd_submission.dnd_denoise import denoise_srgb
from ..util.dnd_submission.pytorch_wrapper import pytorch_denoiser

from ..loss import Loss
from ..datahandler import get_dataset_class
from ..util.file_manager import FileManager
from ..util.logger import Logger
from ..util.util import human_format, rot_hflip_img, psnr, ssim, tensor2np, imread_tensor 
from ..util.util import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling


status_len = 13

class BaseTrainer():
    '''
    Base trainer classs to implement other trainer classes.
    below function would be implemented.
    '''
    def test(self):
        raise NotImplementedError('define this function for each trainer')
    def validation(self):
        raise NotImplementedError('define this function for each trainer')
    def _set_module(self):
        # return dict form with model name.
        raise NotImplementedError('define this function for each trainer')
    def _set_optimizer(self):
        # return dict form with each coresponding model name.
        raise NotImplementedError('define this function for each trainer')
    def _forward_fn(self, module, loss, data):
        # forward with model, loss function and data.
        # return output of loss function.
        raise NotImplementedError('define this function for each trainer')

    #----------------------------#
    #    Train/Test functions    #
    #----------------------------#   
    def __init__(self, cfg):
        self.session_name = cfg['session_name']

        self.checkpoint_folder = 'checkpoint'

        # get file manager and logger class
        self.file_manager = FileManager(self.session_name)
        self.logger = Logger()
        
        self.cfg = cfg
        self.train_cfg = cfg['training']
        self.val_cfg   = cfg['validation']
        self.test_cfg  = cfg['test']
        self.ckpt_cfg  = cfg['checkpoint']

    def train(self):
        # initializing
        self._before_train()

        # warmup
        if self.epoch == 1 and self.train_cfg['warmup']:
            self._warmup()

        # training
        for self.epoch in range(self.epoch, self.max_epoch+1):
            self._before_epoch()
            self._run_epoch()
            self._after_epoch()
        
        self._after_train()

    def _warmup(self):
        self._set_status('warmup')

        # make dataloader iterable.
        self.train_dataloader_iter = {}
        for key in self.train_dataloader:
            self.train_dataloader_iter[key] = iter(self.train_dataloader[key])

        warmup_iter = self.train_cfg['warmup_iter']
        if warmup_iter > self.max_iter:
            self.logger.info('currently warmup support 1 epoch as maximum. warmup iter is replaced to 1 epoch iteration. %d -> %d' \
                % (warmup_iter, self.max_iter))
            warmup_iter = self.max_iter

        for self.iter in range(1, warmup_iter+1):
            self._adjust_warmup_lr(warmup_iter)
            self._before_step()
            self._run_step()
            self._after_step()

    def _before_test(self):
        # initialing
        self.module = self._set_module()

        # load checkpoint file
        ckpt_epoch = self._find_last_epoch() if self.cfg['ckpt_epoch'] == -1 else self.cfg['ckpt_epoch']
        ckpt_name  = self.test_cfg['ckpt_name'] if 'ckpt_name' in self.test_cfg else None
        self.load_checkpoint(ckpt_epoch, name=ckpt_name)
        self.epoch = self.cfg['ckpt_epoch'] # for print or saving file name.

        # test dataset loader
        self.test_dataloader = self._set_dataloader(self.test_cfg, batch_size=1, shuffle=False, num_workers=self.cfg['thread'])

        # wrapping and device setting
        if self.cfg['gpu'] != 'None':
            # model to GPU
            self.model = {key: nn.DataParallel(self.module[key]).cuda() for key in self.module}
        else:
            self.model = {key: nn.DataParallel(self.module[key]) for key in self.module}

        # evaluation mode and set status
        self._eval_mode()
        self._set_status('test %03d'%self.epoch)

        # start message
        self.logger.highlight(self.logger.get_start_msg())

        # set denoiser
        self._set_denoiser()
        
        # wrapping denoiser w/ self_ensemble
        if self.cfg['self_en']:
            # (warning) self_ensemble cannot be applied with multi-input model
            denoiser_fn = self.denoiser
            self.denoiser = lambda *input_data: self.self_ensemble(denoiser_fn, *input_data)

        # wrapping denoiser w/ crop test
        if 'crop' in self.cfg['test']:
            # (warning) self_ensemble cannot be applied with multi-input model
            denoiser_fn = self.denoiser
            self.denoiser = lambda *input_data: self.crop_test(denoiser_fn, *input_data, size=self.cfg['test']['crop'])
            
    def _before_train(self):
        # cudnn
        torch.backends.cudnn.benchmark = False

        # initialing
        self.module = self._set_module()

        # training dataset loader
        self.train_dataloader = self._set_dataloader(self.train_cfg, batch_size=self.train_cfg['batch_size'], shuffle=True, num_workers=self.cfg['thread'])

        # validation dataset loader
        if self.val_cfg['val']:
            self.val_dataloader = self._set_dataloader(self.val_cfg, batch_size=1, shuffle=False, num_workers=self.cfg['thread'])

        # other configuration
        self.max_epoch = self.train_cfg['max_epoch']
        self.epoch = self.start_epoch = 1
        max_len = self.train_dataloader['dataset'].dataset.__len__() # base number of iteration works for dataset named 'dataset'
        self.max_iter = math.ceil(max_len / self.train_cfg['batch_size'])

        self.loss = Loss(self.train_cfg['loss'], self.train_cfg['tmp_info'])
        self.loss_dict = {'count':0}
        self.tmp_info = {}
        self.loss_log = []

        # set optimizer
        self.optimizer = self._set_optimizer()
        for opt in self.optimizer.values():
            opt.zero_grad(set_to_none=True)

        # resume
        if self.cfg["resume"]:
            # find last checkpoint
            load_epoch = self._find_last_epoch()

            # load last checkpoint
            self.load_checkpoint(load_epoch)
            self.epoch = load_epoch+1

            # logger initialization
            self.logger = Logger((self.max_epoch, self.max_iter), log_dir=self.file_manager.get_dir(''), log_file_option='a')
        else:
            # logger initialization
            self.logger = Logger((self.max_epoch, self.max_iter), log_dir=self.file_manager.get_dir(''), log_file_option='w')

        # tensorboard
        tboard_time = datetime.datetime.now().strftime('%m-%d-%H-%M')
        self.tboard = SummaryWriter(log_dir=self.file_manager.get_dir('tboard/%s'%tboard_time))

        # wrapping and device setting
        if self.cfg['gpu'] != 'None':
            # model to GPU
            self.model = {key: nn.DataParallel(self.module[key]).cuda() for key in self.module}
            # optimizer to GPU
            for optim in self.optimizer.values():
                for state in optim.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
        else:
            self.model = {key: nn.DataParallel(self.module[key]) for key in self.module}

        # start message
        self.logger.info(self.summary())
        self.logger.start((self.epoch-1, 0))
        self.logger.highlight(self.logger.get_start_msg())

    def _after_train(self):
        # finish message
        self.logger.highlight(self.logger.get_finish_msg())

    def _before_epoch(self):
        self._set_status('epoch %03d/%03d'%(self.epoch, self.max_epoch))

        # make dataloader iterable.
        self.train_dataloader_iter = {}
        for key in self.train_dataloader:
            self.train_dataloader_iter[key] = iter(self.train_dataloader[key])

        # model training mode
        self._train_mode()

    def _run_epoch(self):
        for self.iter in range(1, self.max_iter+1):
            self._before_step()
            self._run_step()
            self._after_step()

    def _after_epoch(self):
        # save checkpoint
        if self.epoch >= self.ckpt_cfg['start_epoch']:
            if (self.epoch-self.ckpt_cfg['start_epoch'])%self.ckpt_cfg['interval_epoch'] == 0:
                self.save_checkpoint()

        # validation
        if self.val_cfg['val']:
            if self.epoch >= self.val_cfg['start_epoch'] and self.val_cfg['val']:
                if (self.epoch-self.val_cfg['start_epoch']) % self.val_cfg['interval_epoch'] == 0:
                    self._eval_mode()
                    self._set_status('val %03d'%self.epoch)
                    self.validation()

    def _before_step(self):
        pass

    def _run_step(self):
        # get data (data should be dictionary of Tensors)
        data = {}
        for key in self.train_dataloader_iter:
            data[key] = next(self.train_dataloader_iter[key])

        # to device
        if self.cfg['gpu'] != 'None':
            for dataset_key in data:
                for key in data[dataset_key]:
                    data[dataset_key][key] = data[dataset_key][key].cuda()

        # forward, cal losses, backward)
        losses, tmp_info = self._forward_fn(self.model, self.loss, data)
        losses   = {key: losses[key].mean()   for key in losses}
        tmp_info = {key: tmp_info[key].mean() for key in tmp_info}

        # backward
        total_loss = sum(v for v in losses.values())
        total_loss.backward()

        # optimizer step
        for opt in self.optimizer.values():
            opt.step()

        # zero grad
        for opt in self.optimizer.values():
            opt.zero_grad(set_to_none=True) 

        # save losses and tmp_info
        for key in losses:
            if key != 'count':
                if key in self.loss_dict:
                    self.loss_dict[key] += float(losses[key])
                else:
                    self.loss_dict[key] = float(losses[key])
        for key in tmp_info:
            if key in self.tmp_info:
                self.tmp_info[key] += float(tmp_info[key])
            else:
                self.tmp_info[key] = float(tmp_info[key])
        self.loss_dict['count'] += 1

    def _after_step(self):
        # adjust learning rate
        self._adjust_lr()

        # print loss
        if (self.iter%self.cfg['log']['interval_iter']==0 and self.iter!=0) or (self.iter == self.max_iter):
            self.print_loss()

        # print progress
        self.logger.print_prog_msg((self.epoch-1, self.iter-1))

    def test_dataloader_process(self, dataloader, add_con=0., floor=False, img_save=True, img_save_path=None, info=True):
        '''
        do test or evaluation process for each dataloader
        include following steps:
            1. denoise image
            2. calculate PSNR & SSIM
            3. (optional) save denoised image
        Args:
            dataloader : dataloader to be tested.
            add_con : add constant to denoised image.
            floor : floor denoised image. (default range is [0, 255])
            img_save : whether to save denoised and clean images.
            img_save_path (optional) : path to save denoised images.
            info (optional) : whether to print info.
        Returns:
            psnr : total PSNR score of dataloaer results or None (if clean image is not available)
            ssim : total SSIM score of dataloder results or None (if clean image is not available)
        '''
        # make directory
        self.file_manager.make_dir(img_save_path)

        # test start
        psnr_sum = 0.
        ssim_sum = 0.
        count = 0
        for idx, data in enumerate(dataloader):
            # to device
            if self.cfg['gpu'] != 'None':
                for key in data:
                    data[key] = data[key].cuda()

            # forward
            input_data = [data[arg] for arg in self.cfg['model_input']]
            denoised_image = self.denoiser(*input_data)

            # add constant and floor (if floor is on)
            denoised_image += add_con
            if floor: denoised_image = torch.floor(denoised_image)

            # evaluation
            if 'clean' in data:
                psnr_value = psnr(denoised_image, data['clean'])
                ssim_value = ssim(denoised_image, data['clean'])

                psnr_sum += psnr_value
                ssim_sum += ssim_value
                count += 1

            # image save
            if img_save:
                # to cpu
                if 'clean' in data:
                    clean_img = data['clean'].squeeze(0).cpu()
                if 'real_noisy' in self.cfg['model_input']: noisy_img = data['real_noisy']
                elif 'syn_noisy' in self.cfg['model_input']: noisy_img = data['syn_noisy']
                elif 'noisy' in self.cfg['model_input']: noisy_img = data['noisy']
                else: noisy_img = None
                if noisy_img is not None: noisy_img = noisy_img.squeeze(0).cpu()
                denoi_img = denoised_image.squeeze(0).cpu()

                # write psnr value on file name
                denoi_name = '%04d_DN_%.2f'%(idx, psnr_value) if 'clean' in data else '%04d_DN'%idx

                # imwrite
                if 'clean' in data:         self.file_manager.save_img_tensor(img_save_path, '%04d_CL'%idx, clean_img)
                if noisy_img is not None: self.file_manager.save_img_tensor(img_save_path, '%04d_N'%idx, noisy_img)
                self.file_manager.save_img_tensor(img_save_path, denoi_name, denoi_img)

            # procedure log msg
            if info:
                if 'clean' in data:
                    self.logger.note('[%s] testing... %04d/%04d. PSNR : %.2f dB'%(self.status, idx, dataloader.__len__(), psnr_value), end='\r')
                else:
                    self.logger.note('[%s] testing... %04d/%04d.'%(self.status, idx, dataloader.__len__()), end='\r')

        # final log msg
        if count > 0:
            self.logger.val('[%s] Done! PSNR : %.2f dB, SSIM : %.3f'%(self.status, psnr_sum/count, ssim_sum/count))
        else:
            self.logger.val('[%s] Done!'%self.status)

        # return
        if count == 0:
            return None, None
        else:
            return psnr_sum/count, ssim_sum/count

    def test_img(self):
        '''
        Inference a single image with measuring process time.
        '''
        # load image
        noisy = imread_tensor(self.cfg['test_img'])
        noisy = noisy.unsqueeze(0).float()

        # to device
        if self.cfg['gpu'] != 'None':
            noisy = noisy.cuda()

        # forward
        torch.cuda.synchronize()
        start = time.time()
        denoised = self.denoiser(noisy)
        torch.cuda.synchronize()
        end = time.time()

        # post-process
        denoised += self.test_cfg['add_con']
        if self.test_cfg['floor']: denoised = torch.floor(denoised)

        # save image
        denoised = tensor2np(denoised)
        denoised = denoised.squeeze(0)
        
        name = self.cfg['test_img'].split('/')[-1].split('.')[0]
        self.file_manager.save_img_numpy('./', name+'_DN.png', denoised)

        # print message
        self.logger.note('[%s] Done! Time : %.3f sec'%(self.status, end-start))

    def test_DND(self, img_save_path):
        '''
        Benchmarking DND dataset.
        '''
        # make directories for .mat & image saving 
        self.file_manager.make_dir(img_save_path)
        self.file_manager.make_dir(img_save_path + '/mat')
        if self.test_cfg['save_image']: self.file_manager.make_dir(img_save_path + '/img')

        def wrap_denoiser(Inoisy, nlf, idx, kidx):
            noisy = 255 * torch.from_numpy(Inoisy)

            # to device
            if self.cfg['gpu'] != 'None':
                noisy = noisy.cuda()

            noisy = autograd.Variable(noisy)

            # processing
            noisy = noisy.permute(2,0,1)
            noisy = self.test_dataloader['dataset'].dataset._pre_processing({'real_noisy': noisy})['real_noisy']

            noisy = noisy.view(1,noisy.shape[0], noisy.shape[1], noisy.shape[2])

            denoised = self.denoiser(noisy)

            denoised += self.test_cfg['add_con']
            if self.test_cfg['floor']: denoised = torch.floor(denoised)

            denoised = denoised[0,...].cpu().numpy()
            denoised = np.transpose(denoised, [1,2,0])

            # image save
            if self.test_cfg['save_image'] and False:
                self.file_manager.save_img_numpy(img_save_path+'/img', '%02d_%02d_N'%(idx, kidx),  255*Inoisy)
                self.file_manager.save_img_numpy(img_save_path+'/img', '%02d_%02d_DN'%(idx, kidx), denoised)

            return denoised / 255

        denoise_srgb(wrap_denoiser, './dataset/DND/dnd_2017', self.file_manager.get_dir(img_save_path+'/mat'))

        bundle_submissions_srgb(self.file_manager.get_dir(img_save_path+'/mat'))

        # info 
        self.logger.val('[%s] Done!'%self.status)

    def _set_denoiser(self):
        if hasattr(self.model['denoiser'].module, 'denoise'):
            self.denoiser = self.model['denoiser'].module.denoise
        else:
            self.denoiser = self.model['denoiser'].module

    @torch.no_grad()
    def crop_test(self, fn, x, size=512, overlap=0):
        '''
        crop test image and inference due to memory problem
        '''
        b,c,h,w = x.shape
        denoised = torch.zeros_like(x)
        for i in range(0,h,size-overlap):
            for j in range(0,w,size-overlap):
                end_i = min(i+size, h)
                end_j = min(j+size, w)
                x_crop = x[...,i:end_i,j:end_j]
                denoised_crop = fn(x_crop)
                
                start_i = overlap if i != 0 else 0
                start_j = overlap if j != 0 else 0

                denoised[..., i+start_i:end_i, j+start_j:end_j] = denoised_crop[..., start_i:, start_j:]

        return denoised

    @torch.no_grad()
    def self_ensemble(self, fn, x):
        '''
        Geomery self-ensemble function
        Note that in this function there is no gradient calculation.
        Args:
            fn : denoiser function
            x : input image
        Return:
            result : self-ensembled image
        '''
        result = torch.zeros_like(x)

        for i in range(8):
            tmp = fn(rot_hflip_img(x, rot_times=i%4, hflip=i//4))
            tmp = rot_hflip_img(tmp, rot_times=4-i%4)
            result += rot_hflip_img(tmp, hflip=i//4)
        return result / 8

    #----------------------------#
    #      Utility functions     #
    #----------------------------# 
    def print_loss(self):
        temporal_loss = 0.
        for key in self.loss_dict:
            if key != 'count':
                    temporal_loss += self.loss_dict[key]/self.loss_dict['count']
        self.loss_log += [temporal_loss]
        if len(self.loss_log) > 100: self.loss_log.pop(0)

        # print status and learning rate
        loss_out_str = '[%s] %04d/%04d, lr:%s ∣ '%(self.status, self.iter, self.max_iter, "{:.1e}".format(self._get_current_lr()))
        global_iter = (self.epoch-1)*self.max_iter + self.iter

        # print losses
        avg_loss = np.mean(self.loss_log)
        loss_out_str += 'avg_100 : %.3f ∣ '%(avg_loss)
        self.tboard.add_scalar('loss/avg_100', avg_loss, global_iter)

        for key in self.loss_dict:
            if key != 'count':
                loss = self.loss_dict[key]/self.loss_dict['count']
                loss_out_str += '%s : %.3f ∣ '%(key, loss)
                self.tboard.add_scalar('loss/%s'%key, loss, global_iter)
                self.loss_dict[key] = 0.

        # print temporal information
        if len(self.tmp_info) > 0:
            loss_out_str += '\t['
            for key in self.tmp_info:
                loss_out_str += '  %s : %.2f'%(key, self.tmp_info[key]/self.loss_dict['count'])
                self.tmp_info[key] = 0.
            loss_out_str += ' ]'

        # reset
        self.loss_dict['count'] = 0
        self.logger.info(loss_out_str)

    def save_checkpoint(self):
        checkpoint_name = self._checkpoint_name(self.epoch)
        torch.save({'epoch': self.epoch,
                    'model_weight': {key:self.model[key].module.state_dict() for key in self.model},
                    'optimizer_weight': {key:self.optimizer[key].state_dict() for key in self.optimizer}},
                    os.path.join(self.file_manager.get_dir(self.checkpoint_folder), checkpoint_name))

    def load_checkpoint(self, load_epoch=0, name=None):
        if name is None:
            # if scratch, return
            if load_epoch == 0: return
            # load from local checkpoint folder
            file_name = os.path.join(self.file_manager.get_dir(self.checkpoint_folder), self._checkpoint_name(load_epoch))
        else:
            # load from global checkpoint folder
            file_name = os.path.join('./ckpt', name)
        
        # check file exist
        assert os.path.isfile(file_name), 'there is no checkpoint: %s'%file_name

        # load checkpoint (epoch, model_weight, optimizer_weight)
        saved_checkpoint = torch.load(file_name)
        self.epoch = saved_checkpoint['epoch']
        for key in self.module:
            self.module[key].load_state_dict(saved_checkpoint['model_weight'][key])
        if hasattr(self, 'optimizer'):
            for key in self.optimizer:
                self.optimizer[key].load_state_dict(saved_checkpoint['optimizer_weight'][key])

    def _checkpoint_name(self, epoch):
        return self.session_name + '_%03d'%epoch + '.pth'

    def _find_last_epoch(self):
        checkpoint_list = os.listdir(self.file_manager.get_dir(self.checkpoint_folder))
        epochs = [int(ckpt.replace('%s_'%self.session_name, '').replace('.pth', '')) for ckpt in checkpoint_list]
        assert len(epochs) > 0, 'There is no resumable checkpoint on session %s.'%self.session_name
        return max(epochs)

    def _get_current_lr(self):
        for first_optim in self.optimizer.values():
            for param_group in first_optim.param_groups:
                return param_group['lr']

    def _set_dataloader(self, dataset_cfg, batch_size, shuffle, num_workers):
        dataloader = {}
        dataset_dict = dataset_cfg['dataset']
        if not isinstance(dataset_dict, dict):
            dataset_dict = {'dataset': dataset_dict}

        for key in dataset_dict:
            args = dataset_cfg[key + '_args']
            dataset = get_dataset_class(dataset_dict[key])(**args)
            dataloader[key] = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False)

        return dataloader

    def _set_one_optimizer(self, opt, parameters, lr):
        lr = float(self.train_cfg['init_lr'])

        if opt['type'] == 'SGD':
            return optim.SGD(parameters, lr=lr, momentum=float(opt['SGD']['momentum']), weight_decay=float(opt['SGD']['weight_decay']))
        elif opt['type'] == 'Adam':
            return optim.Adam(parameters, lr=lr, betas=opt['Adam']['betas'])
        elif opt['type'] == 'AdamW':
            return optim.Adam(parameters, lr=lr, betas=opt['AdamW']['betas'], weight_decay=float(opt['AdamW']['weight_decay']))
        else:
            raise RuntimeError('ambiguious optimizer type: {}'.format(opt['type']))

    def _adjust_lr(self):
        sched = self.train_cfg['scheduler']

        if sched['type'] == 'step':
            '''
            step decreasing scheduler
            Args:
                step_size: step size(epoch) to decay the learning rate
                gamma: decay rate
            '''
            if self.iter == self.max_iter:
                args = sched['step']
                if self.epoch % args['step_size'] == 0:
                    for optimizer in self.optimizer.values():
                        lr_before = optimizer.param_groups[0]['lr']
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr_before * float(args['gamma'])
        elif sched['type'] == 'linear':
            '''
            linear decreasing scheduler
            Args:
                step_size: step size(epoch) to decrease the learning rate
                gamma: decay rate for reset learning rate
            '''
            args = sched['linear']
            if not hasattr(self, 'reset_lr'):
                self.reset_lr = float(self.train_cfg['init_lr']) * float(args['gamma'])**((self.epoch-1)//args['step_size'])

            # reset lr to initial value
            if self.epoch % args['step_size'] == 0 and self.iter == self.max_iter:
                self.reset_lr = float(self.train_cfg['init_lr']) * float(args['gamma'])**(self.epoch//args['step_size'])
                for optimizer in self.optimizer.values():
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = self.reset_lr
            # linear decaying
            else:
                ratio = ((self.epoch + (self.iter)/self.max_iter - 1) % args['step_size']) / args['step_size']
                curr_lr = (1-ratio) * self.reset_lr
                for optimizer in self.optimizer.values():
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = curr_lr
        else:
            raise RuntimeError('ambiguious scheduler type: {}'.format(sched['type']))

    def _adjust_warmup_lr(self, warmup_iter):
        init_lr = float(self.train_cfg['init_lr'])
        warmup_lr = init_lr * self.iter / warmup_iter

        for optimizer in self.optimizer.values():
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

    def _train_mode(self):
        for key in self.model:
            self.model[key].train()

    def _eval_mode(self):
        for key in self.model:
            self.model[key].eval()

    def _set_status(self, status:str):
        assert len(status) <= status_len, 'status string cannot exceed %d characters, (now %d)'%(status_len, len(status))

        if len(status.split(' ')) > 1:
            s0, s1 = status.split(' ')
            self.status = '%s '%s0.rjust(status_len//2) + \
                          '%s'%s1.ljust((status_len+1)//2)
        else:
            self.status = status.ljust(status_len)

    def summary(self):
        summary = ''

        summary += '-'*100 + '\n'
        # model
        for k, v in self.module.items():
            # get parameter number
            param_num = sum(p.numel() for p in v.parameters())

            # get information about architecture and parameter number
            summary += '[%s] paramters: %s -->'%(k, human_format(param_num)) + '\n'
            summary += str(v) + '\n\n'
        
        # optim

        # Hardware

        summary += '-'*100 + '\n'

        return summary
