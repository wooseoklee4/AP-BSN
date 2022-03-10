import random, os
from typing import Tuple

import cv2
import numpy as np
from scipy.io import savemat
import torch
from torch.utils.data import Dataset


from ..util.util import rot_hflip_img, tensor2np, np2tensor, mean_conv2d

    
class DenoiseDataSet(Dataset):
    def __init__(self, add_noise:str=None, crop_size:list=None, aug:list=None, n_repeat:int=1, n_data:int=None, ratio_data:float=None) -> None:
        '''
        Base denoising dataset class for various dataset.

        to build custom dataset class, below functions must be implemented in the inherited class. (or see other dataset class already implemented.)
            - self._scan(self) : scan image data & save its paths. (saved to self.img_paths)
            - self._load_data(self, data_idx) : load single paired data from idx as a form of dictionary.

        Args:
            add_noise (str)     : configuration of additive noise to synthesize noisy image. (see _add_noise() for more details.)
            crop_size (list)    : crop size, e.g. [W] or [H, W] and no crop if None
            aug (list)          : list of data augmentations (see _augmentation() for more details.)
            n_repeat (int)      : number of repeat for each data.
            n_data (int)        : number of data to be used. (default: None = all data)
            ratio_data (float)  : ratio of data to be used. (activated when n_data=None, default: None = all data)
        '''
        self.dataset_dir = './dataset'
        if not os.path.isdir(self.dataset_dir):
            raise Exception('dataset directory is not exist')
        
        # parse additive noise argument
        self.add_noise_type, self.add_noise_opt, self.add_noise_clamp = self._parse_add_noise(add_noise)

        # set parameters for dataset.
        self.crop_size = crop_size
        self.aug = aug
        self.n_repeat = n_repeat

        # scan all data and fill in self.img_paths
        self.img_paths = []
        self._scan()
        if len(self.img_paths) > 0:
            if self.img_paths[0].__class__.__name__ in ['int', 'str', 'float']:
                self.img_paths.sort()

        # set data amount
        if n_data is not None:       self.n_data = n_data
        elif ratio_data is not None: self.n_data = int(ratio_data * len(self.img_paths))
        else:                        self.n_data = len(self.img_paths)

    def __len__(self):
        return self.n_data * self.n_repeat

    def __getitem__(self, idx):
        '''
        final dictionary shape of data:
        {'clean', 'syn_noisy', 'real_noisy', 'noisy (any of real[first priority] and syn)', etc}
        '''
        # calculate data index
        data_idx = idx % self.n_data

        # load data
        data = self._load_data(data_idx)

        # pre-processing (currently only crop)
        data = self._pre_processing(data)

        # synthesize additive noise 
        if self.add_noise_type is not None:
            if 'clean' in data:
                syn_noisy_img, nlf = self._add_noise(data['clean'], self.add_noise_type, self.add_noise_opt, self.add_noise_clamp)
                data['syn_noisy'] = syn_noisy_img
                data['nlf'] = nlf
            elif 'real_noisy' in data:
                syn_noisy_img, nlf = self._add_noise(data['real_noisy'], self.add_noise_type, self.add_noise_opt, self.add_noise_clamp)
                data['syn_noisy'] = syn_noisy_img
                data['nlf'] = nlf
            else:
                raise RuntimeError('there is no clean or real image to synthesize. (synthetic noise type: %s)'%self.add_noise_type)

        # data augmentation
        if self.aug is not None:
            data = self._augmentation(data, self.aug)

        # add general label 'noisy' to use any of real_noisy or syn_noisy (real first)
        if 'real_noisy' in data or 'syn_noisy' in data:
            data['noisy'] = data['real_noisy'] if 'real_noisy' in data else data['syn_noisy']

        return data

    def _scan(self):
        raise NotImplementedError
        # TODO fill in self.img_paths (include path from project directory)

    def _load_data(self, data_idx):
        raise NotImplementedError
        # TODO load possible data as dictionary
        # dictionary key list :
        #   'clean' : clean image without noise (gt or anything).
        #   'real_noisy' : real noisy image or already synthesized noisy image.
        #   'instances' : any other information of capturing situation.

    #----------------------------#
    #  Image handling functions  #
    #----------------------------#
    def _load_img(self, img_name, as_gray=False):
        img = cv2.imread(img_name, 1)
        assert img is not None, "failure on loading image - %s"%img_name
        return self._load_img_from_np(img, as_gray, RGBflip=True)

    def _load_img_from_np(self, img, as_gray=False, RGBflip=False):
        # if color
        if len(img.shape) != 2:
            if as_gray:
                # follows definition of sRBG in terms of the CIE 1931 linear luminance.
                # because calculation opencv color conversion and imread grayscale mode is a bit different.
                # https://en.wikipedia.org/wiki/Grayscale
                img = np.average(img, axis=2, weights=[0.0722, 0.7152, 0.2126])
                img = np.expand_dims(img, axis=0)
            else:
                if RGBflip:
                    img = np.flip(img, axis=2)
                img = np.transpose(img, (2,0,1))
        # if gray
        else:                   
            img = np.expand_dims(img, axis=0)
        return torch.from_numpy(np.ascontiguousarray(img).astype(np.float32))

    def _pre_processing(self, data):
        # get a patch from image data
        if self.crop_size != None:
            data = self._get_patch(self.crop_size, data)
        return data

    def _get_patch(self, crop_size, data, rnd=True):
        # check all image size is same
        if 'clean' in data and 'real_noisy' in data:
            assert data['clean'].shape[1] == data['clean'].shape[1] and data['real_noisy'].shape[2] == data['real_noisy'].shape[2], \
            'img shape should be same. (%d, %d) != (%d, %d)' % (data['clean'].shape[1], data['clean'].shape[1], data['real_noisy'].shape[2], data['real_noisy'].shape[2])

        # get image shape and select random crop location
        if 'clean' in data:
            max_x = data['clean'].shape[2] - crop_size[0]
            max_y = data['clean'].shape[1] - crop_size[1]
        else:
            max_x = data['real_noisy'].shape[2] - crop_size[0]
            max_y = data['real_noisy'].shape[1] - crop_size[1]

        assert max_x >= 0
        assert max_y >= 0

        if rnd and max_x>0 and max_y>0:
            x = np.random.randint(0, max_x)
            y = np.random.randint(0, max_y)
        else:
            x, y = 0, 0

        # crop
        if 'clean' in data:
            data['clean'] = data['clean'][:, y:y+crop_size[1], x:x+crop_size[0]]
        if 'real_noisy' in data:
            data['real_noisy'] = data['real_noisy'][:, y:y+crop_size[1], x:x+crop_size[0]]
        
        return data

    def normalize_data(self, data, cuda=False):
        # for all image
        for key in data:
            if self._is_image_tensor(data[key]):
                data[key] = self.normalize(data[key], cuda)
        return data

    def inverse_normalize_data(self, data, cuda=False):
        # for all image
        for key in data:
            # is image 
            if self._is_image_tensor(data[key]):
                data[key] = self.inverse_normalize(data[key], cuda)
        return data

    def normalize(self, img, cuda=False):
        if img.shape[0] == 1:
            stds = self.gray_stds
            means = self.gray_means
        elif img.shape[0] == 3:
            stds = self.color_stds
            means = self.color_means
        else:
            raise RuntimeError('undefined image channel length : %d'%img.shape[0])
        
        if cuda:
            means, stds = means.cuda(), stds.cuda() 

        return (img-means) / stds

    def inverse_normalize(self, img, cuda=False):
        if img.shape[0] == 1:
            stds = self.gray_stds
            means = self.gray_means
        elif img.shape[0] == 3:
            stds = self.color_stds
            means = self.color_means
        else:
            raise RuntimeError('undefined image channel length : %d'%img.shape[0])
        
        if cuda:
            means, stds = means.cuda(), stds.cuda() 

        return (img*stds) + means

    def _parse_add_noise(self, add_noise_str:str) -> Tuple:
        '''
        noise_type-opt0:opt1:opt2-clamp
        '''
        if add_noise_str == 'bypass':
            return 'bypass', None, None
        elif add_noise_str != None:
            add_noise_type = add_noise_str.split('-')[0]
            add_noise_opt = [float(v) for v in add_noise_str.split('-')[1].split(':')]
            add_noise_clamp = len(add_noise_str.split('-'))>2 and add_noise_str.split('-')[2] == 'clamp'
            return add_noise_type, add_noise_opt, add_noise_clamp
        else:
            return None, None, None

    def _add_noise(self, clean_img:torch.Tensor, add_noise_type:str, opt:list, clamp:bool=False) -> torch.Tensor:
        '''
        add various noise to clean image.
        Args:
            clean_img (Tensor) : clean image to synthesize on
            add_noise_type : below types are available
            opt (list) : args for synthsize noise
            clamp (bool) : optional, clamp noisy image into [0,255]
        Return:
            synthesized_img
        Noise_types
            - bypass : bypass clean image
            - uni : uniform distribution noise from -opt[0] ~ opt[0]
            - gau : gaussian distribution noise with zero-mean & opt[0] variance
            - gau_blind : blind gaussian distribution with zero-mean, variance is uniformly selected from opt[0] ~ opt[1]
            - struc_gau : structured gaussian noise. gaussian filter is applied to above gaussian noise. opt[0] is variance of gaussian, opt[1] is window size and opt[2] is sigma of gaussian filter.
            - het_gau : heteroscedastic gaussian noise with indep weight:opt[0], dep weight:opt[1]
        '''
        nlf = None

        if add_noise_type == 'bypass':
            # bypass clean image
            synthesized_img = clean_img
        elif add_noise_type == 'uni':
            # add uniform noise
            synthesized_img = clean_img + 2*opt[0] * torch.rand(clean_img.shape) - opt[0]
        elif add_noise_type == 'gau':
            # add AWGN
            nlf = opt[0]
            synthesized_img = clean_img + torch.normal(mean=0., std=nlf, size=clean_img.shape)
        elif add_noise_type == 'gau_blind':
            # add blind gaussian noise
            nlf = random.uniform(opt[0], opt[1])
            synthesized_img = clean_img + torch.normal(mean=0., std=nlf, size=clean_img.shape)
        elif add_noise_type == 'struc_gau':
            # add structured gaussian noise (used in the paper "Noiser2Noise": https://arxiv.org/pdf/1910.11908.pdf)
            nlf = opt[0]
            gau_noise = torch.normal(mean=0., std=opt[0], size=clean_img.shape)
            struc_gau = mean_conv2d(gau_noise,  window_size=int(opt[1]), sigma=opt[2], keep_sigma=True)
            synthesized_img = clean_img + struc_gau
        elif add_noise_type == 'het_gau':
            # add heteroscedastic  guassian noise
            het_gau_std = (clean_img * (opt[0]**2) + torch.ones(clean_img.shape) * (opt[1]**2)).sqrt()
            nlf = het_gau_std
            synthesized_img = clean_img + torch.normal(mean=0., std=nlf)
        else:
            raise RuntimeError('undefined additive noise type : %s'%add_noise_type)

        if clamp:
            synthesized_img = torch.clamp(synthesized_img, 0, 255)

        return synthesized_img, nlf

    def _augmentation(self, data:dict, aug:list):
        '''
        Parsing augmentation list and apply it to the data images.
        '''
        # parsign augmentation
        rot, hflip = 0, 0
        for aug_name in aug:
            # aug : random rotation
            if aug_name == 'rot':
                rot = random.randint(0,3)
            # aug : random flip
            elif aug_name == 'hflip':
                hflip = random.randint(0,1)
            else:
                raise RuntimeError('undefined augmentation option : %s'%aug_name)
        
        # for every data(only image), apply rotation and flipping augmentation.
        for key in data:
            if self._is_image_tensor(data[key]):
                # random rotation and flip
                if rot != 0 or hflip != 0:
                    data[key] = rot_hflip_img(data[key], rot, hflip)
            
        return data

    #----------------------------#
    #   Image saving functions   #
    #----------------------------#
    def save_all_image(self, dir, clean=False, syn_noisy=False, real_noisy=False):
        for idx in range(len(self.img_paths)):
            data = self.__getitem__(idx)

            if clean and 'clean' in data:
                cv2.imwrite(os.path.join(dir, '%04d_CL.png'%idx), tensor2np(data['clean']))
            if syn_noisy and 'syn_noisy' in data:
                cv2.imwrite(os.path.join(dir, '%04d_SN.png'%idx), tensor2np(data['syn_noisy']))
            if real_noisy and 'real_noisy' in data:
                cv2.imwrite(os.path.join(dir, '%04d_RN.png'%idx), tensor2np(data['real_noisy']))

            print('image %04d saved!'%idx)

    def prep_save(self, img_size:int, overlap:int, clean:bool=False, syn_noisy:bool=False, real_noisy:bool=False):
        '''
        chopping images into mini-size patches for efficient training.
        Args:
            img_size (int) : size of image
            overlap (int) : overlap between patches
            clean (bool) : save clean image (default: False)
            syn_noisy (bool) : save synthesized noisy image (default: False)
            real_noisy (bool) : save real noisy image (default: False)
        '''
        d_name = '%s_s%d_o%d'%(self.__class__.__name__, img_size, overlap)
        os.makedirs(os.path.join(self.dataset_dir, 'prep', d_name), exist_ok=True)

        assert overlap < img_size
        stride = img_size - overlap

        if clean:
            clean_dir = os.path.join(self.dataset_dir, 'prep', d_name, 'CL')
            os.makedirs(clean_dir, exist_ok=True)
        if syn_noisy: 
            syn_noisy_dir = os.path.join(self.dataset_dir, 'prep', d_name, 'SN')
            os.makedirs(syn_noisy_dir, exist_ok=True)
        if real_noisy:
            real_noisy_dir = os.path.join(self.dataset_dir, 'prep', d_name, 'RN')
            os.makedirs(real_noisy_dir, exist_ok=True)

        for img_idx in range(self.__len__()):
            data = self.__getitem__(img_idx)

            c,h,w = data['clean'].shape if 'clean' in data else data['real_noisy'].shape
            for h_idx in range((h-img_size)//stride + 1):
                for w_idx in range((w-img_size+1)//stride + 1):
                    hl, hr = h_idx*stride, h_idx*stride+img_size
                    wl, wr = w_idx*stride, w_idx*stride+img_size
                    if clean:      cv2.imwrite(os.path.join(clean_dir,      '%d_%d_%d.png'%(img_idx, h_idx, w_idx)), tensor2np(data['clean'][:,hl:hr,wl:wr])) 
                    if syn_noisy:  cv2.imwrite(os.path.join(syn_noisy_dir,  '%d_%d_%d.png'%(img_idx, h_idx, w_idx)), tensor2np(data['syn_noisy'][:,hl:hr,wl:wr])) 
                    if real_noisy: cv2.imwrite(os.path.join(real_noisy_dir, '%d_%d_%d.png'%(img_idx, h_idx, w_idx)), tensor2np(data['real_noisy'][:,hl:hr,wl:wr])) 

            print('img%d'%img_idx)

    def prep_save_mat(self, img_size:int, overlap:int, clean=False, syn_noisy=False, real_noisy=False):
        d_name = '%s_s%d_o%d'%(self.__class__.__name__, img_size, overlap)
        os.makedirs(os.path.join(self.dataset_dir, 'prep', d_name), exist_ok=True)

        assert overlap < img_size
        stride = img_size - overlap

        if clean:
            clean_dir = os.path.join(self.dataset_dir, 'prep', d_name, 'CL')
            os.makedirs(clean_dir, exist_ok=True)
        if syn_noisy: 
            syn_noisy_dir = os.path.join(self.dataset_dir, 'prep', d_name, 'SN')
            os.makedirs(syn_noisy_dir, exist_ok=True)
        if real_noisy:
            real_noisy_dir = os.path.join(self.dataset_dir, 'prep', d_name, 'RN')
            os.makedirs(real_noisy_dir, exist_ok=True)

        for img_idx in range(self.__len__()):
            data = self.__getitem__(img_idx)

            c,h,w = data['clean'].shape if 'clean' in data else data['real_noisy'].shape
            for h_idx in range((h-img_size)//stride + 1):
                for w_idx in range((w-img_size+1)//stride + 1):
                    hl, hr = h_idx*stride, h_idx*stride+img_size
                    wl, wr = w_idx*stride, w_idx*stride+img_size
                    if clean:      savemat(os.path.join(clean_dir,      '%d_%d_%d.mat'%(img_idx, h_idx, w_idx)), {'x': tensor2np(data['clean'][:,hl:hr,wl:wr])})
                    if syn_noisy:  savemat(os.path.join(syn_noisy_dir,  '%d_%d_%d.mat'%(img_idx, h_idx, w_idx)), {'x': tensor2np(data['syn_noisy'][:,hl:hr,wl:wr])})
                    if real_noisy: savemat(os.path.join(real_noisy_dir, '%d_%d_%d.mat'%(img_idx, h_idx, w_idx)), {'x': tensor2np(data['real_noisy'][:,hl:hr,wl:wr])})

            print('img%d'%img_idx)
        return

    #----------------------------#
    #            etc             #
    #----------------------------#
    def _is_image_tensor(self, x):
        '''
        return input tensor has image shape. (include batched image)
        '''
        if isinstance(x, torch.Tensor):
            if len(x.shape) == 3 or len(x.shape) == 4:
                if x.dtype != torch.bool:
                    return True
        return False

class ReturnMergedDataset():
    def __init__(self, d_list):
        self.d_list = d_list
    def __call__(self, *args, **kwargs):
        return MergedDataset(self.d_list, *args, **kwargs)

class MergedDataset(Dataset):
    def __init__(self, d_list, *args, **kwargs):
        '''
        Merged denoising dataset when you use multiple dataset combined.
        see more details of DenoiseDataSet
        '''
        from ..datahandler import get_dataset_object

        self.dataset_list = []
        for d in d_list:
            self.dataset_list.append(get_dataset_object(d)(*args, **kwargs))

        self.data_contents_flags = {'clean':True, 'noisy':True, 'real_noisy':True}

        self.dataset_length = []
        for d in self.dataset_list:
            self.dataset_length.append(d.__len__())
            data_sample = d.__getitem__(0)
            for key in self.data_contents_flags.keys():
                if not key in data_sample:
                    self.data_contents_flags[key] = False

    def __len__(self):
        return sum(self.dataset_length)
        
    def __getitem__(self, idx):
        t_idx = idx
        for d_idx, d in enumerate(self.dataset_list):
            if t_idx < self.dataset_length[d_idx]:
                data = d.__getitem__(t_idx)
                return_data = {}
                for key in self.data_contents_flags.keys():
                    if self.data_contents_flags[key]:
                        return_data[key] = data[key]
                return return_data
            t_idx -= self.dataset_length[d_idx]
        raise RuntimeError('index of merged dataset contains some bugs, total length %d, requiring idx %d'%(self.__len__(), idx))
