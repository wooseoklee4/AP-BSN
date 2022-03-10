import os

from src.datahandler.denoise_dataset import DenoiseDataSet
from . import regist_dataset


@regist_dataset
class NIND(DenoiseDataSet):
    '''
    NIND datatset class using original images.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):
        dataset_path = os.path.join(self.dataset_dir, 'NIND')
        assert os.path.exists(dataset_path), 'There is no dataset %s'%dataset_path

        # scan all image path & info in dataset path
        for folder_name in os.listdir(dataset_path):
            for (dirpath, _, filenames) in os.walk(os.path.join(dataset_path, folder_name)):
                filenames = sorted(filenames, key=self.ISO_sortkey)
                for filename in filenames:
                    # lowest ISO image is used as clean image
                    if filename == filenames[0]: continue

                    # parse filename into se
                    parsed_name = self._parse_filename(filename)
                    
                    info = {}
                    info['instances']       = parsed_name
                    info['noisy_img_path']  = os.path.join(dirpath, filename)
                    info['clean_img_path']  = os.path.join(dirpath, filenames[0]) # clean image is the lowest ISO image.
                    self.img_paths.append(info)

    def _load_data(self, data_idx):
        info = self.img_paths[data_idx]
        
        clean_img = self._load_img(info['clean_img_path'])
        noisy_img = self._load_img(info['noisy_img_path'])


        return {'clean': clean_img, 'real_noisy': noisy_img} #, 'instances': info['instances']}

    def _parse_filename(self, name):
        parsed = {}
        splited = name.split('.')[0].split('_') # NIND_Scene_ISO
        
        parsed['scene'] = splited[1]
        parsed['ISO']   = splited[2]
        return parsed

    def ISO_sortkey(self, name):
        code = name.split('ISO')[1].split('.')[0]
        if 'H' in code:
            if code == 'H1':
                return 10000
            elif code == 'H2':
                return 20000
            elif code == 'H3':
                return 30000
            elif code == 'H4':
                return 40000
            else:
                raise RuntimeError('%s'%code)
        elif '-' in code:
            return int(code.split('-')[0]) + int(code.split('-')[1])      
        else:
            return int(code)

@regist_dataset
class prep_NIND(DenoiseDataSet):
    '''
    dataset class for prepared NIND dataset which is cropped with overlap.
    [using size 512x512 with 128 overlapping]
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):
        self.dataset_path = os.path.join(self.dataset_dir, 'prep/NIND_s512_o128')
        assert os.path.exists(self.dataset_path), 'There is no dataset %s'%self.dataset_path
        for root, _, files in os.walk(os.path.join(self.dataset_path, 'RN')):
            self.img_paths = files

    def _load_data(self, data_idx):
        file_name = self.img_paths[data_idx]

        noisy_img = self._load_img(os.path.join(self.dataset_path, 'RN' , file_name))
        clean = self._load_img(os.path.join(self.dataset_path, 'CL' , file_name))

        return {'clean': clean, 'real_noisy': noisy_img} #'instances': instance }
