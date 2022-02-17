import os

from src.datahandler.denoise_dataset import DenoiseDataSet
from . import regist_dataset


@regist_dataset
class DND(DenoiseDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):
        dataset_path = os.path.join(self.dataset_dir, 'DND/dnd_2017/png_srgb')
        assert os.path.exists(dataset_path), 'There is no dataset %s'%dataset_path
        for root, _, files in os.walk(dataset_path):
            for file_name in files:
                # TODO parse folder name of each shot
                # parsed_name = self._parse_folder_name(folder_name)
                self.img_paths.append(os.path.join(root, file_name))

    def _load_data(self, data_idx):
        noisy_img = self._load_img(self.img_paths[data_idx])
        return {'real_noisy': noisy_img}

    def _parse_folder_name(self, name):
        raise NotImplementedError
        parsed = {}
        return parsed

@regist_dataset
class prep_DND(DenoiseDataSet):
    '''
    dataset class for prepared DND dataset which is cropped with overlap.
    [using size 512x512 with 128 overlapping]
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):
        self.dataset_path = os.path.join(self.dataset_dir, 'prep/DND_s512_o128')
        assert os.path.exists(self.dataset_path), 'There is no dataset %s'%self.dataset_path
        for root, _, files in os.walk(os.path.join(self.dataset_path, 'RN')):
            self.img_paths = files

    def _load_data(self, data_idx):
        file_name = self.img_paths[data_idx]

        noisy_img = self._load_img(os.path.join(self.dataset_path, 'RN' , file_name))

        return {'real_noisy': noisy_img} #'instances': instance }

@regist_dataset
class DND_benchmark(DenoiseDataSet):
    '''
    dumpy dataset class for DND benchmark
    DND benchmarking code is implemented in the "trainer" directly
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):
        pass

    def _load_data(self, data_idx):
        pass