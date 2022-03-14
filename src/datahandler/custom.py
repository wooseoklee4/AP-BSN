import os

import h5py

from src.datahandler.denoise_dataset import DenoiseDataSet
from . import regist_dataset


@regist_dataset
class CustomSample(DenoiseDataSet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _scan(self):
        # check if the dataset exists
        dataset_path = os.path.join('WRITE_YOUR_DATASET_DIRECTORY')
        assert os.path.exists(dataset_path), 'There is no dataset %s'%dataset_path

        # WRITE YOUR CODE FOR SCANNING DATA
        # example:
        for root, _, files in os.walk(dataset_path):
            for file_name in files:
                self.img_paths.append(os.path.join(root, file_name))

    def _load_data(self, data_idx):
        # WRITE YOUR CODE FOR LOADING DATA FROM DATA INDEX
        # example:
        file_name = self.img_paths[data_idx]

        noisy_img = self._load_img(os.path.join(self.dataset_path, 'RN' , file_name))
        clean_img = self._load_img(os.path.join(self.dataset_path, 'CL' , file_name))

        return {'clean': clean_img, 'real_noisy': noisy_img} # paired dataset
        # return {'real_noisy': noisy_img} # only noisy image dataset