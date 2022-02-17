
import os
from importlib import import_module
from .denoise_dataset import ReturnMergedDataset

dataset_class_dict = {}

def regist_dataset(dataset_class):
    dataset_name = dataset_class.__name__.lower()
    assert not dataset_name in dataset_class_dict, 'there is already registered dataset: %s in dataset_class_dict.' % dataset_name
    dataset_class_dict[dataset_name] = dataset_class

    return dataset_class

def get_dataset_class(dataset_name):
    dataset_name = dataset_name.lower()
    
    # Case of using multiple dataset
    if len(dataset_name.split('+')) > 1:
        merge_data_list = []
        for d in dataset_name.replace(' ', '').split('+'):
            merge_data_list.append(d)
        return ReturnMergedDataset(merge_data_list)

    # Single dataset
    else:
        return dataset_class_dict[dataset_name]

# import all python files in model folder
for module in os.listdir(os.path.dirname(__file__)):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    import_module('src.datahandler.{}'.format(module[:-3]))
del module
