
import os
from importlib import import_module

model_class_dict = {}

def regist_model(model_class):
    model_name = model_class.__name__.lower()
    assert not model_name in model_class_dict, 'there is already registered model: %s in model_class_dict.' % model_name
    model_class_dict[model_name] = model_class

    return model_class

def get_model_class(model_name:str):
    model_name = model_name.lower()
    return model_class_dict[model_name]

# import all python files in model folder
for module in os.listdir(os.path.dirname(__file__)):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    import_module('src.model.{}'.format(module[:-3]))
del module