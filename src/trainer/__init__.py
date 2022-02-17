import os
from importlib import import_module


trainer_class_dict = {}

def regist_trainer(trainer):
    trainer_name = trainer.__name__.lower()
    assert not trainer_name in trainer_class_dict, 'there is already registered dataset: %s in trainer_dict.' % trainer_name
    trainer_class_dict[trainer_name] = trainer

    return trainer

def get_trainer_class(trainer_name:str):
    trainer_name = trainer_name.lower()
    return trainer_class_dict[trainer_name]

# import all python files in trainer folder
for module in os.listdir(os.path.dirname(__file__)):
    if module == '__init__.py' or module[-3:] != '.py':
        continue
    import_module('src.trainer.{}'.format(module[:-3]))
del module
