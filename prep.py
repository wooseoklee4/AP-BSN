import argparse, os
import multiprocessing as mp
from importlib import import_module



from src.datahandler import get_dataset_class


def main():
    # parsing configuration
    args = argparse.ArgumentParser()
    args.add_argument('-d', '--dataset',    default='',   type=str)
    args.add_argument('-s', '--patch_size', default=512,  type=int)
    args.add_argument('-o', '--overlap',    default=128,  type=int)
    args.add_argument('-p', '--process',    default=8,    type=int)

    args = args.parse_args()

    assert args.dataset != '', 'dataset name is needed'

    dataset = get_dataset_class(args.dataset)()
    
    # check what the dataset have images
    data_sample = dataset.__getitem__(0)
    flag_c, flag_n = 'clean' in data_sample, 'real_noisy' in data_sample

    pool = mp.Pool(args.process)
    mp_args = [[data_idx, args.patch_size, args.overlap, flag_c, False, flag_n] for data_idx in range(dataset.__len__())]

    pool.starmap(dataset.prep_save, mp_args)


if __name__ == '__main__':
    main()
