from util.registry import Registry

TRANSFORMS = Registry('Transforms')
DATA = Registry('Data')

import glob
import importlib
import torch
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from .dataset_info import *
from .sampler import *
from timm.data.distributed_sampler import RepeatAugSampler
from torch.utils.data import DataLoader, Subset

files = glob.glob('data/[!_]*.py')
for file in files:
    model_lib = importlib.import_module(file.split('.')[0].replace('/', '.'))

from data.utils import get_transforms


def get_dataset(cfg):
    train_transforms = get_transforms(cfg, train=True, cfg_transforms=cfg.train_data.train_transforms)
    test_transforms = get_transforms(cfg, train=False, cfg_transforms=cfg.test_data.test_transforms)

    train_target_transforms = get_transforms(cfg, train=False, cfg_transforms=cfg.train_data.target_transforms)
    test_target_transforms = get_transforms(cfg, train=False, cfg_transforms=cfg.train_data.target_transforms)

    train_set = DATA.get_module(cfg.train_data.type)(cfg.train_data, train=True, transform=train_transforms,
                                                     target_transform=train_target_transforms)
    test_set = DATA.get_module(cfg.test_data.type)(cfg.test_data, train=False, transform=test_transforms,
                                                   target_transform=test_target_transforms)
    return train_set, test_set


def get_loader(cfg):
    train_set, test_set = get_dataset(cfg)

    test_sampler = None

    if cfg.train_data.sampler.name == 'naive':
        train_sampler = None
    elif cfg.train_data.sampler.name == 'balanced':
        train_sampler = SAMPLER.get(cfg.train_data.sampler.name, None)

        if train_sampler:
            train_sampler = train_sampler(batch_size=cfg.trainer.data.batch_size_per_gpu, dataset=train_set)
    else:
        raise NotImplementedError

    if train_sampler:
        train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                   batch_sampler=train_sampler,
                                                   num_workers=cfg.trainer.data.num_workers_per_gpu,
                                                   pin_memory=cfg.trainer.data.pin_memory,
                                                   persistent_workers=cfg.trainer.data.persistent_workers)
    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                   batch_size=cfg.trainer.data.batch_size_per_gpu,
                                                   shuffle=True,
                                                   sampler=train_sampler,
                                                   num_workers=cfg.trainer.data.num_workers_per_gpu,
                                                   pin_memory=cfg.trainer.data.pin_memory,
                                                   drop_last=cfg.trainer.data.drop_last,
                                                   persistent_workers=cfg.trainer.data.persistent_workers)

    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=cfg.trainer.data.batch_size_per_gpu_test,
                                              shuffle=False,
                                              sampler=test_sampler,
                                              num_workers=cfg.trainer.data.num_workers_per_gpu,
                                              pin_memory=cfg.trainer.data.pin_memory,
                                              drop_last=False,
                                              persistent_workers=cfg.trainer.data.persistent_workers)
    return train_loader, test_loader, train_set, test_set


def get_split_loader(cfg):
    train_set, test_set = get_dataset(cfg)
    test_sampler = None

    if cfg.train_data.sampler.name == 'naive':
        train_sampler = None
    elif cfg.train_data.sampler.name == 'balanced':
        train_sampler = SAMPLER.get(cfg.train_data.sampler.name, None)

        if train_sampler:
            train_sampler = train_sampler(batch_size=cfg.trainer.data.batch_size_per_gpu, dataset=train_set)
    else:
        raise NotImplementedError


    if train_sampler:
        train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                   batch_sampler=train_sampler,
                                                   num_workers=cfg.trainer.data.num_workers_per_gpu,
                                                   pin_memory=cfg.trainer.data.pin_memory,
                                                   persistent_workers=cfg.trainer.data.persistent_workers)
    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                   batch_size=cfg.trainer.data.batch_size_per_gpu,
                                                   shuffle=True,
                                                   sampler=train_sampler,
                                                   num_workers=cfg.trainer.data.num_workers_per_gpu,
                                                   pin_memory=cfg.trainer.data.pin_memory,
                                                   drop_last=cfg.trainer.data.drop_last,
                                                   persistent_workers=cfg.trainer.data.persistent_workers)

    dataset_size = len(train_set)
    subset_size = dataset_size // cfg.subset_num
    # randomly shuffle index of the whole dataset
    all_indices = list(range(dataset_size))
    random.shuffle(all_indices)
    train_set = Subset(train_set, all_indices)
    # split index
    split_indices = []
    for j in range(cfg.subset_num):
        start = j * subset_size
        end = start + subset_size if j < cfg.subset_num - 1 else dataset_size
        split_indices.append(all_indices[start:end])    # select index (randomly shuffled) of subset

    # split dataset by index and creat DataLoader by index
    sub_train_loaders = []
    test_train_loaders = []
    for indices in split_indices:
        subset = Subset(train_set, indices)  # creat subset by index
        if train_sampler:
            sub_train_loader = torch.utils.data.DataLoader(dataset=subset,
                                                       batch_sampler=train_sampler,
                                                       num_workers=cfg.trainer.data.num_workers_per_gpu,
                                                       pin_memory=cfg.trainer.data.pin_memory,
                                                       persistent_workers=cfg.trainer.data.persistent_workers)
        else:
            sub_train_loader = torch.utils.data.DataLoader(dataset=subset,
                                                       batch_size=cfg.trainer.data.batch_size_per_gpu,
                                                       shuffle=True,
                                                       sampler=train_sampler,
                                                       num_workers=cfg.trainer.data.num_workers_per_gpu,
                                                       pin_memory=cfg.trainer.data.pin_memory,
                                                       drop_last=cfg.trainer.data.drop_last,
                                                       persistent_workers=cfg.trainer.data.persistent_workers)
            test_train_loader = torch.utils.data.DataLoader(dataset=subset,
                                                            batch_size=cfg.trainer.data.batch_size_per_gpu,
                                                            shuffle=False,
                                                            sampler=train_sampler,
                                                            num_workers=cfg.trainer.data.num_workers_per_gpu,
                                                            pin_memory=cfg.trainer.data.pin_memory,
                                                            drop_last=cfg.trainer.data.drop_last,
                                                            persistent_workers=cfg.trainer.data.persistent_workers)
        sub_train_loaders.append(sub_train_loader)
        test_train_loaders.append(test_train_loader)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=cfg.trainer.data.batch_size_per_gpu_test,
                                              shuffle=False,
                                              sampler=test_sampler,
                                              num_workers=cfg.trainer.data.num_workers_per_gpu,
                                              pin_memory=cfg.trainer.data.pin_memory,
                                              drop_last=False,
                                              persistent_workers=cfg.trainer.data.persistent_workers)
    return train_loader, sub_train_loaders, test_train_loaders, test_loader, train_set, test_set
