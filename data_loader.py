import torch
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

def data_loader_cifar10(path):
    NUM_TRAIN = 490000
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Spliting dataset object for each split(train / val / test)
    train_data = dset.CIFAR10(path, train=True, download=True, transform=transform)
    loader_train = DataLoader(train_data, batch_size=64,
                                sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
    val_data = dset.CIFAR10(path, train=True, download=True, transform=transform)
    loader_val = DataLoader(val_data, batch_size=64,
                                sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))
    
    test_data = dset.CIFAR10(path, train=False, download=True, transform=transform)
    loader_test = DataLoader(test_data, batch_size=64)

    return loader_train, loader_val, loader_test