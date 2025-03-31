'''
This file does preproccessing of the training data.
Set configurations for the cifar-10 dataset (from og git).

Function for loading picklefile of dataset.
Custom Dataset Class for my images, used for efficient data processing in Pytorch.

Configuration added for custom vae dataset
'''
from torchvision import datasets, transforms
#for custom vae dataset
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np



class NoisyCleanDataset(Dataset):
    '''Class for custom vae dataset of noisy and clean image pairs'''
    def __init__(self, noisy_images, clean_images):
        """
        Parameters:
            noisy_images (numpy.ndarray): Noisy input images, shape (N, 32, 32, 3)
            clean_images (numpy.ndarray): Corresponding clean images, shape (N, 32, 32, 3)
            transform (callable, optional): Optional transform to apply to images.
        """
        assert noisy_images.shape == clean_images.shape, "Noisy and clean images must have the same shape"
        self.noisy_images = noisy_images
        self.clean_images = clean_images

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        noisy = self.noisy_images[idx]  # Shape (32, 32, 3)
        clean = self.clean_images[idx]  # Shape (32, 32, 3)

        # Convert from NumPy (H, W, C) to PyTorch Tensor (C, H, W) for VAE
        noisy = torch.tensor(noisy, dtype=torch.float32).permute(2, 0, 1) / 255.0 #normalizes the image
        clean = torch.tensor(clean, dtype=torch.float32).permute(2, 0, 1) / 255.0 #normalizes the image

        return noisy, clean



_CIFAR_TRAIN_TRANSFORMS = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
]

_CIFAR_TEST_TRANSFORMS = [
    transforms.ToTensor(),
]


TRAIN_DATASETS = {
    'cifar10': datasets.CIFAR10(
        './datasets/cifar10', train=True, download=True,
        transform=transforms.Compose(_CIFAR_TRAIN_TRANSFORMS)
    )
}


TEST_DATASETS = {
    'cifar10': datasets.CIFAR10(
        './datasets/cifar10', train=False,
        transform=transforms.Compose(_CIFAR_TEST_TRANSFORMS)
    )
}


DATASET_CONFIGS = {
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
}
