"""
This file loads and processes the custom dataset of noisy-clean image pairs.

Function for loading picklefile of dataset.
Custom Dataset Class for my images, used for efficient data processing in Pytorch.
"""
#for custom vae dataset
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split # for reconstruction datasets (only paired images)
import numpy as np
import os


def load_pickle_dataset(pickle_path):
    '''
    Parameters:
        pickle_path (String): path to dataset
    Return:
        noisy_images (NumPy Array): array of noisy images (1111, 32, 32, 3)
        clean_images (NumPy Array): array of clean images (1111, 32, 32, 3)
    '''
    # Load dataset from pickle file
    with open(pickle_path, 'rb') as f:
        dataset_dict = pickle.load(f)  # Load the dictionary
    
    noisy_images = dataset_dict["noise"]  # Extract noisy images
    clean_images = dataset_dict["clean"]  # Extract clean images

    return noisy_images, clean_images

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


def get_nc_datasets():
    '''
    Loads and processes noisy clean dataset for VAE. Returns the 3 split of data
    Returns:
        train_loader (DataLoader): training set of the noisy clean data
        test_loader (DataLoader): test set of the noisy clean data
        val_loader (DataLoader): validation set of the noisy clean data
    '''
    #grab custom dataset
    cwd = os.getcwd()
    pickle_path = os.path.join(cwd, "datasets", "vae_dataset.pkl")
    noisy_images, clean_images = load_pickle_dataset(pickle_path)

    #turn into custom class
    noisyclean_dataset = NoisyCleanDataset(noisy_images, clean_images)

    #split into train, test, validate
    #calculate size of each dataset 80-10-10 split
    data_size = len(noisyclean_dataset)
    train_size = int(0.8 * data_size)
    test_size = int(0.1 * data_size)
    val_size = data_size - train_size - test_size

    #split noisyclean into train, test, val
    train_dataset, test_dataset, val_dataset = random_split(noisyclean_dataset, [train_size, test_size, val_size])

    #make dataloaders for each dataset
    batch_size = 64
    #train is shuffled so model doesn't memorize order, test and val are NOT shuffled so testing is consistent
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader
