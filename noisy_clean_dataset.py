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

        # print(type(noisy),noisy.shape)

        # # Check the raw min and max before normalization
        # print(f"Raw min value of noisy image: {noisy.min()}")
        # print(f"Raw max value of noisy image: {noisy.max()}")

        # # Convert from NumPy (H, W, C) to PyTorch Tensor (C, H, W) for VAE
        # noisy = torch.tensor(noisy, dtype=torch.float32).permute(2, 0, 1) / 255.0 #normalizes the image
        # clean = torch.tensor(clean, dtype=torch.float32).permute(2, 0, 1) / 255.0 #normalizes the image
        
        # Normalize the values to the range [0, 1]
        noisy = (noisy - noisy.min()) / (noisy.max() - noisy.min())  # Normalize noisy image to [0, 1]
        clean = (clean - clean.min()) / (clean.max() - clean.min())  # Normalize clean image to [0, 1]

        # Convert from NumPy (H, W, C) to PyTorch Tensor (C, H, W)
        noisy = torch.tensor(noisy, dtype=torch.float32).permute(2, 0, 1)
        clean = torch.tensor(clean, dtype=torch.float32).permute(2, 0, 1)
        
        
        
        # print("AFTER NORNAIZATION IN GET ITEM NOISY CLEAN")
        # print(f"Min value of noisy image: {noisy.min().item()}")
        # print(f"Max value of noisy image: {noisy.max().item()}")





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
    #total 1111 pairs, 101 original images with 11 noisy images
    # 81-10-10
    data_size = len(noisyclean_dataset)
    n=81
    train_size = 11*n
    test_size = 110 + (891-train_size)
    val_size = 110
    print(data_size, train_size, test_size, val_size)
    assert_string = f"DATASET SIZE SPLIT DOES NOT MATCH {train_size}+{test_size}+{val_size}={train_size+test_size+val_size} should be {data_size}"
    assert (train_size+test_size+val_size == data_size, assert_string)

    #split noisyclean into train, test, val
    #plit into subsets SAME SUBSET EVERY TIME U BITCH
    train_dataset = torch.utils.data.Subset(noisyclean_dataset, range(0, train_size))  # First 'train_size' items
    test_dataset = torch.utils.data.Subset(noisyclean_dataset, range(train_size, train_size + test_size))  # Next 'test_size' items
    val_dataset = torch.utils.data.Subset(noisyclean_dataset, range(train_size + test_size, train_size + test_size + val_size))  # Next 'val_size' items
    experiment_dataset = torch.utils.data.Subset(noisyclean_dataset, range(train_size, train_size + test_size + val_size))
   
   
    #make dataloaders for each dataset
    batch_size = 64
    #train is shuffled so model doesn't memorize order, test and val are NOT shuffled so testing is consistent
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    experiment_loader = DataLoader(experiment_dataset, batch_sampler=batch_size) #potentionally change this?

    return train_loader, test_loader, val_loader, experiment_loader
