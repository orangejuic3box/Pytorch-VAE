'''
This file does preproccessing of the training data.
Set configurations for the cifar-10 dataset (from og git).

Configuration added for custom vae dataset
'''
from torchvision import datasets, transforms # for classification datasets
from noisy_clean_dataset import get_nc_datasets

#load and process noisy_clean datasets
nc_train, nc_test, nc_val, nc_exp = get_nc_datasets()

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
    ),
    'noisy_clean': nc_train
}


TEST_DATASETS = {
    'cifar10': datasets.CIFAR10(
        './datasets/cifar10', train=False,
        transform=transforms.Compose(_CIFAR_TEST_TRANSFORMS)
    ),
    "noisy_clean": nc_test 

}

VAL_DATASETS = {
    "noisy_clean": nc_val
}

EXPERIMENT_DATASETS = {
    "noisy_clean": nc_exp #may need to change batch_sizer
    #this is nc_val and nc_test combines but
    #its fine bc i never used them
}


DATASET_CONFIGS = {
    #size = 32x32 pixels, 3 channels
    'cifar10': {'size': 32, 'channels': 3, 'classes': 10},
    'noisy_clean': {'size': 32, 'channels': 3}  # No 'classes' needed
}
