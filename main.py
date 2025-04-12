#!/usr/bin/env python3
import argparse
import torch
import torchvision
from model import VAE
from mags_model import MVAE
from data import TRAIN_DATASETS, DATASET_CONFIGS
from train import train_model, train_model_nc


parser = argparse.ArgumentParser('VAE PyTorch implementation')
parser.add_argument('--dataset', default='mnist',
                    choices=list(TRAIN_DATASETS.keys()))

parser.add_argument('--kernel-num', type=int, default=16) #128
parser.add_argument('--z-size', type=int, default=64 ) #64 #128
parser.add_argument('--layers', type=int, default=3 ) #64 #128

parser.add_argument('--epochs', type=int, default=500) #10
#                       BATCH SIZE SHOULD BE NUMBER OF PAIRS IN DATASET?
parser.add_argument('--batch-size', type=int, default=64) #32
parser.add_argument('--sample-size', type=int, default=32)

parser.add_argument('--lr', type=float, default=0.0005) #5e-03 $5e-04 
#5e-03 is doing something interesting, had 1e-03 before
parser.add_argument('--weight-decay', type=float, default=1e-4) #1e-03

parser.add_argument('--inter', type=int, default=100)
parser.add_argument('--loss-log-interval', type=int, default=100)
parser.add_argument('--image-log-interval', type=int, default=500)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
parser.add_argument('--custom', type=str, default='')
parser.add_argument('--sample-dir', type=str, default='./samples')
parser.add_argument('--no-gpus', action='store_false', dest='cuda')

main_command = parser.add_mutually_exclusive_group(required=True)
main_command.add_argument('--test', action='store_false', dest='train')
main_command.add_argument('--train', action='store_true')


if __name__ == '__main__':
    args = parser.parse_args()
    cuda = args.cuda and torch.cuda.is_available()
    dataset_config = DATASET_CONFIGS[args.dataset]
    dataset = TRAIN_DATASETS[args.dataset]
    print(len(dataset), "DATASET SIZE")

    #original vae model
    # vae = VAE(
    #     label=args.dataset,
    #     image_size=dataset_config['size'],
    #     channel_num=dataset_config['channels'],
    #     kernel_num=args.kernel_num,
    #     z_size=args.z_size,
    # )

    #MAGS VAE MODEL
    vae = MVAE(
            label=args.dataset,
            image_size=dataset_config['size'],
            channel_num=dataset_config['channels'],
            kernel_num=args.kernel_num,
            z_size=args.z_size,
            layers=args.layers
        )
    # move the model parameters to the gpu if needed.
    if cuda:
        vae.cuda()

    # run a test or a training process.
    if args.dataset == "noisy_clean":
        print("training for noisy clean")
        train_model_nc(
            vae, dataset=dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sample_size=args.sample_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            checkpoint_dir=args.checkpoint_dir,
            custom=args.custom,
            inter=args.inter,
            resume=args.resume,
            cuda=cuda,)
    elif args.train: #this is the og git version
        print("training for NOT noisy clean")
        train_model(
            vae, dataset=dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sample_size=args.sample_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            checkpoint_dir=args.checkpoint_dir,
            loss_log_interval=args.loss_log_interval,
            image_log_interval=args.image_log_interval,
            resume=args.resume,
            cuda=cuda,
        )
    else: #this is not actually testing, this is just sampling from the model
        images = vae.sample(args.sample_size)
        torchvision.utils.save_image(images, args.sample_dir)
