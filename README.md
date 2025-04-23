# Model adapted for WINTER 2025 RESEARCH: XAI for Generative Diffusion Models.
Using a custom dataset (from a diffusion model) prepared for my experiment, VAE was trained to reconstruct the generated image from a noisy image during the diffusion process. Essentially mapping noise to a final image, however this task is quite hard and VAE underfits. 

Original Project did not work properly for CIFAR-10, unsure how they got their reconstructed samples. 


## To run model:
#### (1) only do this if you modify requirements.txt or changed envr like a reinstall
```
pip install -r requirements.txt
```

#### (2) create virtual environment, conda environment (only do this again if you deleted the envr)
```
conda env create
```

#### (3) activate vm envr
```
conda activate pytorch-vae
```

#### (4) setup in development mode (only do this if envr was deleted or new envr)
```
pip install -e
```

#### (5) start visdom server for visualization, at ```localhost:8097```
```
python -m visdom.server
nohup visdom > visdom.log 2>&1 &
```

#### (6) to kill visdom server

first run this to find the pid
```
ps aux | grep visdom
```
then replace with your pid
```
kill pid
```

#### (7) deacrivate envr
```
conda deactivate
```

## Train:
Check main.py for default parameters. Only thing that can't be changed is layers. Default parameters were for the ```noisy_clean``` dataset. Unsure of performance on cifar10. 
```
python main.py --train
 [--dataset {cifar10, noisy_clean}]
 [--kernel-num KERNEL_NUM] [--z-size Z_SIZE]
 [--epochs EPOCHS] [--batch-size BATCH_SIZE]
 [--sample-size SAMPLE_SIZE] [--lr LR]
 [--weight-decay WEIGHT_DECAY]
 [--inter INTERVAL_PRINT]
 [--loss-log-interval LOSS_LOG_INTERVAL]
 [--image-log-interval IMAGE_LOG_INTERVAL]
 [--resume] [--checkpoint-dir CHECKPOINT_DIR]
 [--sample-dir SAMPLE_DIR] [--no-gpus]
```

## Experiment:
Generates new images based off what wasn't used in training. Data is ordered for experiment purposes. Calculates MSE/SSIM metrics. Needs custom path to model checkpoint.
```
python main.py --exp
  [--custom CUSTOM]
```
```
python main.py --exp --custom MVAE-16k-z64-noisy_clean-3layers/best-loss0.04207158461213112epoch1092/checkpoint.pth
```

## Generate Images From The Model:
```
python gen_images.py
  [--checkpoint CHECKPOINT_PATH]
  [--batch_size BATCH_SIZE]
```
```
python gen_images.py --checkpoint ./checkpoints/MVAE-16k-z64-noisy_clean-3layers/best-loss0.04207158461213112epoch1092/checkpoint.pth
```

### BEST MODELS SO FAR BELOW

./checkpoints/MVAE-16k-z64-noisy_clean-3layers/best-loss0.04207158461213112epoch1092/checkpoint.pth

/Users/magui/Desktop/Pytorch-VAE/checkpoints/MVAE-16k-z64-noisy_clean-3layers/best-loss0.04207158461213112epoch1092/checkpoint.pth

checkpoint = ./checkpoints/MVAE-16k-z64-noisy_clean-3layers/epochs1000-1800-batch_size64-lr0.0005-wd0.0001/checkpoint.pth

