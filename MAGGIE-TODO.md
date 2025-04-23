(1) install conda
(2) get the vae running on vscode
(3) train model on cifar10 here
(4) generate a few images for the dataset
(5) test that the model is being trained



To run model:
(1) only do this if you modify requirements.txt or changed envr like a reinstall
pip install -r requirements.txt

(2) create virtual environment, conda environment (only do this again if you deleted the envr)
conda env create

(3) activate vm envr
conda activate pytorch-vae

(4) setup in development mode (only do this if envr was deleted or new envr)
pip install -e 


(5) start visdom server for visualization
python -m visdom.server
nohup visdom > visdom.log 2>&1 &


(6) kill visdom server
ps aux | grep visdom

kill pid

(7)deacrivate envr
conda deactivate


python main.py --train --dataset noisy_clean --epochs 25 --batch-size 64



potentially change z in new model

(8) run experiment
python main.py --exp

python main.py --exp --custom MVAE-16k-z64-noisy_clean-3layers/best-loss0.04207158461213112epoch1092/checkpoint.pth

(9) generate images from the model
python gen_images --checkpoint --batch_size(optional)

checkpoint = ./checkpoints/MVAE-16k-z64-noisy_clean-3layers/epochs1000-1800-batch_size64-lr0.0005-wd0.0001/checkpoint.pth


BEST SO FAR BELOW

./checkpoints/MVAE-16k-z64-noisy_clean-3layers/best-loss0.04207158461213112epoch1092/checkpoint.pth

/Users/magui/Desktop/Pytorch-VAE/checkpoints/MVAE-16k-z64-noisy_clean-3layers/best-loss0.04207158461213112epoch1092/checkpoint.pth


NEXT
(1) SAVE TENSOR OF RECONTRUCTED AND CLEAN
(2) CALCULATE MSE/SSIM