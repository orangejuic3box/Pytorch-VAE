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