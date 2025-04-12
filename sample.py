'''
Takes in a checkpoint path to a saved trained model

Generates an image and displays it on the vis server
'''
import argparse
import utils
from mags_model import MVAE
import visual

def parse_args():
    parser = argparse.ArgumentParser(description="Generate image from trained VAE model")
    parser.add_argument('--custom', type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for generating images")
    return parser.parse_args()

def sample_images(custom, n):
    #create model
    model = MVAE(
            label="noisy_clean",
            image_size=32,
            channel_num=3,
            kernel_num=16,
            z_size=64,
            layers=3
    )
    #load model
    model_dir = "./checkpoints"
    utils.load_checkpoint(model, model_dir, custom)
    #model should have updated weights now
    images = model.sample(n)
    visual.visualize_images(
                    images, f'generated samples {model.name}',
                    env="main"
                )
    

if __name__ == '__main__':
    args = parse_args()
    sample_images(args.custom, args.batch_size)
