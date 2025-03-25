'''
Takes in a checkpoint path to a saved trained model

Generates an image and displays it on the vis server
'''

import torch
import argparse
from torch import nn
import visdom
from model import VAE  # Replace with the actual import of your model

# Step 1: Set up command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Generate image from trained VAE model")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument('--latent_dim', type=int, default=100, help="Latent space dimension")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for generating images")
    return parser.parse_args()

def generate_image_from_model(checkpoint_path, latent_dim=100, batch_size=64):
    # Step 2: Load the Trained Model
    model = VAE()  # Replace with the actual model initialization
    checkpoint = torch.load(checkpoint_path)  # Path to your checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode

    # Step 3: Generate a Random Sample (from latent space)
    z = torch.randn(batch_size, latent_dim)  # Batch of random latent vectors

    # Generate the image(s) using the decoder (assuming your model has a decode method)
    with torch.no_grad():  # Disable gradient calculation during inference
        generated_images = model.decode(z)  # Modify if your model uses different function names

    # Step 4: Visualize the Image using Visdom
    # First, initialize Visdom
    vis = visdom.Visdom()

    # If you want to visualize the first image in the batch
    image_to_display = generated_images[0]  # Get the first image from the batch

    # Normalize the image (Visdom requires the image to be in the range [0, 1])
    image_to_display = image_to_display.cpu().numpy()  # Move the tensor to CPU and convert to NumPy
    image_to_display = (image_to_display - image_to_display.min()) / (image_to_display.max() - image_to_display.min())

    # Visualize using Visdom's image display function
    vis.image(image_to_display, win='generated_image', opts=dict(title='Generated Image'))


if __name__ == '__main__':
    # Parse the command-line arguments
    args = parse_args()

    # Generate an image from the model and visualize it
    generate_image_from_model(args.checkpoint, args.latent_dim, args.batch_size)
