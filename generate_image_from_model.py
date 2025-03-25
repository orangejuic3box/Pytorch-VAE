'''
Takes in a checkpoint path to a saved trained model

Generates an image and displays it on the vis server
'''

import torch
import argparse
import visdom
from model import VAE  # Import the VAE model

# Step 1: Set up command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Generate image from trained VAE model")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for generating images")
    return parser.parse_args()

# Step 2: Load model with correct parameters
def load_model_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

    # Retrieve saved parameters from the checkpoint
    label = checkpoint['label']
    image_size = checkpoint['image_size']
    channel_num = checkpoint['channel_num']
    kernel_num = checkpoint['kernel_num']
    z_size = checkpoint['z_size']

    # Initialize VAE with extracted parameters
    model = VAE(label, image_size, channel_num, kernel_num, z_size)

    # Load model weights
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()  # Set to evaluation mode

    return model, z_size

# Step 3: Generate and visualize images
def generate_image_from_model(checkpoint_path, batch_size=64):
    model, z_size = load_model_from_checkpoint(checkpoint_path)

    # Generate random latent vectors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = torch.randn(batch_size, z_size, device=device)

    # Generate images
    with torch.no_grad():  # No gradients needed during inference
        generated_images = model.sample(batch_size)  # Use `sample()` instead of `decode()`

    # Step 4: Visualize using Visdom
    vis = visdom.Visdom()

    # Get the first generated image
    image_to_display = generated_images[0].cpu().numpy()  # Convert to NumPy

    # Normalize to [0,1] for visualization
    image_to_display = (image_to_display - image_to_display.min()) / (image_to_display.max() - image_to_display.min())

    # Display image in Visdom
    vis.image(image_to_display, win='generated_image', opts=dict(title='Generated Image'))

if __name__ == '__main__':
    args = parse_args()
    generate_image_from_model(args.checkpoint, args.batch_size)
