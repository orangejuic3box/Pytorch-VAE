import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np

def evaluate_reconstruction_mse(model, data_loader, device='cuda'):
    model.eval()  # Set model to evaluation mode
    total_mse = 0
    num_samples = 0
    
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for noisy_x, clean_x in data_loader:
            noisy_x, clean_x = noisy_x.to(device), clean_x.to(device)
            
            # Forward pass
            (_, _), x_reconstructed = model(noisy_x)
            
            # Compute Mean Squared Error
            mse_loss = F.mse_loss(x_reconstructed, clean_x, reduction='sum')
            
            total_mse += mse_loss.item()
            num_samples += clean_x.shape[0]
    
    avg_mse = total_mse / num_samples
    print(f"Average Reconstruction MSE: {avg_mse:.4f}")
    return avg_mse


def evaluate_ssim(model, data_loader, device='cuda'):
    model.eval()
    total_ssim = 0
    num_samples = 0

    with torch.no_grad():
        for noisy_x, clean_x in data_loader:
            noisy_x, clean_x = noisy_x.to(device), clean_x.to(device)

            (_, _), x_reconstructed = model(noisy_x)

            for i in range(clean_x.shape[0]):
                clean_img = clean_x[i].cpu().numpy().transpose(1, 2, 0)
                recon_img = x_reconstructed[i].cpu().numpy().transpose(1, 2, 0)

                # Compute SSIM
                ssim_value = ssim(clean_img, recon_img, multichannel=True)
                total_ssim += ssim_value
                num_samples += 1

    avg_ssim = total_ssim / num_samples
    print(f"Average SSIM: {avg_ssim:.4f}")
    return avg_ssim
