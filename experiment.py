'''
This file runs my experiment of analyzing our critical timestep.

Takes in the model with the correct parameters and the unseen data.
Loads up the correctly trained weights.

Goes through each datapair:
    - generates a new reconstructed version
    - saves noisy, clean, and reconstructed
'''
import utils
import pickle
import torch
from visual import visualize_images
from torchvision.utils import save_image, make_grid
import os
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np

def calculate_mse(tensor1, tensor2):
    """Calculate Mean Squared Error between two tensors of shape [C, H, W]
    Args:
    - tensor1 (torch.Tensor): The first tensor (e.g., reconstructed image)
    - tensor2 (torch.Tensor): The second tensor (e.g., clean image)"""
    mse_loss = F.mse_loss(tensor1, tensor2)
    return mse_loss.item()

def calculate_ssim(tensor1, tensor2, win_size=3, data_range = 255.0):
    """Calculate Structural Similarity Index (SSIM) between two tensors of shape [C, H, W]
    Args:
    - tensor1 (torch.Tensor): The first tensor (e.g., reconstructed image)
    - tensor2 (torch.Tensor): The second tensor (e.g., clean image)"""
    # Convert tensors to numpy arrays
    tensor1_np = tensor1.detach().cpu().numpy()
    tensor2_np = tensor2.detach().cpu().numpy()

    # If the tensors are grayscale, their shape will be [H, W]. For color images, it will be [C, H, W].
    if tensor1_np.shape[0] == 1:
        tensor1_np = tensor1_np[0]
        tensor2_np = tensor2_np[0]
    
    # Normalize the range if needed (ensure it's in the range [0, 255])
    tensor1_np = tensor1_np * 255
    tensor2_np = tensor2_np * 255

    # Calculate SSIM
    ssim_value, _ = ssim(tensor1_np, tensor2_np, win_size=win_size, full=True, data_range=data_range)
    
    return ssim_value

def tensor_to_image(tensor):
    #helps for save formatting
    return tensor.detach().cpu()


### run the experiment ###
def run_analysis(model, dataset, custom="MVAE-16k-z64-noisy_clean-3layers/epochs1000-1800-batch_size64-lr0.0005-wd0.0001/checkpoint.pth"):
    
    output_dir ="./outputs/"
    path_to_model = custom
    utils.load_checkpoint(model, "./checkpoints", custom=custom)
    #model should now be updated

    #put into evaluation mode
    model.eval()

    #go through each datapair and gather images and torch tensors
    set = 1
    set_imgs = []
    t = 350
    analysis = {}

    #noisy_x and clean_x are torch.Tensor types
    for i, (noisy_x, clean_x) in enumerate(dataset):
        if i%11 == 0 and  i!=0:
            #reset set images when all 11 have been processed
            set += 1
            set_imgs = []
            t = 350
        # print(f"--------{i} set {set}-------")
        #noise, clean, reconstructed image list
        ncr = []

        # print(type(noisy_x), type(clean_x[0]))
        # print(noisy_x.shape, noisy_x[0].shape)


        #add noisy and clean images
        n = tensor_to_image(noisy_x[0]) #[0] is C, H, W
        c = tensor_to_image(clean_x[0])

        ncr.append(n)
        ncr.append(c)

        #add reconstructed
        _, x_reconstructed = model(noisy_x)

        r = tensor_to_image(x_reconstructed[0])
        ncr.append(r)

        #after getting all 3, add to the set
        set_imgs += ncr

        #calculate mse and ssim
        mse = calculate_mse(x_reconstructed[0], clean_x[0])
        ssim = calculate_ssim(x_reconstructed[0], clean_x[0])
        print(f"Set{set} Timestep{t}: MSE {mse:.4f} | SSIM {ssim:.4f}")
        
        #put all the things in a dictionary
        if set not in analysis:
            analysis[set] = {} #intiialize first one
        analysis[set][t] = {
                            "tensors":[noisy_x[0], clean_x[0], x_reconstructed[0]],
                            "metrics":[mse, ssim]
                            }

        t -= 10
        #checking if its the LAST ONE 0-11, 12-21, etc.
        #DISPLAYS IMAGES ON VISDOM AND SAVES IMAGES
        if (i+1) %11 == 0 and i!=0:
            # Saves the imgaes
            img_tensors = torch.stack(set_imgs)

            # Display the image using Visdom
            visualize_images(img_tensors, f"Set {set} NCR", w=400, h=400)

            # Make a grid (you can adjust nrow as needed)
            grid = make_grid(img_tensors, nrow=3, normalize=True)

            # Save the grid to a single image file
            output_path = os.path.join(output_dir, f"Set{set}.png")
            save_image(grid, output_path)

            # # Optional: pause to see the image in the browser
            # usr = input("Press Enter to continue or q to exit...")  # This ensures you can view the image before exiting.
            # if usr =="q":
            #     exit()

    #Dumpt to pickle file
    # Open the file in write-binary mode and dump the dictionary
    pickle_path = os.path.join(output_dir, f"exp_results.pkl")
    with open(pickle_path, 'wb') as f:
        pickle.dump(analysis, f)

    print(f"Dictionary saved to {pickle_path}")