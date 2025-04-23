import visdom
import numpy as np
from noisy_clean_dataset import get_nc_datasets
from torchvision import transforms
from visual import visualize_images
import torch

# Assuming your image is in a tensor (torch.Size([C, H, W])) and needs to be visualized
def tensor_to_image(tensor):
    # Assuming tensor is a [C, H, W] and you need to convert it to numpy array
    tensor = tensor.cpu().detach()  # Detach from graph and move to CPU
    # tensor = tensor.numpy().transpose(1, 2, 0)  # Convert to HWC format for display
    return tensor

# Connect to Visdom
vis = visdom.Visdom()

# Check if Visdom server is running
if not vis.check_connection():
    print("Visdom server is not running. Please start it.")
else:
    train_loader, _, _, experiment_loader = get_nc_datasets()

    print("INSIDE VIS TEST")
    j=0
    imgs = []
    set = 0

    loader = experiment_loader #train_loader

    print("loader is", len(loader), "long")
    
    # Get the first batch from the DataLoader
    for i, (image, clean) in enumerate(loader):
        if i%11 == 0 and  i!=0:
            imgs = []
        # print(f"--------{i} set {set}-------")

        # print("Image shape:", image.shape)
       
        img = tensor_to_image(image[0])
        imgs.append(img)
        if (i+1) %11 == 0 and i!=0:  # Access the first batch
            set = i//11
            cln = tensor_to_image(clean[0])
            imgs.append(cln)

            #############
            print(type(image), type(clean))
            #############

            # Optional: pause to see the image in the browser
            usr = input("Press Enter to continue...")  # This ensures you can view the image before exiting.
            if usr =="q":
                exit()
    
            img_tensors = torch.stack(imgs)

            # Display the image using Visdom
            visualize_images(img_tensors, f"Set{set} Noisy-Clean", w=400, h=400)

    # vis.image(img, opts=dict(title="Input Image"))
    # vis.image(cln, opts=dict(title=f"Output Image {j}"))

            