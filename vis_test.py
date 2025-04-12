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

    print("in tensor_to_image")
    print(tensor.shape)

    return tensor

# Connect to Visdom
vis = visdom.Visdom()

# Check if Visdom server is running
if not vis.check_connection():
    print("Visdom server is not running. Please start it.")
else:
    train_loader, _, _ = get_nc_datasets()

    print("INSIDE VIS TEST")
    j=0
    imgs = []
    
    # Get the first batch from the DataLoader
    for i, (image, clean) in enumerate(train_loader):
        print("---------------")
        # Check the type of your image
        print("image is type", type(image))  # It should output something like <class 'torch.Tensor'>

        print("Image shape:", image.shape)
        print("Image dtype:", image.dtype)
        # Assuming your image is in a tensor format
        print("Min value:", image.min().item())  # Get the minimum pixel value
        print("Max value:", image.max().item())  # Get the maximum pixel value
        if i%11 == 0:  # Access the first batch
            j+=1
            image = image[0]
            clean = clean[0]

            # Convert the first image in the batch (assuming batch size > 1)
            img = tensor_to_image(image)  # Get the first image from the batch
            cln = tensor_to_image(clean)
            imgs.append(cln)

    # Optional: pause to see the image in the browser
    input("Press Enter to continue...")  # This ensures you can view the image before exiting.
    img_tensors = torch.stack(imgs)

    # Display the image using Visdom
    visualize_images(img_tensors, f"{j} Output Images", w=800, h=800)

    # vis.image(img, opts=dict(title="Input Image"))
    # vis.image(cln, opts=dict(title=f"Output Image {j}"))

            