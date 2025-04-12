import os
import os.path
import torch
from torch.utils.data import DataLoader


def get_data_loader(dataset, batch_size, cuda=False):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        **({'num_workers': 1, 'pin_memory': True} if cuda else {})
    )


def save_checkpoint(model, model_dir, model_dict,custom=""):
    #general path directory
    if custom == "":
        epochs = model_dict["epochs"]
        batch_size = model_dict["batch_size"]
        weight_decay = model_dict["weight_decay"]
        lr = model_dict["lr"]

        path = os.path.join(
            model_dir, f"VAE-{model.kernel_num}k-{model.label}-{model.channel_num}x{model.image_size}x{model.image_size}-z{model.z_size}-batch{batch_size}-wd{weight_decay}-epochs{epochs}-lr{lr}"
        )
    #path directory for resp experiment
    else:
        path = os.path.join(model_dir, custom)


    # save the checkpoint.
    if not os.path.exists(path):
        os.makedirs(path)
        print("made the path", path)

    print("directory", model_dir)
    print("path", path)


    # Save model parameters and state
    checkpoint = {
        'state_dict': model.state_dict(),  # Saves the actual model weights
        'label': model.label,
        'image_size': model.image_size,
        'channel_num': model.channel_num,
        'kernel_num': model.kernel_num,
        'z_size': model.z_size,
        'epoch': model_dict["epoch"],
        'optimizer': model_dict["optimizer"],
        'model':model
    }
    # Ensure to save with a file name
    checkpoint_path = os.path.join(path, "checkpoint.pth")

    torch.save(checkpoint, checkpoint_path)
    # notify that we successfully saved the checkpoint.
    print(f'=> Saved the model {model.name} to {path}')


def load_checkpoint(model, model_dir, custom=""):
    #general path
    if custom == "":
        path = os.path.join(model_dir, model.name)
    #experiment path
    else:
        path = os.path.join(model_dir, custom)
        print(f"the path traveled: {model_dir} + {custom}")


    print("pls load the path", path)
    # load the checkpoint.
    checkpoint = torch.load(path)
    print('=> loaded checkpoint of {name} from {path}'.format(
        name=model.name, path=(path)
    ))

    print("attempting a thing")
    print(model.encoder)
    
    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state_dict'])
  


    epoch = checkpoint['epoch']
    optimizer = checkpoint["optimizer"]
    return epoch, optimizer


def xavier_initialize(model):
    modules = [
        m for n, m in model.named_modules() if
        'conv' in n or 'linear' in n
    ]

    parameters = [
        p for
        m in modules for
        p in m.parameters() if
        p.dim() >= 2
    ]

    for p in parameters:
        init.xavier_normal(p)
