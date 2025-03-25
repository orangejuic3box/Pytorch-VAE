import os
import os.path
import torch
from torch.utils.data import DataLoader


def get_data_loader(dataset, batch_size, cuda=False):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        **({'num_workers': 1, 'pin_memory': True} if cuda else {})
    )


def save_checkpoint(model, model_dir, epoch):
    path = os.path.join(model_dir, model.name)

    # save the checkpoint.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # this is the old save
    # torch.save({'state': model.state_dict(), 'epoch': epoch}, path)

    # Save model parameters and state
    checkpoint = {
        'state_dict': model.state_dict(),  # Saves the actual model weights
        'epoch': epoch,
        'label': model.label,
        'image_size': model.image_size,
        'channel_num': model.channel_num,
        'kernel_num': model.kernel_num,
        'z_size': model.z_size
    }

    torch.save(checkpoint, path)
    # notify that we successfully saved the checkpoint.
    print(f'=> Saved the model {model.name} to {path}')


def load_checkpoint(model, model_dir):
    path = os.path.join(model_dir, model.name)

    # load the checkpoint.
    checkpoint = torch.load(path)
    print('=> loaded checkpoint of {name} from {path}'.format(
        name=model.name, path=(path)
    ))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])
    epoch = checkpoint['epoch']
    return epoch


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
