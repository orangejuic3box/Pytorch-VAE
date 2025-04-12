import numpy as np
from torch.cuda import FloatTensor as CUDATensor
from visdom import Visdom

_WINDOW_CASH = {}

def _vis(env='main'):
    return Visdom(env=env)

import numpy as np
import torch
from PIL import Image

def visualize_image(tensor, name, label=None, env='main', w=250, h=250,
                    update_window_without_label=False):
    # Ensure tensor is on CPU for compatibility
    tensor = tensor.cpu() if isinstance(tensor, torch.Tensor) else tensor

    # Convert tensor to numpy array if it's a torch tensor
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().numpy()

    # Ensure the tensor is of the shape (height, width, channels) for RGB images
    if tensor.ndim != 3 or tensor.shape[-1] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3). Got shape {}".format(tensor.shape))

    # # Normalize if the values are not in [0, 1] or [0, 255]
    # if tensor.max() > 1:
    #     tensor = tensor / 255.0  # Normalize to [0, 1] if it is too large

    # Ensure the tensor is in the proper range [0, 255] and dtype uint8
    tensor = np.clip(tensor, 0, 255).astype(np.uint8)

    # Title and visualization using Visdom
    title = name + ('-{}'.format(label) if label is not None else '')

    # Create a PIL Image and show it
    try:
        im = Image.fromarray(tensor)
        print("WHAT IS THE FUCKING PROBLEM")
        im_array = np.array(im)
        print(im_array.size, im_array.shape)
        print("WHAT IS THE FUCKING PROBLEM not the im array")
        _WINDOW_CASH[title] = _vis(env).image(
            im_array, win=_WINDOW_CASH.get(title),
            opts=dict(title=name, width=w, height=h)
        )
        print("WHAT IS THE FUCKING PROBLEM DAWG")
    except Exception as e:
        print(f"Error converting tensor to image: {e}")
        exit(0)

    # # Create a PIL Image and show it
    # im = Image.fromarray(tensor)

    

    # _WINDOW_CASH[title] = _vis(env).image(
    #     np.array(im), win=_WINDOW_CASH.get(title),
    #     opts=dict(title=title, width=w, height=h)
    # )

    if update_window_without_label:
        _WINDOW_CASH[name] = _vis(env).image(
            np.array(im), win=_WINDOW_CASH.get(name),
            opts=dict(title=name, width=w, height=h)
        )


# def visualize_image(tensor, name, label=None, env='main', w=250, h=250,
#                     update_window_without_label=False):
#     # Check if the input is a PyTorch tensor; if so, convert to NumPy.
#     if not isinstance(tensor, np.ndarray):
#         # If the tensor is on GPU, move it to CPU first.
#         tensor = tensor.cpu() if hasattr(tensor, "cpu") else tensor
#         tensor = tensor.numpy()
        
#     title = name + ('-{}'.format(label) if label is not None else '')

#     _WINDOW_CASH[title] = _vis(env).image(
#         tensor, win=_WINDOW_CASH.get(title),
#         opts=dict(title=title, width=w, height=h)
#     )

#     # Update a separate window without label if requested
#     if update_window_without_label:
#         _WINDOW_CASH[name] = _vis(env).image(
#             tensor, win=_WINDOW_CASH.get(name),
#             opts=dict(title=name, width=w, height=h)
#         )


def visualize_images(tensor, name, label=None, env='main', w=850, h=450,
                     update_window_without_label=False):
    tensor = tensor.cpu() if isinstance(tensor, CUDATensor) else tensor
    title = name + ('-{}'.format(label) if label is not None else '')

    _WINDOW_CASH[title] = _vis(env).images(
        tensor.numpy(), win=_WINDOW_CASH.get(title),
        opts=dict(title=title, width=w, height=h)
    )

    # This is useful when you want to maintain the most recent images.
    if update_window_without_label:
        _WINDOW_CASH[name] = _vis(env).images(
            tensor.numpy(), win=_WINDOW_CASH.get(name),
            opts=dict(title=name, width=w, height=h)
        )


def visualize_kernel(kernel, name, label=None, env='main', w=250, h=250,
                     update_window_without_label=False, compress_tensor=False):
    # Do not visualize kernels that does not exists.
    if kernel is None:
        return

    assert len(kernel.size()) in (2, 4)
    title = name + ('-{}'.format(label) if label is not None else '')
    kernel = kernel.cpu() if isinstance(kernel, CUDATensor) else kernel
    kernel_norm = kernel if len(kernel.size()) == 2 else (
        (kernel**2).mean(-1).mean(-1) if compress_tensor else
        kernel.view(
            kernel.size()[0] * kernel.size()[2],
            kernel.size()[1] * kernel.size()[3],
        )
    )
    kernel_norm = kernel_norm.abs()

    visualized = (
        (kernel_norm - kernel_norm.min()) /
        (kernel_norm.max() - kernel_norm.min())
    ).numpy()

    _WINDOW_CASH[title] = _vis(env).image(
        visualized, win=_WINDOW_CASH.get(title),
        opts=dict(title=title, width=w, height=h)
    )

    # This is useful when you want to maintain the most recent images.
    if update_window_without_label:
        _WINDOW_CASH[name] = _vis(env).image(
            visualized, win=_WINDOW_CASH.get(name),
            opts=dict(title=name, width=w, height=h)
        )


def visualize_scalar(scalar, name, iteration, env='main'):
    visualize_scalars(
        [scalar] if isinstance(scalar, float) or len(scalar) == 1 else scalar,
        [name], name, iteration, env=env
    )


def visualize_scalars(scalars, names, title, iteration, env='main'):
    assert len(scalars) == len(names)
    # Convert scalar tensors to numpy arrays.
    scalars, names = list(scalars), list(names)
    scalars = [s.cpu() if isinstance(s, CUDATensor) else s for s in scalars]
    scalars = [s.numpy() if hasattr(s, 'numpy') else np.array([s]) for s in scalars]
    multi = len(scalars) > 1
    num = len(scalars)

    options = dict(
        fillarea=True,
        legend=names,
        width=800,
        height=400,
        xlabel='Iterations',
        ylabel="Loss",
        title=title,
        marginleft=70,
        marginright=30,
        marginbottom=80,
        margintop=50,
    )

    X = (
        np.column_stack(np.array([iteration] * num)) if multi else
        np.array([iteration] * num)
    )
    Y = np.column_stack(scalars) if multi else scalars[0]

    if title in _WINDOW_CASH:
        # Use the 'line' method to update the plot with new data.
        _vis(env).line(X=X, Y=Y, win=_WINDOW_CASH[title], update='append', opts=options)
    else:
        # Create a new plot if the window doesn't exist.
        _WINDOW_CASH[title] = _vis(env).line(X=X, Y=Y, opts=options)
