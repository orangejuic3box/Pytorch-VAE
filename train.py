'''Modified verion of train-old.py because of no cuda environment.

Getting a tensor Index err bc of 0-dim tensor, need to convert to a number

'''

from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
import utils
import visual
import math


def train_model(model, dataset, epochs=10,
                batch_size=32, sample_size=32,
                lr=3e-04, weight_decay=1e-5,
                loss_log_interval=30,
                image_log_interval=300,
                checkpoint_dir='./checkpoints',
                resume=False,
                cuda=False):
    # prepare optimizer and model
    model.train()
    optimizer = optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=weight_decay,
    )

    if resume:
        epoch_start = utils.load_checkpoint(model, checkpoint_dir)
    else:
        epoch_start = 1

    for epoch in range(epoch_start, epochs+1):
        data_loader = utils.get_data_loader(dataset, batch_size, cuda=cuda)
        data_stream = tqdm(enumerate(data_loader, 1))

        for batch_index, (x, _) in data_stream:
            # where are we?
            iteration = (epoch-1)*(len(dataset)//batch_size) + batch_index

            # prepare data on gpu if needed
            x = Variable(x).cuda() if cuda else Variable(x)

            # flush gradients and run the model forward
            optimizer.zero_grad()
            (mean, logvar), x_reconstructed = model(x)
            reconstruction_loss = model.reconstruction_loss(x_reconstructed, x)
            kl_divergence_loss = model.kl_divergence_loss(mean, logvar)
            total_loss = reconstruction_loss + kl_divergence_loss

            # backprop gradients from the loss
            total_loss.backward()
            optimizer.step()

            # update progress
            data_stream.set_description((
                'epoch: {epoch} | '
                'iteration: {iteration} | '
                'progress: [{trained}/{total}] ({progress:.0f}%) | '
                'loss => '
                'total: {total_loss:.4f} / '
                're: {reconstruction_loss:.3f} / '
                'kl: {kl_divergence_loss:.3f}'
            ).format(
                epoch=epoch,
                iteration=iteration,
                trained=batch_index * len(x),
                total=len(data_loader.dataset),
                progress=(100. * batch_index / len(data_loader)),
                total_loss=total_loss.item(),  # Change here
                reconstruction_loss=reconstruction_loss.item(),  # Change here
                kl_divergence_loss=kl_divergence_loss.item(),  # Change here
            ))

            if iteration % loss_log_interval == 0:
                losses = [
                    reconstruction_loss.item(),  # Change here
                    kl_divergence_loss.item(),  # Change here
                    total_loss.item()  # Change here
                ]
                names = ['reconstruction', 'kl divergence', 'total']
                visual.visualize_scalars(
                    losses, names, 'loss',
                    iteration, env=model.name)

            if iteration % image_log_interval == 0:
                images = model.sample(sample_size)
                visual.visualize_images(
                    images, 'generated samples',
                    env=model.name
                )

        # notify that we've reached to a new checkpoint.
        print()
        print()
        print('#############')
        print('# checkpoint!')
        print('#############')
        print()

        model_dict = {"epochs":epochs,
                      "batch_size":batch_size,
                      "weight_decay":weight_decay,
                      "lr":lr}

        # save the checkpoint.
        utils.save_checkpoint(model, checkpoint_dir, model_dict)
        print()

#### train model for reconstruction not classification
def train_model_nc(model, dataset, epochs=10,
                batch_size=32, sample_size=32,
                lr=3e-04, weight_decay=1e-5,
                checkpoint_dir='./checkpoints',
                resume=False,
                cuda=False):
    # prepare optimizer and model
    model.train()
    optimizer = optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=weight_decay,
    )

    if resume:
        epoch_start = utils.load_checkpoint(model, checkpoint_dir)
    else:
        epoch_start = 1
    
    batches_per_epoch = len(dataset)

    print(model.training, "should be True")

    # print(batches_per_epoch, len(dataset), batch_size)
    
    for epoch in range(epoch_start, epochs+1):
        data_loader = dataset
        data_stream = tqdm(enumerate(data_loader, 1))

        reconstruction_loss = None
        kl_divergence_loss = None
        total_loss = None

        for batch_index, (noisy_x, clean_x) in data_stream:
            #THERE ARE 14 BATCHES 888/64 = 14 PER EPOCH

            # where are we?
            # iteration is global count

            iteration = (epoch-1)*(batches_per_epoch) + batch_index
            # print(f"\nepoch: {epoch} | batch_index:{batch_index} | iteration:{iteration} | bpe: {batches_per_epoch}")

            # prepare data on gpu if needed
            noisy_x = Variable(noisy_x).cuda() if cuda else Variable(noisy_x)
            clean_x = Variable(clean_x).cuda() if cuda else Variable(clean_x)
            
            # flush gradients and run the model forward
            optimizer.zero_grad()
            # forward pass
            (mean, logvar), x_reconstructed = model(noisy_x)
           
            # get loss 
            reconstruction_loss = model.reconstruction_loss(x_reconstructed, clean_x)
            # kl_divergence_loss = model.kl_divergence_loss(mean, logvar)
            total_loss = reconstruction_loss #+ kl_divergence_loss

            # backprop gradients from the loss
            total_loss.backward()
            optimizer.step()

            # update progress
            data_stream.set_description((
                'epoch: {epoch} | '
                'batch i: {batch_index} | '
                'iter: {iteration} | '
                'progress: [{trained}/{total}] ({progress:.0f}%) | '
                'loss => '
                'total: {total_loss:.4f} = '
                're: {reconstruction_loss:.3f} / '
                # 'kl: {kl_divergence_loss:.3f} '
            ).format(
                epoch=epoch,
                batch_index = batch_index,
                iteration=iteration,
                trained=batch_index * len(noisy_x),
                total=len(data_loader.dataset),
                progress=(100. * batch_index / len(data_loader)),
                total_loss=total_loss.item(),  # Change here
                reconstruction_loss=reconstruction_loss.item(),  # Change here
                # kl_divergence_loss=kl_divergence_loss.item(),  # Change here
            ))

            
            if iteration % batches_per_epoch == 0: #0 doesnt work bc iteragions too small?
                losses = [
                    reconstruction_loss.item(),  # Change here
                    # kl_divergence_loss.item(),  # Change here
                    # total_loss.item()  # Change here
                ]
                names = ['reconstruction', 'kl divergence', 'total']
                name = ['reconstruction']
                visual.visualize_scalars(
                    losses, name, 'loss',
                    iteration, env=model.name)

            if iteration % (batches_per_epoch) == 0:
                images = model.sample(sample_size)
                visual.visualize_images(
                    images, f'generated samples',
                    env=model.name
                )
            
        
        # losses = [
        #     reconstruction_loss.item(),  # Change here
        #     kl_divergence_loss.item(),  # Change here
        #     total_loss.item()  # Change here
        # ]
        # names = ['reconstruction', 'kl divergence', 'total']
        
        # visual.visualize_scalars(
        #     losses, names, 'loss',
        #     iteration, env=model.name)

        # # if iteration % image_log_interval == 2:
        # images = model.sample(sample_size)
        # visual.visualize_images(
        #     images, 'generated samples',
        #     env=model.name
        # )

        # notify that we've reached to a new checkpoint.
        print()
        print()
        print('#############')
        print('# checkpoint!')
        print('#############')
        print()

        model_dict = {"epochs":epochs,
                      "batch_size":batch_size,
                      "weight_decay":weight_decay,
                      "lr":lr}

        # save the checkpoint.
        utils.save_checkpoint(model, checkpoint_dir, model_dict, custom=f"VAE-RESPONSIBILITY-EXPERIMENT-noKL-epochs{epochs}")
        print()
