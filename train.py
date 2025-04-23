'''Modified verion of train-old.py because of no cuda environment.

Getting a tensor Index err bc of 0-dim tensor, need to convert to a number

'''
import torch
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import utils
import visual
import math


#not even the good one
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
                checkpoint_dir='./checkpoints/',
                custom="",
                inter = 100,
                resume=False,
                cuda=False):
    
    #change checkpoint
    # checkpoint_dir+"/"+model.name+"/"
    optimizer = optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=weight_decay,
    )
    
    if resume:
        epoch_start, optimizer_state = utils.load_checkpoint(model, "./checkpoints", custom=custom)
        optimizer.load_state_dict(optimizer_state)
        print("uploaded the optimzer")
        #same as before potentially change to loading this
        # optimizer = optim.Adam(
        # model.parameters(), lr=lr, 
        # weight_decay=weight_decay
        # )

    else:
        epoch_start = 1
        #xaiver weights?
        model.apply_xavier_initialization()

    #always put into training mode
    model.train()
    
    batches_per_epoch = len(dataset)

    print(model.training, "should be value True")
    print(F'BATCH SIZE {batch_size} and batches per epoch {batches_per_epoch}')

    # print(batches_per_epoch, len(dataset), batch_size)
    
    print(f"STARTING AT EPOCH {epoch_start} AND GOING TILL EPOCH {epochs}")

    best_loss = float('inf')

    for epoch in range(epoch_start, epochs+1):

        data_loader = dataset

        dataload_stream = enumerate(data_loader, 1)
        print("in da epoch")

        data_stream = tqdm(dataload_stream)

        reconstruction_loss = None
        kl_divergence_loss = None
        total_loss = None

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=True)
        epoch_loss = 0

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
            (mean, logvar), x_reconstructed = model(noisy_x) #model.forward is being called here
           
            # print(f"was reconstructed to {x_reconstructed.shape}")

            # get loss 
            reconstruction_loss = model.reconstruction_loss(x_reconstructed, clean_x)
            kl_divergence_loss = model.kl_divergence_loss(mean, logvar)
            total_loss = reconstruction_loss + 0.01 * kl_divergence_loss

            epoch_loss += total_loss

            # print("what happened to loss?", total_loss, "iter", iteration)

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

            parameter_str = f'w/ lr {lr}, z{model.z_size}, k{model.kernel_num}, epochs{epoch_start}/{epochs}'

            if iteration % inter == 0: #0 doesnt work bc iteragions too small?
                losses = [
                    # reconstruction_loss.item(),  
                    # kl_divergence_loss.item(),  
                    total_loss.item() 
                ]
                names = ['reconstruction', 'kl divergence', 'total']
                name = ['total loss']
                visual.visualize_scalars(
                    losses, name, f'total loss {parameter_str}',
                    iteration, env=model.name)

            if iteration % inter == 0:
                images = model.sample(sample_size)
                visual.visualize_images(
                    images, f'generated samples {parameter_str}',
                    env=model.name
                )
            
            if total_loss < best_loss and epoch > 300:
                best_loss = total_loss

                model_dict = {"epochs":epochs,
                      "batch_size":batch_size,
                      "weight_decay":weight_decay,
                      "lr":lr,
                      "epoch":epoch,
                      "optimizer": optimizer.state_dict()}

                # save the checkpoint.
                utils.save_checkpoint(model, checkpoint_dir+"/"+model.name+"/", model_dict, custom=f"best-loss{best_loss}epoch{epoch}")
                print()

        
        #calculate epoch loss for lr optimizer
        epoch_loss /= len(data_loader)
        # 🔥 adjust LR if plateauing  
        scheduler.step(epoch_loss)  

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
                      "lr":lr,
                      "epoch":epoch,
                      "optimizer": optimizer.state_dict()}

        # save the checkpoint.
        utils.save_checkpoint(model, checkpoint_dir+"/"+model.name+"/", model_dict, custom=f"epochs{epoch_start}-{epochs}-batch_size{batch_size}-lr{lr}-wd{weight_decay}")
        print()
