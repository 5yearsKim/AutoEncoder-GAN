from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from models import AutoEncoder, Discriminator
from utils import weights_init, to_one_hot_vector



def main(args):
    # Set random seed for reproducibility
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)


    dataroot = args.dataroot
    workers = args.workers
    batch_size = args.batch_size
    nc = args.nc
    ngf = args.ngf
    ndf = args.ndf
    nhd = args.nhd
    num_epochs =args.num_epochs
    lr = args.lr
    beta1 = args.beta1
    ngpu = args.ngpu
    resume = args.resume
    record_pnt = args.record_pnt
    log_pnt = args.log_pnt
    mse = args.mse

    '''
    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)
    '''
    dataset = dset.MNIST(root=dataroot,
                         transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   # transforms.Normalize(0.5, 0.5),
                               ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)


    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")




    # Create the generator
    netG = AutoEncoder(nc, ngf, nhd=nhd).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)




    # Create the Discriminator
    netD = Discriminator(nc, ndf, ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    #resume training if args.resume is True
    if resume:
        ckpt = torch.load('ckpts/recent.pth')
        netG.load_state_dict(ckpt["netG"])
        netD.load_state_dict(ckpt["netD"])


    # Initialize BCELoss function
    criterion = nn.BCELoss()
    MSE = nn.MSELoss()
    mse_coeff = 1.
    center_coeff = 0.001


    # Establish convention for real and fake flags during training
    real_flag = 1
    fake_flag = 0

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.dec.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerAE = optim.Adam(netG.parameters(), lr=lr, betas=(beta1,0.999))




    # Training Loop

    # Lists to keep track of progress
    iters = 0

    R_errG=0
    R_errD=0
    R_errAE=0
    R_std = 0
    R_mean = 0


    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_img, label = data
            real_img, label = real_img.to(device), to_one_hot_vector(10, label).to(device)

            b_size = real_img.size(0)
            flag = torch.full((b_size,), real_flag, device=device)
            # Forward pass real batch through D
            output = netD(real_img, label).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, flag)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate fake image batch with G
            noise = torch.randn(b_size, nhd, 1, 1).to(device)
            fake = netG.dec(noise, label)
            flag.fill_(fake_flag)
            # Classify all fake batch with D
            output = netD(fake.detach(), label).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, flag)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.dec.zero_grad()
            flag.fill_(real_flag)  # fake flags are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake, label).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, flag)
            # Calculate gradients for G
            errG.backward()
            # Update G
            optimizerG.step()


            ############################
            # (3) Update AE network: minimize reconstruction loss
            ###########################
            netG.zero_grad()
            new_img = netG(real_img, label, label)
            hidden = netG.enc(real_img, label)
            central_loss = MSE(hidden, torch.zeros(hidden.shape).to(device))
            errAE = mse_coeff* MSE(real_img, new_img) \
                    + center_coeff* central_loss
            errAE.backward()
            optimizerAE.step()

            R_errG += errG.item()
            R_errD += errD.item()
            R_errAE += errAE.item()
            R_std += (hidden**2).mean().item()
            R_mean += hidden.mean().item()
            # Output training stats
            if i % log_pnt == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss_AE: %.4f\t'
                      % (epoch, num_epochs, i, len(dataloader),
                         R_errD/log_pnt, R_errG/log_pnt, R_errAE/log_pnt))
                print('mean: %.4f\tstd: %.4f\tcentral/msecoeff: %4f'
                      % (R_mean/log_pnt, R_std/log_pnt, center_coeff/mse_coeff))
                R_errG = 0.
                R_errD = 0.
                R_errAE = 0.
                R_std = 0.
                R_mean = 0.


            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % record_pnt == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                vutils.save_image(fake.to("cpu"), './samples/image_{}.png'.format(iters//record_pnt))
                torch.save({
                    "netG": netG.state_dict(),
                    "netD": netD.state_dict(),
                    "nc": nc,
                    "ngf":ngf,
                    "ndf":ndf
                }, 'ckpts/recent.pth')

            iters += 1


if __name__ == "__main__":


    def str2bool(s):
        return s.lower().startswith('t')

    parser = argparse.ArgumentParser(description='ENCODING GAN')

    parser.add_argument('--dataroot', default="./data/MNIST", type=str, help='root directory for dataset')
    parser.add_argument('--workers', default=2, type=int, help='Number of workers for dataloader')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size for training')
    parser.add_argument('--image_size', default=28, type=int, help='image size')
    parser.add_argument('--nc', default=1, type=int,
                        help='Number of channels in the training images. For color images this is 3')
    parser.add_argument('--ngf', default=32, type=int, help='size of feature maps in generator')
    parser.add_argument('--ndf', default=32, type=int, help='size of feature maps in discriminator')
    parser.add_argument('--nhd', default=16, type=int, help='size of hidden dimension')
    parser.add_argument('--num_epochs', default=20, type=int, help='number of training epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate for optimizers')
    parser.add_argument('--beta1', default=0.5, type=float, help='Beta1 hyperparam for Adam optimizers')
    parser.add_argument('--ngpu', default=1, type=int, help='Number of GPUs available. use 0 for CPU mode')
    parser.add_argument('--record_pnt', default=500, type=int, help='Saving frequency')
    parser.add_argument('--log_pnt', default=100, type=int, help='print status frequency')
    parser.add_argument('--resume', default=False, type=str2bool, help='resume training')
    parser.add_argument('--mse', default=True, type=str2bool, help='add mse loss to generator loss')


    main(parser.parse_args())



