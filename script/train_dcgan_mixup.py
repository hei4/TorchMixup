# -*- coding:utf-8 -*-
import argparse
import os
import random
import torch.nn as nn

import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from net import weights_init, Generator, Discriminator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batch', type=int, default=50, help='input batch size')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ng_ch', type=int, default=64)
    parser.add_argument('--nd_ch', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--outf', default='./result', help='folder to output images and model checkpoints')

    opt = parser.parse_args()
    print(opt)

    batch_size = opt.batch
    epoch_size = opt.epoch

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    random.seed(0)
    torch.manual_seed(0)

    dataset = dset.SVHN(root='../svhn_root', download=True,
                        transform=transforms.Compose([
                            transforms.Resize(64),
                            transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=int(opt.workers))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    nz = int(opt.nz)

    netG = Generator().to(device)
    netG.apply(weights_init)
    print(netG)

    netD = Discriminator().to(device)
    netD.apply(weights_init)
    print(netD)

    criterion = nn.BCEWithLogitsLoss()

    fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=1e-5)

    beta_dist = torch.distributions.beta.Beta(0.2, 0.2)

    for epoch in range(epoch_size):
        for itr, data in enumerate(dataloader):
            real_image = data[0].to(device)
            sample_size = real_image.size(0)
            noise = torch.randn(sample_size, nz, 1, 1, device=device)            
        
            mix_label = beta_dist.sample((sample_size, )).to(device)
            mix_rate = mix_label.reshape(sample_size, 1, 1, 1)
        
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            
            fake_image = netG(noise)
            mix_image = mix_rate * real_image + (1. - mix_rate) * fake_image            

            output = netD(mix_image)
            errD = criterion(output, mix_label)
            errD.backward()
            optimizerD.step()

            D_mix1 = output.mean().item()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()

            fake_image = netG(noise)
            mix_image = mix_rate * real_image + (1. - mix_rate) * fake_image

            output = netD(mix_image)
            errG = criterion(output, 1. - mix_label)
            errG.backward()
            optimizerG.step()

            D_mix2 = output.mean().item()

            print('[{}/{}][{}/{}] Loss_D: {:.3f} Loss_G: {:.3f} D(mix): {:.3f}/{:.3f}'
                  .format(epoch + 1, epoch_size,
                          itr + 1, len(dataloader),
                          errD.item(), errG.item(), D_mix1, D_mix2))

            if epoch == 0 and itr == 0:
                vutils.save_image(real_image, '{}/real_samples.png'.format(opt.outf),
                                  normalize=True, nrow=10)

        fake_image = netG(fixed_noise)
        vutils.save_image(fake_image.detach(), '{}/fake_samples_epoch_{:03d}.png'.format(opt.outf, epoch + 1),
                          normalize=True, nrow=10)

        # do checkpointing
        if (epoch + 1) % 100 == 0:
            torch.save(netG.state_dict(), '{}/netG_epoch_{}.pth'.format(opt.outf, epoch + 1))
            torch.save(netD.state_dict(), '{}/netD_epoch_{}.pth'.format(opt.outf, epoch + 1))


if __name__ == '__main__':
    main()
