#!/usr/bin/env python3

import torch
from torchvision.utils import save_image

import numpy as np


def save_recon_images(args, model, validation_loader, data_shape):
    """Saves reconstructed images to samples folder."""

    with torch.no_grad():

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        validation_set_enumerate = enumerate(validation_loader)

        _, data = next(validation_set_enumerate)

        if args.data == "piv":
            x, y = data['ComImages'], data['AllGenDetails']
            x = x.to(device)
        else:
            x, y = data
            x = x.to(device)

        if args.model == "ffjord":
            generated_sample = model(x, reverse=True)
        elif args.model == "snf":
            generated_sample, z_mu, z_var, ldj, z0, z_k = model(x)

        generated_sample = generated_sample.cpu()
        images = x.cpu()

        if args.data == "piv":
            x_cat_1 = torch.cat([images[:args.save_recon_images_size,0,:,:], generated_sample[:args.save_recon_images_size,0,:,:]],
                    0).view(-1, 1, data_shape[1], data_shape[2])
            x_cat_2 = torch.cat([images[:args.save_recon_images_size,1,:,:], generated_sample[:args.save_recon_images_size,1,:,:]],
                    0).view(-1, 1, data_shape[1], data_shape[2])

            images = [x_cat_1.cpu().data,x_cat_2.cpu().data]

            count = 0
            for image in images:
                name = args.save + '/reconstruction_' + args.experiment_name + str(count) + '.png'
                save_image(image, name, nrow=8 )
                count += 1

        else:
            x_cat = torch.cat([images[:args.save_recon_images_size,0,:,:], generated_sample[:args.save_recon_images_size,0,:,:]],
                    0).view(-1, 1, data_shape[1], data_shape[2])
            name = args.save + '/reconstruction_' + args.experiment_name + '.png'
            save_image(x_cat, name, nrow=8)


def save_fixed_z_image(args, model, data_shape):
    """ Save samples with fixed z. """

    with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)
        fixed_z = cvt(torch.randn(100, *data_shape))

        if args.model == "ffjord":
            generated_sample = model(fixed_z, reverse=True)
        elif args.model == "snf":
            generated_sample, z_mu, z_var, ldj, z0, z_k = model(fixed_z)

        if args.data == "piv":
            x_cat_1 = torch.cat([generated_sample[:args.save_recon_images_size,0,:,:]],0).view(-1, 1, data_shape[1], data_shape[2])
            x_cat_2 = torch.cat([generated_sample[:args.save_recon_images_size,1,:,:]],0).view(-1, 1, data_shape[1], data_shape[2])

            images = [x_cat_1.cpu().data,x_cat_2.cpu().data]

            count = 0
            for image in images:
                name = args.save + '/fixed_z_' + args.experiment_name + str(count) + '.png'
                save_image(image, name, nrow=8 )
                count += 1

        else:
            x_cat = torch.cat([generated_sample[:args.save_recon_images_size,0,:,:]],
                    0).view(-1, 1, data_shape[1], data_shape[2])
            name = args.save + '/fixed_z_' + args.experiment_name + '.png'
            save_image(x_cat, name, nrow=8)


