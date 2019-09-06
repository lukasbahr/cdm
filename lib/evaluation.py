#!/usr/bin/env python3

import torch
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


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

    number_img = 100

    with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)

        if args.model == "ffjord":
            fixed_z = cvt(torch.randn(number_img, *data_shape))
            generated_sample = model(fixed_z, reverse=True)

        elif args.model == "snf":
            fixed_z = cvt(torch.randn(number_img, args.z_size))
            generated_sample = model.decode(fixed_z)

        if args.data == "piv":
            x_cat_1 = torch.cat([generated_sample[:number_img, 0,:,:]],0).view(-1, 1, data_shape[1], data_shape[2])
            x_cat_2 = torch.cat([generated_sample[:number_img, 1,:,:]],0).view(-1, 1, data_shape[1], data_shape[2])

            images = [x_cat_1.cpu().data,x_cat_2.cpu().data]

            count = 0
            for image in images:
                name = args.save + '/fixed_z_' + args.experiment_name + str(count) + '.png'
                save_image(image, name, nrow=10 )
                count += 1

        else:
            x_cat = torch.cat([generated_sample[:number_img,0,:,:]],
                    0).view(-1, 1, data_shape[1], data_shape[2])
            name = args.save + '/fixed_z_' + args.experiment_name + '.png'
            save_image(x_cat, name, nrow=10)


def save_2D_manifold(args, model, data_shape):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n = 15
    digit_size = data_shape[1]
    figure = np.zeros((digit_size*n, digit_size*n))

    grid_x = norm.ppf(np.linspace(0.05,0.95, n))
    grid_y = norm.ppf(np.linspace(0.05,0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = torch.Tensor([xi, yi])
            z_sample = z_sample.to(device)
            generated_sample = model.decode(z_sample)
            digit = generated_sample[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i+1)*digit_size, j * digit_size:(j + 1) *
                    digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap=’Greys’)
    plt.show()


