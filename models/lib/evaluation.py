#!/usr/bin/env python3

import torch
from torchvision.utils import save_image

import numpy as np


def save_recon_images(model, validation_loader):
    """Saves reconstructed images to samples folder."""


    validation_set_enumerate = enumerate(validation_loader)

    i, data = next(validation_set_enumerate)
    images, labels = data['ComImages'] ,data['AllGenDetails']

    images = images.to(torch.device("cuda:0"))


    generated_sample = model(images, reverse=True)
    print(generated_sample.size())

    generated_sample = generated_sample.cpu()
    images = images.cpu()

    x_cat_1 = torch.cat([images[:8,0,:,:], generated_sample[:8,0,:,:]], 0).view(-1, 1, 32, 32)
    x_cat_2 = torch.cat([images[:8,1,:,:], generated_sample[:8,1,:,:]], 0).view(-1, 1, 32, 32)

    images = [x_cat_1.cpu().data,x_cat_2.cpu().data]

    count = 0
    for i in images:
        name = 'samples/reconstruction_' + str(count) + '.png'
        save_image(
                i,
                name,
                nrow=8
                )
        count += 1

