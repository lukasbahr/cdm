#!/usr/bin/env python3

import torch.nn as nn
import torch
from torchvision import datasets, transforms
import numpy as np

import lib.loss as loss_function

def fc_train(recon_images, labels, model_fc, optimizer_fc, loss_func):

    labels = labels.to(torch.device("cuda:0"))
    recon_images = recon_images.view(recon_images.size(0), -1)

    optimizer_fc.zero_grad()

    out = model_fc(recon_images)

    #  loss_func = nn.MSELoss()
    loss = loss_func(out, labels)

    loss.backward()

    optimizer_fc.step()

    return loss.item()


def gm_train(epoch, train_loader, model, optimizer,beta):

    model.train()
    num_data = 0

    #  break_training = 2000
    for i, data in enumerate(train_loader):
        #  if i > break_training:
            #  break
        images, labels = data['ComImages'].float(),data['AllGenDetails'].float()

        images = images.to(torch.device("cuda:0"))


        optimizer.zero_grad()


        recon_images, z_mu, z_var, ldj, z0, z_k = model(images)


        loss, rec, kl = loss_function.binary_loss_function(recon_images, images, z_mu, z_var, z0, z_k, ldj, beta)

        #  loss.backward(retain_graph = True)
        loss.backward()

        optimizer.step()

        rec = rec.item()
        kl = kl.item()
        num_data += len(data)

        loss_fc = 0

#          if epoch > 4:
            #  loss_fc = fc_train(recon_images, labels, model_fc, optimizer_fc, loss_func)
#
        if i%20==0:
            print('Epoch: {:3d} [{:5d}/{:5d} ({:2.0f}%)] \tLoss:{:11.6f}\trec:{:11.6f} \tkl: {:11.6f} \tLoss_fc: {:11.6f}'.format(
                        epoch,
                        num_data,
                        len(train_loader.sampler),
                        100.*i/len(train_loader),
                        loss.item(),
                        rec,
                        kl,
                        loss_fc)
                    )

    return loss.item(), recon_images, labels


