#!/usr/bin/env python3

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def log_normal_standard(x, dim=None):
    log_norm = -0.5 * x * x

    return torch.sum(log_norm, dim)


def log_normal_diag(x, mean, log_var, dim=None):
    log_norm_diag = -0.5 * (log_var + ((x - mean) **2) * log_var.exp().reciprocal())

    return torch.sum(log_norm_diag, dim)


def binary_loss_function(recon_x, x, z_mu, z_var, z_0, z_k, ldj, beta):
    """
    Computes the binary loss function while summing over batch dimension, not averaged!
    :param recon_x: shape: (batch_size, num_channels, pixel_width, pixel_height), bernoulli parameters p(x=1)
    :param x: shape (batchsize, num_channels, pixel_width, pixel_height), pixel values rescaled between [0, 1].
    :param z_mu: mean of z_0
    :param z_var: variance of z_0
    :param z_0: first stochastic latent variable
    :param z_k: last stochastic latent variable
    :param ldj: log det jacobian
    :param beta: beta for kl loss
    :return: loss, ce, kl
    """

    reconstruction_function = nn.BCELoss(reduction='sum')

    batch_size = x.size(0) + x.size(0)

    # - N E_q0 [ ln p(x|z_k) ]
    bce = reconstruction_function(recon_x, x)

    # ln p(z_k)  (not averaged)
    log_p_zk = log_normal_standard(z_k, dim=1)
    # ln q(z_0)  (not averaged)
    log_q_z0 = log_normal_diag(z_0, mean=z_mu, log_var=z_var.log(), dim=1)
    # N E_q0[ ln q(z_0) - ln p(z_k) ]
    summed_logs = torch.sum(log_q_z0 - log_p_zk)

    # sum over batches
    summed_ldj = torch.sum(ldj)

    # ldj = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ]
    kl = (summed_logs - summed_ldj)
    loss = bce + beta * kl

    loss = loss / float(batch_size)
    bce = bce / float(batch_size)
    kl = kl / float(batch_size)

    return loss, bce, kl


