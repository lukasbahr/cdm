#!/usr/bin/env python3

import argparse
import os
import time
import numpy as np

import torch
import torch.optim as optim

import lib.evaluation as evaluation
import lib.utils as utils

import lib_ffjord.layers as layers
import lib_ffjord.odenvp as odenvp
import lib_ffjord.multiscale_parallel as multiscale_parallel

from lib_ffjord.train_misc import standard_normal_logprob
from lib_ffjord.train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from lib_ffjord.train_misc import add_spectral_norm, spectral_norm_power_iteration
from lib_ffjord.train_misc import create_regularization_fns, get_regularization, append_regularization_to_log

import models.resnet_pretrained as resnet_pretrained


def update_lr(args, optimizer, itr):
    iter_frac = min(float(itr + 1) / max(args.warmup_iters, 1), 1.0)
    lr = args.lr * iter_frac
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def compute_bits_per_dim(x, model):
    zero = torch.zeros(x.shape[0], 1).to(x)

    # Don't use data parallelize if batch size is small.
    # if x.shape[0] < 200:
    #     model = model.module

    z, delta_logp = model(x, zero)  # run model forward

    logpz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
    logpx = logpz - delta_logp

    logpx_per_dim = torch.sum(logpx) / x.nelement()  # averaged over batches
    bits_per_dim = -(logpx_per_dim - np.log(256)) / np.log(2)

    return bits_per_dim


def create_model(args, data_shape, regularization_fns):
    hidden_dims = tuple(map(int, args.dims.split(",")))
    strides = tuple(map(int, args.strides.split(",")))

    if args.multiscale:
        model = odenvp.ODENVP(
            (args.batch_size, *data_shape),
            n_blocks=args.num_blocks,
            intermediate_dims=hidden_dims,
            nonlinearity=args.nonlinearity,
            alpha=args.alpha,
            cnf_kwargs={"T": args.time_length, "train_T": args.train_T, "regularization_fns": regularization_fns},
        )
    elif args.parallel:
        model = multiscale_parallel.MultiscaleParallelCNF(
            (args.batch_size, *data_shape),
            n_blocks=args.num_blocks,
            intermediate_dims=hidden_dims,
            alpha=args.alpha,
            time_length=args.time_length,
        )
    else:
        if args.autoencode:

            def build_cnf():
                autoencoder_diffeq = layers.AutoencoderDiffEqNet(
                    hidden_dims=hidden_dims,
                    input_shape=data_shape,
                    strides=strides,
                    conv=args.conv,
                    layer_type=args.layer_type,
                    nonlinearity=args.nonlinearity,
                )
                odefunc = layers.AutoencoderODEfunc(
                    autoencoder_diffeq=autoencoder_diffeq,
                    divergence_fn=args.divergence_fn,
                    residual=args.residual,
                    rademacher=args.rademacher,
                )
                cnf = layers.CNF(
                    odefunc=odefunc,
                    T=args.time_length,
                    regularization_fns=regularization_fns,
                    solver=args.solver,
                )
                return cnf
        else:

            def build_cnf():
                diffeq = layers.ODEnet(
                    hidden_dims=hidden_dims,
                    input_shape=data_shape,
                    strides=strides,
                    conv=args.conv,
                    layer_type=args.layer_type,
                    nonlinearity=args.nonlinearity,
                )
                odefunc = layers.ODEfunc(
                    diffeq=diffeq,
                    divergence_fn=args.divergence_fn,
                    residual=args.residual,
                    rademacher=args.rademacher,
                )
                cnf = layers.CNF(
                    odefunc=odefunc,
                    T=args.time_length,
                    train_T=args.train_T,
                    regularization_fns=regularization_fns,
                    solver=args.solver,
                )
                return cnf

        chain = [layers.LogitTransform(alpha=args.alpha)] if args.alpha > 0 else [layers.ZeroMeanTransform()]
        chain = chain + [build_cnf() for _ in range(args.num_blocks)]
        if args.batch_norm:
            chain.append(layers.MovingBatchNorm2d(data_shape[0]))
        model = layers.SequentialFlow(chain)
    return model


def run(args, logger, train_loader, validation_loader, data_shape):

    # get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)

    # build model
    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = create_model(args, data_shape, regularization_fns)

    if args.spectral_norm: add_spectral_norm(model, logger)
    set_cnf_options(args, model)

    logger.info(model)
    logger.info("Number of trainable parameters: {}".format(count_parameters(model)))

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0

    # restore parameters
    if args.resume is not None:
        checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpt["state_dict"])
        if "optim_state_dict" in checkpt.keys():
                optimizer.load_state_dict(checkpt["optim_state_dict"])
                # Manually move optimizer state to device.
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = cvt(v)
        args = checkpt["args"]
        start_epoch = checkpt["epoch"] + 1
        logger.info("Resuming at epoch {} with args {}.".format(start_epoch,
            args))

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    time_meter = utils.RunningAverageMeter(0.97)
    loss_meter = utils.RunningAverageMeter(0.97)
    steps_meter = utils.RunningAverageMeter(0.97)
    grad_meter = utils.RunningAverageMeter(0.97)
    tt_meter = utils.RunningAverageMeter(0.97)

    if args.spectral_norm and not args.resume: spectral_norm_power_iteration(model, 500)

    best_loss = float("inf")

    itr = 0
    train_loader_break = 3000
    break_train = int(train_loader_break/args.batch_size)
    break_training = 50

    for epoch in range(start_epoch, args.num_epochs):
        logger.info("Epoch {}/{}".format(epoch, args.num_epochs))
        model.train()
        for idx_count, (data) in enumerate(train_loader):
            #  if idx_count > break_train:
                #  break

            if args.data == 'piv':
                x_, y_ = data['ComImages'],data['AllGenDetails']

                if args.heterogen:
                    x = torch.zeros([x_.size(0), 4, 32, 32])
                    x[:,:2,:,:] = x_
                    for idx in range(x_.size(0)):
                        u_vector = torch.zeros([1,32,32])
                        u_vector.fill_(y_[idx][0]/20*0.5 + 0.5)

                        v_vector = torch.zeros([1,32,32])
                        v_vector.fill_(y_[idx][1]/20*0.5 + 0.5)

                        x[idx, 2,:,:] = u_vector
                        x[idx, 3,:,:] = v_vector

                    #  import pdb; pdb.set_trace()

                else:
                    x = x_
                    y = y_

            elif args.data == 'mnist' and args.heterogen:
                    x_,y_ = data

                    x = torch.zeros([x_.size(0), 2, 28, 28])
                    x[:,:1,:,:] = x_
                    for idx in range(x_.size(0)):
                        labels = torch.zeros([1,28,28])
                        labels.fill_(y_[idx])

                        x[idx, 1,:,:] = labels

            elif args.data == 'cifar10' and args.heterogen:
                    x_,y_ = data

                    x = torch.zeros([x_.size(0), 4, 32, 32])
                    x[:,:3,:,:] = x_
                    for idx in range(x_.size(0)):
                        labels = torch.zeros([1,32,32])
                        labels.fill_(y_[idx])

                        x[idx, 3,:,:] = labels

            else:
                x, y = data

            start = time.time()
            update_lr(args, optimizer, itr)
            optimizer.zero_grad()

            if not args.conv:
                x = x.view(x.shape[0], -1)

            # cast data and move to device
            x = cvt(x)

            # compute loss
            loss = compute_bits_per_dim(x, model)
            if regularization_coeffs:
                reg_states = get_regularization(model, regularization_coeffs)
                reg_loss = sum(
                    reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
                )
                loss = loss + reg_loss
            total_time = count_total_time(model)
            loss = loss + total_time * args.time_penalty

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()

            if args.spectral_norm: spectral_norm_power_iteration(model, args.spectral_norm_niter)

            time_meter.update(time.time() - start)
            loss_meter.update(loss.item())
            steps_meter.update(count_nfe(model))
            grad_meter.update(grad_norm)
            tt_meter.update(total_time)

            if itr % args.log_freq == 0:
                log_message = (
                    "Iter {:04d} | Time {:.4f}({:.4f}) | Bit/dim {:.4f}({:.4f}) | "
                    "Steps {:.0f}({:.2f}) | Grad Norm {:.4f}({:.4f}) | Total Time {:.2f}({:.2f})".format(
                        itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg, steps_meter.val,
                        steps_meter.avg, grad_meter.val, grad_meter.avg, tt_meter.val, tt_meter.avg
                    )
                )
                if regularization_coeffs:
                    log_message = append_regularization_to_log(log_message, regularization_fns, reg_states)
                logger.info(log_message)

            itr += 1

        # Evaluate and save model
        if args.evaluate:
            if epoch % args.val_freq == 0:
                model.eval()
                with torch.no_grad():
                    start = time.time()
                    logger.info("validating...")

                    losses = []
                    losses_vec_recon_images = []
                    losses_vec_images_recon_images = []

                    for _,(data) in enumerate(validation_loader):
                    #  for _,(data) in enumerate(train_loader):
                        if  _ > break_training:
                            break
                        if args.data == 'piv':
                            x_, y_ = data['ComImages'],data['AllGenDetails']

                            if args.heterogen:
                                x = torch.zeros([x_.size(0), 4, 32, 32])
                                x[:,:2,:,:] = x_
                                for idx in range(x_.size(0)):
                                    u_vector = torch.zeros([1,32,32])
                                    u_vector.fill_(y_[idx][0]/20*0.5 + 0.5)

                                    v_vector = torch.zeros([1,32,32])
                                    v_vector.fill_(y_[idx][1]/20*0.5 + 0.5)

                                    x[idx, 2,:,:] = u_vector
                                    x[idx, 3,:,:] = v_vector

                            else:
                                x = x_
                                y = y_

                        elif args.data == 'mnist' and args.heterogen:
                            x_,y_ = data

                            x = torch.zeros([x_.size(0), 2, 28, 28])
                            x[:,:1,:,:] = x_
                            for idx in range(x_.size(0)):
                                labels = torch.zeros([1,28,28])
                                labels.fill_(y_[idx])

                                x[idx, 1,:,:] = labels

                        elif args.data == 'cifar10' and args.heterogen:
                            x_,y_ = data

                            x = torch.zeros([x_.size(0), 4, 32, 32])
                            x[:,:3,:,:] = x_
                            for idx in range(x_.size(0)):
                                labels = torch.zeros([1,32,32])
                                labels.fill_(y_[idx])

                                x[idx, 3,:,:] = labels

                        else:
                            x, y = data

                        if not args.conv:
                            x = x.view(x.shape[0], -1)
                        x = cvt(x)

                        zero = torch.zeros(x.shape[0], 1).to(x)
                        z, delta_logp = model(x, zero)  # run model forward

                        recon_images = model(z, reverse=True)
                        loss = compute_bits_per_dim(x, model)
                        losses.append(loss.item())

                        if args.data == "piv" and args.heterogen == False:
                            loss_vec_recon_images, loss_vec_images_recon_images = resnet_pretrained.run(args, logger,
                                    recon_images, x, y, data_shape)
                            losses_vec_recon_images.append(loss_vec_recon_images.item())
                            losses_vec_images_recon_images.append(loss_vec_images_recon_images.item())


                    if args.data == "piv" and args.heterogen == False:
                        logger.info("Loss vector reconstructed images {}, Loss vector images reconstructed images {}".format(np.mean(losses_vec_recon_images),
                                    np.mean(losses_vec_images_recon_images)))

                    loss = np.mean(losses)
                    logger.info("Epoch {:04d} | Time {:.4f}, Bit/dim {:.4f}".format(epoch, time.time() - start, loss))
                    if loss < best_loss:
                        best_loss = loss
                        torch.save({
                            "args": args,
                            "epoch": epoch,
                            "state_dict": model.module.state_dict() if torch.cuda.is_available() else model.state_dict(),
                            "optim_state_dict": optimizer.state_dict(),
                        }, os.path.join(args.save, "checkpt.pth"))
                        logger.info("Saving model at epoch {}.".format(epoch))

            # visualize samples and density
            evaluation.save_recon_images(args, model, validation_loader,
                    data_shape, logger)
            evaluation.save_fixed_z_image(args, model, data_shape, logger)

