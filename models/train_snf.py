#!/usr/bin/env python3

import torch
import time
import numpy as np
import os

import lib_snf.module as module
import lib_snf.loss as loss_function
import lib.evaluation as evaluation
import lib.utils as utils
import models.resnet_pretrained as resnet_pretrained


def run(args, logger, train_loader, validation_loader, data_shape):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = module.HouseholderSylvesterVAE(args, data_shape)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0

    # restore parameters
    if args.resume is not None:
        checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpt["state_dict"])
        optimizer.load_state_dict(checkpt["optim_state_dict"])
        args = checkpt["args"]
        start_epoch = checkpt["epoch"] + 1
        logger.info("Resuming at epoch {} with args {}.".format(start_epoch,
            args))

    time_meter = utils.RunningAverageMeter(0.97)

    beta = 0.01
    break_training = 10

    best_loss = float("inf")
    itr = 0
    for epoch in range(start_epoch, args.num_epochs):
        logger.info('Epoch: {} \tBeta: {}'.format(epoch,beta))

        model.train()
        num_data = 0

        for idx_count, data in enumerate(train_loader):
            #  if idx_count > break_training:
                #  break

            if args.data == "piv":
                x, y = data['ComImages'].float(), data['AllGenDetails'].float()
                x = x.to(device)
            else:
                x, y = data
                x = x.to(device)

            start = time.time()
            optimizer.zero_grad()

            recon_images, z_mu, z_var, ldj, z0, z_k = model(x)

            loss, rec, kl = loss_function.binary_loss_function(recon_images, x, z_mu, z_var, z0, z_k, ldj, beta)

            loss.backward()

            optimizer.step()

            rec = rec.item()
            kl = kl.item()
            num_data += len(data)

            time_meter.update(time.time() - start)

            if itr % args.log_freq == 0:
                log_message = (
                    "Epoch {:03d} | Time {:.4f}({:.4f}) | [{:5d}/{:5d} ({:2.0f}%)] | Loss: {:11.6f} |"
                    "rec:{:11.6f} | kl: {:11.6f}".format(
                        epoch, time_meter.val, time_meter.avg, num_data, len(train_loader.sampler), 100.*idx_count/len(train_loader),
                        loss.item(), rec, kl)
                    )
                logger.info(log_message)

            itr += 1

        # Evaluate and save model
        if args.evaluate:
            if epoch % args.val_freq == 0:
                model.eval()
                with torch.no_grad():
                    start = time.time()
                    logger.info("validating...")

                    losses_vec_recon_images = []
                    losses_vec_images_recon_images = []
                    losses = []

                    for (data) in validation_loader:
                        if args.data == 'piv':
                            x, y = data['ComImages'],data['AllGenDetails']
                            x = x.to(device)
                        else:
                            x, y = data
                            x = x.to(device)

                        recon_images, z_mu, z_var, ldj, z0, z_k = model(x)
                        loss, rec, kl = loss_function.binary_loss_function(recon_images, x, z_mu, z_var, z0, z_k, ldj, beta)
                        losses.append(loss.item())


                        if args.data == "piv":
                            loss_vec_recon_images, loss_vec_images_recon_images = resnet_pretrained.run(args, logger,
                                    recon_images, x, y, data_shape)
                            losses_vec_recon_images.append(loss_vec_recon_images)
                            losses_vec_images_recon_images.append(loss_vec_images_recon_images)


                    if args.data == "piv":
                        logger.info("Loss vector reconstructed images {}, Loss
                                vector images reconstructed images
                                {}".format(np.mean(losses_vec_recon_images,
                                    losses_vec_images_recon_images)))

                    loss = np.mean(losses)
                    logger.info("Epoch {:04d} | Time {:.4f} | Loss {:.4f}".format(epoch, time.time() - start, loss))
                    if loss < best_loss:
                        best_loss = loss
                        utils.makedirs(args.save)
                        torch.save({
                            "args": args,
                            "epoch": epoch,
                            "state_dict":  model.state_dict(),
                            "optim_state_dict": optimizer.state_dict(),
                        }, os.path.join(args.save, "checkpt.pth"))
                        logger.info("Saving model at epoch {}.".format(epoch))

            beta += 0.01

            # Evaluation
            evaluation.save_recon_images(args, model, validation_loader, data_shape)
            evaluation.save_fixed_z_image(args, model, data_shape)

