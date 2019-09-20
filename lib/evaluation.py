#!/usr/bin/env python3

import torch
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import csv
import datetime
import time
import numpy as np
from scipy.stats import norm


def save_recon_images(args, model, validation_loader, data_shape, logger):
    """Saves reconstructed images to samples folder."""

    with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        validation_set_enumerate = enumerate(validation_loader)

        _, data = next(validation_set_enumerate)

        if args.data == 'piv':
            x_, y_ = data['ComImages'],data['AllGenDetails']

            if args.heterogen:
                x = torch.zeros([x_.size(0), 4, 32, 32])
                x[:,:2,:,:] = x_
                for idx in range(x_.size(0)):
                    u_vector = torch.zeros([1,32,32])
                    u_vector.fill_((y_[idx][0]/20)*0.5 + 0.5)

                    v_vector = torch.zeros([1,32,32])
                    v_vector.fill_((y_[idx][1]/20)*0.5 + 0.5)

                    x[idx, 2,:,:] = u_vector
                    x[idx, 3,:,:] = v_vector

            else:
                x = x_

        elif args.data == 'mnist' and args.heterogen:
                x_,y_ = data

                x = torch.zeros([x_.size(0), 2, 28, 28])
                x[:,:1,:,:] = x_
                for idx in range(x_.size(0)):
                    labels = torch.zeros([1,28,28])
                    labels.fill_(y_[idx].item()/10)

                    x[idx, 1,:,:] = labels

        elif args.data == 'cifar10' and args.heterogen:
                x_,y_ = data

                x = torch.zeros([x_.size(0), 4, 32, 32])
                x[:,:3,:,:] = x_
                for idx in range(x_.size(0)):
                    labels = torch.zeros([1,32,32])
                    labels.fill_(y_[idx]/10)

                    x[idx, 3,:,:] = labels

        else:
            x, y = data

        x = x.to(device)

        if args.model == "ffjord":
            # In order to get the reconstructed images we need to run the model
            # generate z and run the model backward with given z to generate x
            zero = torch.zeros(x.shape[0], 1).to(x)
            z, delta_logp = model(x, zero)  # run model forward

            recon_img = model(z, reverse=True)
        elif args.model == "snf":
            recon_img, z_mu, z_var, ldj, z0, z_k = model(x)

        recon_img = recon_img.cpu()
        images = x.cpu()

        date_string = time.strftime("%Y-%m-%d-%H:%M")

        if args.data == "piv":
            if args.heterogen:
                logger.info("learned u vector {}, v vector {}".format(recon_img[0][2][0], recon_img[0][3][0]))
                logger.info("true u vector {}, v vector {}".format(images[0][2][0], images[0][3][0]))

            with open(args.save +'/' + args.experiment_name + '.csv',
                            mode='a') as label_file:
                        label_file_writer = csv.writer(label_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                        u_recon_list = [(recon_img[i][2][0][0].item()-0.5)*10 for i in
                                range(args.save_recon_images_size)]
                        v_recon_list = [(recon_img[i][3][0][0].item()-0.5)*10 for i in
                                range(args.save_recon_images_size)]

                        u_true_list = [(images[i][2][0][0].item()-0.5)*10 for i in
                                range(args.save_recon_images_size)]
                        v_true_list = [(images[i][3][0][0].item()-0.5)*10 for i in
                                range(args.save_recon_images_size)]

                        u_recon_list.insert(0,date_string + '_u_recon')
                        v_recon_list.insert(0,date_string + '_v_recon')

                        u_true_list.insert(0,date_string + '_u_true')
                        v_true_list.insert(0,date_string + '_v_true')

                        label_file_writer.writerow(u_recon_list)
                        label_file_writer.writerow(v_recon_list)

                        label_file_writer.writerow(u_true_list)
                        label_file_writer.writerow(v_true_list)


            x_cat_1 = torch.cat([images[:args.save_recon_images_size,0,:,:],
                recon_img[:args.save_recon_images_size,0,:,:]],
                    0).view(-1, 1, data_shape[1], data_shape[2])
            x_cat_2 = torch.cat([images[:args.save_recon_images_size,1,:,:],
                recon_img[:args.save_recon_images_size,1,:,:]],
                    0).view(-1, 1, data_shape[1], data_shape[2])

            images = [x_cat_1.cpu().data,x_cat_2.cpu().data]

            count = 0
            for image in images:
                name = args.save + '/reconstruction_' + args.experiment_name + '_' + date_string + '_' + str(count) + '.png'
                save_image(image, name, nrow=10 )
                count += 1

        elif args.data == "mnist":
            if args.heterogen:
                with open(args.save +'/' + args.experiment_name + '.csv',
                        mode='a') as label_file:
                    label_file_writer = csv.writer(label_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                    label_recon_list = [recon_img[i][1][0][0].item() for i in
                            range(args.save_recon_images_size)]
                    label_true_list = [images[i][1][0][0].item() for i in
                            range(args.save_recon_images_size)]

                    label_recon_list.insert(0,date_string+'_label_recon')
                    label_true_list.insert(0,date_string+'_label_true')

                    label_file_writer.writerow(label_recon_list)
                    label_file_writer.writerow(label_true_list)


                logger.info("learned labels {}".format(recon_img[0][1][0]))
                logger.info("true labels {}".format(images[0][1][0]))

            x_cat = torch.cat([images[:args.save_recon_images_size,0,:,:],
                recon_img[:args.save_recon_images_size,0,:,:]],
                    0).view(-1, 1, data_shape[1], data_shape[2])
            name = args.save + '/reconstruction_' + args.experiment_name + '_' + date_string + '.png'
            save_image(x_cat, name, nrow=10)

        elif args.data == "cifar10":
            if args.heterogen:
                with open(args.save +'/' + args.experiment_name + '.csv',
                        mode='a') as label_file:
                    label_file_writer = csv.writer(label_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                    label_recon_list = [recon_img[i][3][0][0].item() for i in
                            range(args.save_recon_images_size)]
                    label_true_list = [images[i][3][0][0].item() for i in
                            range(args.save_recon_images_size)]

                    label_recon_list.insert(0,date_string+'_label_recon')
                    label_true_list.insert(0,date_string+'_label_true')

                    label_file_writer.writerow(label_recon_list)
                    label_file_writer.writerow(label_true_list)


                logger.info("learned labels {}".format(recon_img[0][3][0]))
                logger.info("true labels {}".format(images[0][3][0]))

            x_cat = torch.cat([images[:args.save_recon_images_size,:3,:,:],
                recon_img[:args.save_recon_images_size,:3,:,:]],
                    0).view(-1, 1, data_shape[1], data_shape[2])
            name = args.save + '/reconstruction_' + args.experiment_name + '_' + date_string + '.png'
            save_image(x_cat, name, nrow=10)



def save_fixed_z_image(args, model, data_shape, logger, epoch):
    """ Save samples with fixed z. """

    number_img = 1

    with torch.no_grad():

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)

        if args.model == "ffjord":
            fixed_z = cvt(torch.randn(number_img, *data_shape))
            generated_sample = model(fixed_z, reverse=True)

        elif args.model == "snf":
            fixed_z = cvt(torch.randn(number_img, args.z_size))
            generated_sample = model.decode(fixed_z)

        ts = time.time()
        date_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        #  date_string = time.strftime("%Y-%m-%d-%H:%M")
        time.sleep(1)

        stemp = 0
        if args.data == "piv":

            if args.heterogen:
                logger.debug("fixed_z learned u vector {}, v vector{}".format(generated_sample[0][2][0], generated_sample[0][3][0]))

                with open(args.save +'/' + args.experiment_name + '_fixed_z.csv',
                            mode='a') as label_file:

                    label_file_writer = csv.writer(label_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                    u_recon_list = [(generated_sample[i][2][0][0].item()-0.5)*10 for i in
                            range(number_img)]
                    v_recon_list = [(generated_sample[i][3][0][0].item()-0.5)*10 for i in
                            range(number_img)]

                    u_recon_list.insert(0,str(epoch) + '_u_recon')
                    v_recon_list.insert(0,str(epoch) + '_v_recon')

                    label_file_writer.writerow(u_recon_list)
                    label_file_writer.writerow(v_recon_list)

            x_cat_1 = torch.cat([generated_sample[:number_img, 0,:,:]],0).view(-1, 1, data_shape[1], data_shape[2])
            x_cat_2 = torch.cat([generated_sample[:number_img, 1,:,:]],0).view(-1, 1, data_shape[1], data_shape[2])

            images = [x_cat_1.cpu().data,x_cat_2.cpu().data]

            count = 0
            for image in images:
                name = args.save + '/fixed_z_' + args.experiment_name + '_' + str(epoch) + '_' + str(count) + '.png'
                save_image(image, name, nrow=10 )
                count += 1

        elif args.data == "mnist":
            if args.heterogen:
                with open(args.save +'/' + args.experiment_name + '_fixed_z.csv',
                        mode='a') as label_file:
                    label_file_writer = csv.writer(label_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                    label_recon_list = [generated_sample[i][1][0][0].item() for i in
                            range(number_img)]

                    label_recon_list.insert(0,date_string+'_label_recon')

                    label_file_writer.writerow(label_recon_list)

            x_cat = torch.cat([generated_sample[:number_img,:1,:,:]],
                    0).view(-1, 1, data_shape[1], data_shape[2])
            name = args.save + '/fixed_z_' + args.experiment_name + '_' + date_string + '.png'
            save_image(x_cat, name, nrow=10)


        elif args.data == "cifar10":
            if args.heterogen:
                with open(args.save +'/' + args.experiment_name + '_fixed_z.csv',
                        mode='a') as label_file:
                    label_file_writer = csv.writer(label_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                    label_recon_list = [generated_sample[i][3][0][0].item() for i in
                            range(number_img)]

                    label_recon_list.insert(0,date_string+'_label_recon')

                    label_file_writer.writerow(label_recon_list)

            #  x_cat = generated_sample.view(-1. *data_shape)

            x_cat = torch.cat([generated_sample[:number_img,:1,:,:]],
                    0).view(-1, 1, data_shape[1], data_shape[2])
            name = args.save + '/fixed_z_' + args.experiment_name + '_' + date_string + '.png'
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
            recon_img = model.decode(z_sample)
            digit = recon_img[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i+1)*digit_size, j * digit_size:(j + 1) *
                    digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gray')
    plt.show()


