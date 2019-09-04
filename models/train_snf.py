#!/usr/bin/env python3

import torch

import lib_snf.module as module
import lib_snf.loss as loss_function

import lib.evaluation as evaluation

def fc_train(recon_images, labels, model_fc, optimizer_fc):
    model_fc.train()

    optimizer_fc.zero_grad()

    out = model_fc(recon_images)

    loss_func = nn.MSELoss()
    loss = loss_func(out, labels)

    loss.backward()

    optimizer_fc.step()


def run(args, logger, train_loader, validation_loader, data_shape):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available:
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = None

    print('Running on:' + str(device) + ' ' + str(device_name))

    model = module.HouseholderSylvesterVAE(args, data_shape)
    model_fc = module.VectorModel()

    model.to(device)
    model_fc.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer_fc = torch.optim.Adam(model_fc.parameters(), 0.001)

    print(model)
    print(model_fc)

    beta = 0.01
    break_training = 10

    for epoch in range(args.num_epochs):
        print('Epoch: {} \tBeta: {}'.format(epoch,beta))

        model.train()
        num_data = 0

        for idx_count, data in enumerate(train_loader):
            if idx_count > break_training:
                break

            if args.data == "piv":
                x, y = data['ComImages'].float(), data['AllGenDetails'].float()
                x = x.to(torch.device("cuda:0"))
            else:
                x, y = data
                x = x.to(torch.device("cuda:0"))

            optimizer.zero_grad()

            recon_images, z_mu, z_var, ldj, z0, z_k = model(x)

            loss, rec, kl = loss_function.binary_loss_function(recon_images, x, z_mu, z_var, z0, z_k, ldj, beta)

            loss.backward()

            optimizer.step()

            rec = rec.item()
            kl = kl.item()
            num_data += len(data)

            if idx_count%20==0:
                print('Epoch: {:3d} [{:5d}/{:5d} ({:2.0f}%)] \tLoss: {:11.6f}\trec:{:11.6f} \tkl: {:11.6f}'.format(
                            epoch,
                            num_data,
                            len(train_loader.sampler),
                            100.*idx_count/len(train_loader),
                            loss.item(),
                            rec,
                            kl)
                        )

            #  fc_train(recon_images, y)

        beta += 0.01
        evaluation.save_recon_images(args, model, validation_loader, data_shape)

