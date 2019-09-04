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

    itr = 0
    for epoch in range(args.num_epochs):
        print('Epoch: {} \tBeta: {}'.format(epoch,beta))

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

            optimizer.zero_grad()

            recon_images, z_mu, z_var, ldj, z0, z_k = model(x)

            loss, rec, kl = loss_function.binary_loss_function(recon_images, x, z_mu, z_var, z0, z_k, ldj, beta)

            loss.backward()

            optimizer.step()

            rec = rec.item()
            kl = kl.item()
            num_data += len(data)
            if itr % args.log_freq == 0:
                log_message = (
                    "Epoch {:03d} | [{:5d}/{:5d} ({:2.0f}%)] | Loss: {:11.6f} | "
                    "rec:{:11.6f} | kl: {:11.6f}".format(
                        epoch, num_data, len(train_loader.sampler), 100.*idx_count/len(train_loader),
                        loss.item(), rec, kl)
                    )
                logger.info(log_message)

            itr += 1

        # Evaluate and save model
        if epoch % args.val_freq == 0 and args.evaluation_numerical:
            model.eval()
            with torch.no_grad():
                start = time.time()
                logger.info("validating...")
                losses = []
                for (data) in test_loader:
                    if args.data == 'piv':
                        x, y = data['ComImages'],data['AllGenDetails']
                    else:
                        x, y = data

                    recon_images, z_mu, z_var, ldj, z0, z_k = model(x)
                    loss, rec, kl = loss_function.binary_loss_function(recon_images, x, z_mu, z_var, z0, z_k, ldj, beta)
                    losses.append(loss.item())

                loss = np.mean(losses)
                logger.info("Epoch {:04d} | Time {:.4f}, Bit/dim {:.4f}".format(epoch, time.time() - start, loss))
                if loss < best_loss:
                    best_loss = loss
                    utils.makedirs(args.save)
                    torch.save({
                        "args": args,
                        "state_dict": model.module.state_dict() if torch.cuda.is_available() else model.state_dict(),
                        "optim_state_dict": optimizer.state_dict(),
                    }, os.path.join(args.save, "checkpt.pth"))

            #  fc_train(recon_images, y)

        beta += 0.01

        # Evaluation
        evaluation.save_recon_images(args, model, validation_loader, data_shape)
        evaluation.save_fixed_z_image(args, model, data_shape)

