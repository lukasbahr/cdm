#!/usr/bin/env python3
import argparse
import os
import time
import numpy as np

import torch
import torchvision.datasets as dset
import torchvision.transforms as tforms

import lib.utils as utils
import lib.dataset as dataset

import models.train_cnf as train_cnf
import models.train_snf as train_snf


# ============================================================================
# Arguments general
# ============================================================================
torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser("Continuous Depth Models")

parser.add_argument("--model", choices=["ffjord", "snf"], type=str)
parser.add_argument("--data", choices=["piv","mnist", "cifar10"], type=str, default="mnist")
parser.add_argument("--resume", type=str, default=None)

parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=128) # Try 32, 64, 128, 256
parser.add_argument("--batch_size_schedule", type=str, default="", help="Increases the batchsize at every given epoch, dash separated.")
parser.add_argument("--test_batch_size", type=int, default=200)
parser.add_argument("--lr", type=float, default=1e-3)

# ============================================================================
# Arguments evaluation
# ============================================================================
parser.add_argument("--evaluate", type=bool, default=True)
parser.add_argument("--resnet_checkpoint", type=str, default=None)
parser.add_argument("--save_recon_images_size", type=int, default=8)
parser.add_argument("--experiment_name", type=str, default=None)
parser.add_argument("--save", type=str, default=None)

parser.add_argument("--val_freq", type=int, default=1)
parser.add_argument("--log_freq", type=int, default=10)

# ============================================================================
# Arguments for ffjord
# ============================================================================
SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams']
parser.add_argument("--dims", type=str, default="8,32,32,8")
parser.add_argument("--strides", type=str, default="2,2,1,-2,-2")
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')

parser.add_argument("--conv", type=eval, default=True, choices=[True, False])
parser.add_argument(
    "--layer_type", type=str, default="ignore",
    choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
)
parser.add_argument("--divergence_fn", type=str, default="approximate", choices=["brute_force", "approximate"])
parser.add_argument(
    "--nonlinearity", type=str, default="softplus", choices=["tanh", "relu", "softplus", "elu", "swish"]
)
parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
parser.add_argument('--atol', type=float, default=1e-5)
parser.add_argument('--rtol', type=float, default=1e-5)
parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")

parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
parser.add_argument('--test_atol', type=float, default=None)
parser.add_argument('--test_rtol', type=float, default=None)

parser.add_argument("--imagesize", type=int, default=None)
parser.add_argument("--alpha", type=float, default=1e-6)
parser.add_argument('--time_length', type=float, default=1.0)
parser.add_argument('--train_T', type=eval, default=True)

parser.add_argument("--warmup_iters", type=float, default=1000)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--spectral_norm_niter", type=int, default=10)

parser.add_argument("--add_noise", type=eval, default=True, choices=[True, False])
parser.add_argument("--batch_norm", type=eval, default=False, choices=[True, False])
parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
parser.add_argument('--autoencode', type=eval, default=False, choices=[True, False])
parser.add_argument('--rademacher', type=eval, default=True, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--multiscale', type=eval, default=False, choices=[True, False])
parser.add_argument('--parallel', type=eval, default=False, choices=[True, False])

# Regularizations
parser.add_argument('--l1int', type=float, default=None, help="int_t ||f||_1")
parser.add_argument('--l2int', type=float, default=None, help="int_t ||f||_2")
parser.add_argument('--dl2int', type=float, default=None, help="int_t ||f^T df/dt||_2")
parser.add_argument('--JFrobint', type=float, default=None, help="int_t ||df/dx||_F")
parser.add_argument('--JdiagFrobint', type=float, default=None, help="int_t ||df_i/dx_i||_F")
parser.add_argument('--JoffdiagFrobint', type=float, default=None, help="int_t ||df/dx - df_i/dx_i||_F")

parser.add_argument("--time_penalty", type=float, default=0, help="Regularization on the end_time.")
parser.add_argument("--max_grad_norm", type=float, default=1e10,
    help="Max norm of graidents (default is just stupidly high to avoid any clipping)")

# ============================================================================
# Arguments for snf
# ============================================================================
parser.add_argument('--z_size', type=int, default=64, metavar='ZSIZE', help='how many stochastic hidden units')
parser.add_argument('--num_flows', type=int, default=4,metavar='NUM_FLOWS', help='Number of flow layers, ignored in absence of flows')
parser.add_argument('--num_householder', type=int, default=8, metavar='NUM_HOUSEHOLDERS',help=""" For Householder Sylvester flow: Number of Householder matrices per flow. Ignored for other flow types.""")


args = parser.parse_args()

if args.model == "ffjord" and args.save == None:
    args.save = "experiments/ffjord"
elif args.model == "snf" and args.save == None:
    args.save = "experiments/snf"

if args.data == "piv" and args.experiment_name == None:
    args.experiment_name = "piv"
elif args.data == "mnist" and args.experiment_name == None:
    args.experiment_name = "mnist"
elif args.data == "cifar10" and args.experiment_name == None:
    args.experiment_name = "cifar10"

if torch.cuda.is_available:
    device_name = torch.cuda.get_device_name(0)
    device = torch.device("cuda:0")
else:
    device_name = None
    device = torch.device("cpu")

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
logger.info('Running on:' + str(device) + ' ' + str(device_name))

if args.layer_type == "blend":
    logger.info("!! Setting time_length from None to 1.0 due to use of Blend layers.")
    args.time_length = 1.0

logger.info(args)


# ============================================================================
# Data handling
# ============================================================================
def add_noise(x):
    """
    [0, 1] -> [0, 255] -> add noise -> [0, 1]
    """
    if args.add_noise:
        noise = x.new().resize_as_(x).uniform_()
        x = x * 255 + noise
        x = x / 256
    return x


def get_train_loader(train_set, epoch):
    if args.batch_size_schedule != "":
        epochs = [0] + list(map(int, args.batch_size_schedule.split("-")))
        n_passed = sum(np.array(epochs) <= epoch)
        current_batch_size = int(args.batch_size * n_passed)
    else:
        current_batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=current_batch_size, shuffle=True, drop_last=True, pin_memory=True
    )
    logger.info("===> Using batch size {}. Total {} iterations/epoch.".format(current_batch_size, len(train_loader)))
    return train_loader


def get_dataset(args):
    trans = lambda im_size: tforms.Compose([tforms.Resize(im_size), tforms.ToTensor(), add_noise])

    if args.data == "mnist":
        im_dim = 1
        im_size = 28 if args.imagesize is None else args.imagesize
        train_set = dset.MNIST(root="./data", train=True, transform=trans(im_size), download=True)
        test_set = dset.MNIST(root="./data", train=False, transform=trans(im_size), download=True)
    elif args.data == "piv":
        im_dim = 2
        im_size = 32 if args.imagesize is None else args.imagesize
        train_set = dataset.H5Dataset("/home/bahr/cdm/data/ISPIV_dataset/Batch_Training-Dataset_2Labels_S12_SynthImg_Alex.hdf5")
        test_set = dataset.H5Dataset("/home/bahr/cdm/data/ISPIV_dataset/Batch_Validation-Dataset_2Labels_S12_SynthImg_Alex.hdf5")
    elif args.data == "cifar10":
        im_dim = 3
        im_size = 32 if args.imagesize is None else args.imagesize
        train_set = dset.CIFAR10(
            root="./data", train=True, transform=tforms.Compose([
                tforms.Resize(im_size),
                tforms.RandomHorizontalFlip(),
                tforms.ToTensor(),
                add_noise,
            ]), download=True
        )
        test_set = dset.CIFAR10(root="./data", train=False, transform=trans(im_size), download=True)

    data_shape = (im_dim, im_size, im_size)
    if not args.conv:
        data_shape = (im_dim * im_size * im_size,)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=args.test_batch_size, shuffle=False, drop_last=True
    )
    return train_set, test_loader, data_shape


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":

    train_set, test_loader, data_shape = get_dataset(args)
    train_loader = get_train_loader(train_set, args.num_epochs)

    if args.model == "ffjord":
        train_cnf.run(args, logger, train_loader, test_loader, data_shape)
    elif args.model == "snf":
        train_snf.run(args, logger, train_loader, test_loader, data_shape)
