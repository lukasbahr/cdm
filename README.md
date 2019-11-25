# On generative models learning complex distributions

<p align="center">
<img align="middle" src="./assets/mnist_samples.png" width="590" height="354" />
</p>

> One disadvantage of neural networks is the missing capability to estimate uncertainties. As highly expressive general function estimators neural networks can approximate arbitrarily complex functions but come at the drawback that epistemic uncertainty is not reflected. Usually, neural networks are trained by a maximum likelihood approach which only permits aleatoric uncertainty estimates inherent in the training data, simply because neural networks are not designed to account for epistemic uncertainty. Bayesian neural networks address this problem by using a prior probability distribution, e.g. Gaussian distribution, for the model parameters that is updated during training. Some priors converge to posterior distributions that can be approximated well, e.g., Gaussian processes, Brownian, frationally Brownian or non-Gaussian stable processes. However, in general, there is no guarantee that a complex posterior distribution can be covered precisely by a simple prior during training. Generative models are one of the most promising approaches to analyze and understand the true posterior distribution of a data set. Normalizing flows transform this simple Gaussian distribution into a more complex one. We show that a new approach with continuous-depth hidden states called Free-form Continuous Dynamics for Scalable Reversible GenerativeModels (FFJORD) extends this approach by making discreteflows continuous. We show that modern generative networks are capable of learning the observation and corresponding labels. This is shown by evaluating different generative models on a dataset with complex distribution.

>[continous_depth_models.pdf](https://github.com/lukasbahr/cdm/blob/master/assets/cdm.pdf)

## Prerequisites

Install `torchdiffeq` from https://github.com/rtqichen/torchdiffeq.

## Usage

Different scripts are provided for different datasets. To see all options, use the `-h` flag.

VAE Experiments (based on [Sylvester VAE](https://github.com/riannevdberg/sylvester-flows)) with flag 'snf' for --model.

### Homogeneous datasets

MNIST:
```
python train_cnf.py --model ffjord  --data mnist --dims 64,64,64 --strides 1,1,1,1 --num_blocks 2 --layer_type concat --multiscale True --rademacher True
```

CIFAR10:
```
python train_cnf.py --model ffjord  --data cifar10 --dims 64,64,64 --strides 1,1,1,1 --num_blocks 2 --layer_type concat --multiscale True --rademacher True
```

### Heterogeneous datasets 

MNIST:
```
python train_cnf.py --model ffjord  --data mnist --dims 64,64,64 --strides 1,1,1,1 --num_blocks 2 --layer_type concat --multiscale True --rademacher True --heterogen true
```
## Datasets

### VAE datasets
Follow instructions from https://github.com/riannevdberg/sylvester-flows and place them in `data/`.

