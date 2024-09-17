# Unsupervised End-to-End Training with a Self-defined Target

In this work, we introduce a 'self-defined target' at the network's last layer to realize unsupervised end-to-end training. 
This target is defined by Winner-Take-All (WTA) selectivity combined with homeostasis mechanism. 
We calculate the unsupervised global mean square error (MSE) loss, which is the difference between the network's output 
and our unsupervisedly defined target, and update the weight with the gradient-descent based algorithms.

This approach, framework-agnostic and compatible with both global and local learning rules, 
in our case, Backpropation (BP) and Equilibrium propagation (EP) seperately, achieves a 97.6\% test accuracy on the MNIST dataset. 
We also train a CNN using our unsupervised method (with BP only), achieving test accuracies of 99.2% on MNIST, 90.3% on Fashion-MNIST, and 81.5% on the SVHN dataset.
Extending to semi-supervised learning, our method dynamically adjusts the target according to data availability, reaching a 96.6\% accuracy with just 600 labeled MNIST samples. 

_In unsupervised training_: 
- We train two distinct network architectures using our unsupervised learning framework: a **one-layer** network (784-2000) and **two-layer** network (784-2000-2000);  
- We train a convolutional network (32f5-128f3-3000) using our unsupervised target, with weight updates performed using backpropagation.  
- The final accuracy of the trained network is obtained through two different classification method: **direct association** and **linear classifier**.

_In semi-supervised training_:
- We keep the same architecture used by [Lee, 2013](https://www.researchgate.net/publication/280581078_Pseudo-Label_The_Simple_and_Efficient_Semi-Supervised_Learning_Method_for_Deep_Neural_Networks), which is a 784-5000-10 muti-layer perceptron;
- We test the performance in five scenarios with different amounts of available labeled data: 100, 300, 600, 1000, 3000 randomly selected labelled mnist samples. 

## Getting Started

### Prerequisites
python >= 3.8  
[pytorch version 2.2.1](https://pytorch.org/)  
[cuda 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive) (if you use GPU to accelerate)
### Installing
Clone the repository:  
`git clone https://github.com/yourusername/projectname.git`

#### Packages for training:
[skit-learn](https://scikit-learn.org/stable/install.html):
`pip install -U scikit-learn`  
[Image](https://pypi.org/project/image/):
`pip install image`  
[Optuna](https://optuna.org/):
`pip install optuna`  
[pandas](https://pypi.org/project/pandas/):
`pip install pandas`  

#### Packages for plotting:
[jupyter notebook](https://jupyter.org/install):
`pip install notebook`  
[matplotlib](https://matplotlib.org/):
`pip install matplotlib`  
[seaborn](https://seaborn.pydata.org/):
`pip install seaborn`  
_For plotting the Fig.3, brokenaxes package should be installed:_  
`pip install brokenaxes`

_For plotting the Fig.8, UMAP package should be installed:_  
`pip install umap-learn`

## File organization
- bp/: code of the backpropagation training.
- ep/: code of the equilibrium propagation training.
- funcs/: common functions shared by bp and ep training.
- configFile/: config files.
- simuResults/: our trained simulation result files.
- figuresPlot/: plotted figures in the article.
- reproduce_figures.ipynb: jupyter file to reproduce the figures.  

_The following two folders will be generated after the training :_

- data/: data used for training.  
- DATA-0/: training results.


## Training

### Parameters explanation

_Parameters defined in **config.json**._

| Name                     | Description                                                                                                        | Examples                                                                                                         |
|--------------------------|--------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `device`                 | Execution device: CPU (`-1`) or GPU (device number).                                                               | `-1`, `0`, `1`                                                                                                   |
| `dataset`                | Dataset for training.                                                                                              | `"mnist"`, `"fashionMnist"`, `"cifar10"`                                                                         |
| `n_class`                | Number of dataset classes.                                                                                         | `10`                                                                                                             |
| `action`                 | Training type: BP or EP, supporting various modes.                                                                 | BP: `"bp"`, `"unsupervised_bp"`, `"semi_supervised_bp"`; EP: `"ep"`, `"unsupervised_ep"`, `"semi_supervised_ep"` |
| `epochs`                 | Total training epochs.                                                                                             | `200`                                                                                                            |
| `batchSize`              | Mini batch size for training; set to 1 for unsupervised sequential mode.                                           | `256`                                                                                                            |
| `test_batchSize`         | Mini batch size for testing.                                                                                       | `512`                                                                                                            |
| `cnn`                    | Whether use a convolutional network.                                                                               | `false`, `true`                                                                                                  |
| `fcLayers`               | Architecture of fully connected layers (input, hidden, output).                                                    | `[784, 2000, 2000]`                                                                                              |
| `lr`                     | Layer-specific learning rates.                                                                                     | `[0.5, 0.3]`                                                                                                     |
| `activation_function`    | Layer-specific activation in BP; identical across all EP layers.                                                   | BP: `["x", "relu", "hardsigm"]`; EP: `"hardsigm"`                                                                |
| `optimizer`              | Optimizer for training, specifying the unsupervised optimizer in semi-supervised setups.                           | `"Adam"`, `"SGD"`                                                                                                |
| `loss`                   | Loss function.                                                                                                     | `"MSE"`, `"Cross-entropy"`                                                                                       |
| `eta`                    | Exponential moving average parameter for unsupervised stochastic mode.                                             | `0.6`                                                                                                            |
| `gamma`                  | Coefficient for homeostasis in unsupervised training.                                                              | `0.5`                                                                                                            |
| `nudge`                  | Number of winners in unsupervised target (k-WTA).                                                                  | `3`                                                                                                              |
| `smooth`                 | Enable label smoothing.                                                                                            | `true`, `false`                                                                                                  |
| `dropProb`               | Dropout probability per layer; 0 disables dropout.                                                                 | `[0.2, 0.2, 0.2]`                                                                                                |
| `semi_seed`              | Method to separate labeled/unlabeled data in semi-supervised learning (`-1` for random, positive for seed number). | `-1`, `13`                                                                                                       |
| `train_label_number`     | Labeled data count in semi-supervised learning.                                                                    | `100`, `300`, `600`, `1000`, `3000`                                                                              |
| `pre_epochs`             | Pre-training epochs (N<sub>1</sub>) for semi-supervised learning.                                                  | `80`                                                                                                             |
| `pre_batchSize`          | Mini batch size for pre-training in semi-supervised learning.                                                      | `32`                                                                                                             |
| `pre_lr`                 | Learning rate for pre-training in semi-supervised learning.                                                        | `[0.001, 0.003]`                                                                                                 |
| `class_label_percentage` | Label percentage for class association in unsupervised training.                                                   | `0.01`                                                                                                           |
| `class_activation`       | Activation function for added linear classifier.                                                                   | `"x"`                                                                                                            |
| `class_lr`               | Learning rate for linear classifier.                                                                               | `[0.1]`                                                                                                          |
| `class_epoch`            | Training epochs for linear classifier.                                                                             | `50`                                                                                                             |
| `class_optimizer`        | Optimizer for linear classifier training.                                                                          | `"Adam"`                                                                                                         |
| `class_dropProb`         | Dropout probability at classifier input; 0 disables dropout.                                                       | `0.4`                                                                                                            |
| `class smooth`           | Enable label smoothing for classifier training.                                                                    | `true`, `false`                                                                                                  |
 
_Some parameters especially used in EP training:_ 

| Name             | Description                                                            | Examples                 |
|------------------|------------------------------------------------------------------------|--------------------------|
| `dt`             | Discretization time interval.                                          | `0.2`                    |
| `T`              | Free phase duration, measured in time steps.                           | `60`                     |
| `Kmax`           | Nudging phase duration, in time steps.                                 | `10`                     |
| `beta`           | Nudging coefficient, within the range [0, 1].                          | `0.5`                    |
| `clamped`        | Restrict neuron values to the range [0, 1].                            | `true`, `false`          |
| `error_estimate` | Error estimation strategy: either single-phase or symmetric two-phase. | `one-sided`, `symmetric` |
| `dropout`        | Enable dropout regularization.                                         | `true`, `false`          |

### Commands

The commands should be executed at the same level as the README file in bash script.

#### Unsupervised BP
_Train a one-layer network:_  
``python -m bp.main --json_path "./configFile/unsupervised_bp/1layer" ``  
_Train a two-layer network:_  
``python -m bp.main --json_path "./configFile/unsupervised_bp/2layer" ``  
_Train a CNN for MNIST dataset:_  
``python -m bp.main --json_path "./configFile/unsupervised_bp/cnn/mnist" ``  
_Train a CNN for Fashion-MNIST dataset:_  
``python -m bp.main --json_path "./configFile/unsupervised_bp/cnn/fashionMnist" ``  
_Train a CNN for SVHN dataset:_  
``python -m bp.main --json_path "./configFile/unsupervised_bp/cnn/svhn" ``  
_Train a CNN for CIFAR-10 dataset:_  
``python -m bp.main --json_path "./configFile/unsupervised_bp/cnn/cifar10" ``  


#### Unsupervised EP 
_Train a one-layer network:_  
``python -m ep.main --json_path "./configFile/unsupervised_ep/1layer" ``  
_Train a two-layer network:_   
``python -m ep.main --json_path "./configFile/unsupervised_ep/2layer" ``  

#### Semi-supervised BP
_Different configuration files are used for various labeled number scenarios._  
``python -m bp.main --json_path "./configFile/semi_bp/100" ``  
``python -m bp.main --json_path "./configFile/semi_bp/300" ``  
``python -m bp.main --json_path "./configFile/semi_bp/600" ``  
``python -m bp.main --json_path "./configFile/semi_bp/1000" ``  
``python -m bp.main --json_path "./configFile/semi_bp/3000" ``  

#### Semi-supervised EP  
_Different configuration files are used for various labeled number scenarios._  
``python -m ep.main --json_path "./configFile/semi_ep/100" ``  
``python -m ep.main --json_path "./configFile/semi_ep/300" ``  
``python -m ep.main --json_path "./configFile/semi_ep/600" ``  
``python -m ep.main --json_path "./configFile/semi_ep/1000" ``  
``python -m ep.main --json_path "./configFile/semi_ep/3000" ``  

#### Supervised BP
_Train a perceptron with 2000 output neurons:_  
`python -m bp.main --json_path "./configFile/supervised_bp/1layer" `  
_Train a MLP with 2000 output neurons:_    
`python -m bp.main --json_path "./configFile/supervised_bp/2layer" ` 

#### Supervised EP  
_Train a perceptron with 10 output neurons:_    
`python -m ep.main --json_path './configFile/supervised_ep/1layer" `  
_Train a MLP with 10 output neurons:_    
`python -m ep.main --json_path './configFile/supervised_ep/2layer" `

## Hyperparameters research
We use the random research offer by [optuna](https://optuna.org/) to do the hyper-parameters optimization.

### Parameters explanation  
_We list here the parameters defined in *optuna_config.json* that differ from those in the config.json file._  

| Name        | Description                                                                       | Examples              |
|-------------|-----------------------------------------------------------------------------------|-----------------------|
| `mode`      | Defines how the neural average activity is calculated: in batch or sequentially.  | `batch`, `sequential` |
| `nudge_max` | Sets the upper limit for the number of winners that can be selected.              | `10`                  |



### Commands 
We take the two-layer network as an example. 
For alternative network structures, simply modify the network architecture in the _pre_config.file_.
#### BP hyperparameters  
_Supervised learning with a MLP:_  
`python -m bp.optuna_optim --json_path "./configFile/hyper_bp/supervised"`  
_Unsupervised learning with a two-layer network:_  
`python -m bp.optuna_optim --json_path "./configFile/hyper_bp/unsupervised"`  
_Unsupervised learning with a CNN:_  
`python -m bp.optuna_optim --json_path "./configFile/hyper_bp/unsupervised_cnn"`  
_Semi-supervised learning with a MLP (change the _train_label_number_ in pre_config file):_  
`python -m bp.optuna_optim --json_path "./configFile/hyper_bp/semi"`  
_Train the linear classifier on top of a trained unsupervised network:_  
`python -m bp.optuna_optim --json_path "./configFile/hyper_bp/classifier' --trained_path './simuResults/unsupervised_bp_2layer/model" `

#### EP hyperparameters  
_Supervised learning with a MLP:_  
`python -m ep.optuna_optim --json_path "./configFile/hyper_ep/supervised"`    
_Unsupervised learning with a two-layer network:_  
`python -m ep.optuna_optim --json_path "./configFile/hyper_ep/unsupervised"`    
_Semi-supervised learning with a MLP (change the _train_label_number_ in pre_config file):_  
`python -m ep.optuna_optim --json_path "./configFile/hyper_ep/semi"`   
_Train the linear classifier on top of a trained unsupervised network:_  
`python -m ep.optuna_optim --json_path "./configFile/hyper_ep/classifier' --trained_path './simuResults/unsupervised_ep_2layer/model" `

## Plotting the figures

We use the _reproduce_figures.ipynb_ to plot all the figures used in the article.

