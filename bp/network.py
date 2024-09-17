from funcs.activations import *
import torch.nn.utils.prune as prune
import numpy as np


class MLP(nn.Module):
    """
    Fully connected layers used for supervised/unsupervised/semi-supervised training.
    """
    def __init__(self, jparams):
        super(MLP, self).__init__()

        self.batchSize = jparams['batchSize']
        self.eta = jparams['eta']
        self.output_num = jparams['fcLayers'][-1]
        self.fcLayers = jparams['fcLayers']

        self.W = nn.ModuleList(None)

        # Put model on GPU is available and asked
        if jparams['device'] >= 0 and torch.cuda.is_available():
            device = torch.device("cuda:" + str(jparams['device']))
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False

        self.device = device

        # Construct fully connected networks
        self.fcnet = torch.nn.Sequential()

        for i in range(len(jparams['fcLayers'])-1):
            self.fcnet.add_module("rho_" + str(i), func_dict[jparams['activation_function'][i]]())
            self.fcnet.add_module("drop_"+str(i), nn.Dropout(p=float(jparams['dropProb'][i])))
            w = nn.Linear(jparams['fcLayers'][i], jparams['fcLayers'][i + 1], bias=True)
            nn.init.xavier_uniform_(w.weight, gain=0.5)
            nn.init.zeros_(w.bias)
            self.fcnet.add_module("fc_"+str(i), w)
        self.fcnet.add_module("rho_"+str(len(jparams['fcLayers'])-1), func_dict[jparams['activation_function'][-1]]())

        self = self.to(device)

    def forward(self, x):
        return self.fcnet(x)


class CNN(nn.Module):
    """
    Convolutional neural networks used for supervised/unsupervised training.
    """
    def __init__(self, jparams):
        super(CNN, self).__init__()
        self.conv_number = 0
        if jparams['dataset'] == 'mnist' or jparams['dataset'] == 'fashionMnist':
            input_size = 28
            input_channel = 1
        elif jparams['dataset'] == 'cifar10' or jparams['dataset'] == 'svhn':
            input_size = 32
            input_channel = 3
        else:
            raise ValueError("The convolutional network now is only designed for mnist/fashionmnist/svhn dataset")

        self.Conv = nn.ModuleList(None)  # TODO tobe removed?

        # define the CNN structure
        C_list = jparams["C_list"]

        # C_list = [input_channel, 96, 384]
        Pad = [2, 1]
        convF = [5, 3]
        Fpool = [4, 4]
        Spool = [2, 2]
        Ppool = [1, 1]
        conv_number = int(len(C_list) - 1)

        self.conv_number = conv_number
        self.fcLayers = jparams['fcLayers']
        # calculate the output size of each layer
        size_convpool_list = [input_size]

        # the size after the pooling layer
        for i in range(conv_number):
            size_convpool_list.append(int(np.floor(
                (size_convpool_list[i] - convF[i] + 1 + 2 * Pad[i] - Fpool[i] + 2 * Ppool[i]) / Spool[i] + 1)))

        print("output size of layers are:", size_convpool_list)

        self.convNet = nn.Sequential()
        # construct the convolutional layer
        self.convNet.add_module("rho_0", func_dict[jparams['activation_function'][0]]())

        for i in range(conv_number):
            self.convNet.add_module("conv_" + str(i), nn.Conv2d(in_channels=C_list[i], out_channels=C_list[i + 1],
                                                                kernel_size=convF[i], padding=Pad[i]))
            self.convNet.add_module("rho_" + str(i + 1), func_dict[jparams['activation_function'][i + 1]]())
            # add Avg/Max pool
            if jparams["pool_type"] == 'max':
                self.convNet.add_module("Pool_" + str(i), nn.MaxPool2d(kernel_size=Fpool[i], stride=Spool[i], padding=Ppool[i]))
            elif jparams["pool_type"] == 'av':
                self.convNet.add_module("Pool_" + str(i), nn.AvgPool2d(kernel_size=Fpool[i], stride=Spool[i], padding=Ppool[i]))
            else:
                raise ValueError(f"f'{jparams['pool_type']}' Pooling type is not defined!")

        conv_output = C_list[-1] * size_convpool_list[-1] ** 2
        # define the fully connected layer
        self.fcLayers.insert(0, conv_output)

        self.fcNet = nn.Sequential()
        # dropout for the flattened cnn output
        self.fcNet.add_module("drop_0", nn.Dropout(p=float(jparams['dropProb'][0])))

        for i in range(len(jparams['fcLayers']) - 1):
            self.fcNet.add_module("fc_" + str(i),
                                  nn.Linear(jparams['fcLayers'][i], jparams['fcLayers'][i + 1], bias=True))
            self.fcNet.add_module("rho_" + str(i + conv_number + 1),
                                  func_dict[jparams['activation_function'][i + conv_number + 1]]())

            if i < len(jparams['fcLayers']) - 2:
                self.fcNet.add_module("drop_" + str(i + 1), nn.Dropout(p=float(jparams['dropProb'][i + 1])))

        # self.batchNorm = batchNorm
        if jparams['device'] >= 0 and torch.cuda.is_available():
            device = torch.device("cuda:" + str(jparams['device']))
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False
        self = self.to(device)
        self.device = device

    def prune_network(self, amount):
        # Prune each convolutional layer in the model
        for name, module in self.convNet.named_children():
            if name.startswith("conv_"):
                prune.random_unstructured(module, name="weight", amount=amount)

    def forward(self, x):
        x = self.convNet(x)
        x = x.view(x.size(0), -1)
        if len(self.fcLayers) > 1:
            x = self.fcNet(x)
        return x


class Classifier(nn.Module):
    """
    Linear classifier added at the end of unsupervised network.
    Input neurons number = unsupervised network output neurons number.
    """

    def __init__(self, jparams):
        super(Classifier, self).__init__()
        # construct the classifier layer
        self.classifier = torch.nn.Sequential(nn.Dropout(p=float(jparams['class_dropProb'])),
                                              nn.Linear(jparams['fcLayers'][-1], jparams['n_class']),
                                              func_dict[jparams['class_activation']]())

        if jparams['device'] >= 0 and torch.cuda.is_available():
            device = torch.device("cuda:" + str(jparams['device']))
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False

        self.device = device
        self = self.to(device)

    def forward(self, x):
        return self.classifier(x)
