from funcs.activations import *


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
