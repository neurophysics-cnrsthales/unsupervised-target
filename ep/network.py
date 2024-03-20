# File defining the network and the oscillators composing the network
import torch.jit as jit
from torch.nn import functional as F
from funcs.activations import *
from typing import List, Optional


class MlpEP(jit.ScriptModule):

    def __init__(self, jparams, rho, rhop):

        super(MlpEP, self).__init__()

        self.T = jparams['T']
        self.Kmax = jparams['Kmax']
        self.dt = jparams['dt']
        self.beta = torch.tensor(jparams['beta'])
        self.clamped = jparams['clamped']
        self.batchSize = jparams['batchSize']
        self.fcLayers = jparams['fcLayers']
        self.error_estimate = jparams['error_estimate']
        self.gamma = jparams['gamma']
        self.rho = rho
        self.rhop = rhop
        if jparams['loss'] == 'Cross-entropy':
            self.softmax_output = True
        else:
            self.softmax_output = False

        # define the device
        if jparams['device'] >= 0 and torch.cuda.is_available():
            device = torch.device("cuda:" + str(jparams['device']))
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False

        self.device = device

        # We define the parameters to be trained

        W: List[torch.Tensor] = []
        for i in range(len(jparams['fcLayers']) - 1):
            w = torch.empty(jparams['fcLayers'][i + 1], jparams['fcLayers'][i], device=device, requires_grad=True)
            # bound = 1 / np.sqrt(jparams['fcLayers'][i + 1])
            nn.init.xavier_uniform_(w, gain=0.5)
            # nn.init.uniform_(w, a=-bound, b=bound)
            W.append(w)
        self.W = W

        # We define the list to save the bias
        bias: List[torch.Tensor] = []
        for i in range(len(jparams['fcLayers']) - 1):
            b = torch.empty(jparams['fcLayers'][i], device=device, requires_grad=True)
            # bound = 1 / np.sqrt(jparams['fcLayers'][i])
            # nn.init.uniform_(b, a=-bound, b=bound)

            bias.append(b)
        self.bias = bias

        self = self.to(device)


    @jit.script_method
    def stepper_softmax(self, s: List[torch.Tensor], p_distribut: Optional[List[torch.Tensor]],
                        target: Optional[torch.Tensor] = None, beta: Optional[float] = None):

        if len(s) < 3:
            raise ValueError("Input list 's' must have at least three elements for softmax-readout.")
        if p_distribut is not None:
            if len(p_distribut) != len(s):
                raise ValueError("p_distribut must have the same number of elements as s.")

        # Separate 'h' elements and 'y'
        h = s[1:]  # All but the first element are considered 'h'
        y = F.softmax(torch.mm(self.rho(h[0]), self.W[0]) + self.bias[0], dim=1)
        if p_distribut is not None:
            y = p_distribut[0] * y

        dhdt = [-h[0] + (self.rhop(h[0]) * (torch.mm(self.rho(h[1]), self.W[1]) + self.bias[1]))]
        if target is not None and beta is not None:
            dhdt[0] = dhdt[0] + beta * torch.mm((target - y), self.W[0].T)

        for layer in range(1, len(h) - 1):
            dhdt.append(-h[layer] + self.rhop(h[layer]) *
                        (torch.mm(self.rho(h[layer + 1]), self.W[layer + 1])
                         + self.bias[layer + 1] + torch.mm(self.rho(h[layer - 1]), self.W[layer].T)))
        # update h
        for (layer, dhdt_item) in enumerate(dhdt):
            if p_distribut is not None:
                h[layer] = p_distribut[layer + 1] * (h[layer] + self.dt * dhdt_item)
            else:
                h[layer] = h[layer] + self.dt * dhdt_item
            if self.clamped:
                h[layer] = h[layer].clamp(0, 1)

        return [y] + h

    @jit.script_method
    def stepper_c(self, s: List[torch.Tensor], p_distribut: Optional[List[torch.Tensor]],
                     target: Optional[torch.Tensor] = None,
                     beta: Optional[float] = None):
        """
        stepper function for energy-based dynamics of EP
        """
        if len(s) < 2:
            raise ValueError("Input list 's' must have at least two elements.")
        if p_distribut is not None:
            if len(p_distribut) != len(s):
                raise ValueError("p_distribut must have the same number of elements as s.")

        dsdt = [-s[0] + (self.rhop(s[0]) * (torch.mm(self.rho(s[1]), self.W[0]) + self.bias[0]))]

        if target is not None and beta is not None:
            dsdt[0] = dsdt[0] + beta * (target - s[0])

        for layer in range(1, len(s) - 1):  # start at the first hidden layer and then to the before last hidden layer
            dsdt.append(-s[layer] + self.rhop(s[layer]) * (
                    torch.mm(self.rho(s[layer + 1]), self.W[layer])
                    + self.bias[layer] + torch.mm(self.rho(s[layer - 1]), self.W[layer - 1].T)))

        for (layer, dsdt_item) in enumerate(dsdt):
            if p_distribut is not None:
                s[layer] = p_distribut[layer] * (s[layer] + self.dt * dsdt_item)
            else:
                s[layer] = s[layer] + self.dt * dsdt_item
            if self.clamped:
                s[layer] = s[layer].clamp(0, 1)

        return s

    @jit.script_method
    def forward(self, s: List[torch.Tensor], p_distribut: Optional[List[torch.Tensor]] = None,
                beta: Optional[float] = None, target: Optional[torch.Tensor] = None) -> List[torch.Tensor]:

        T, Kmax = self.T, self.Kmax

        with torch.no_grad():
            # continuous time EP
            if beta is None and target is None:
                # free phase
                if self.softmax_output:
                    for t in range(T):
                        s = self.stepper_softmax(s, p_distribut, target=target, beta=beta)
                else:
                    for t in range(T):
                        s = self.stepper_c(s, p_distribut, target=target, beta=beta)
            else:
                # nudged phase
                if self.softmax_output:
                    for t in range(Kmax):
                        s = self.stepper_softmax(s, p_distribut, target=target, beta=beta)
                else:
                    for t in range(Kmax):
                        s = self.stepper_c(s, p_distribut, target=target, beta=beta)

        return s

    # @jit.script_method
    def compute_gradients_ep(self, s: List[torch.Tensor],
                             seq: List[torch.Tensor],
                             target: Optional[torch.Tensor] = None):
        """
        Compute EQ gradient to update the synaptic weight -
        for classic EP! for continuous time dynamics and prototypical
        """
        batch_size = s[0].size(0)
        # learning rate should be the 1/beta of the BP learning rate
        # in this way the learning rate is corresponded with the sign of beta
        coef = 1 / (self.beta * batch_size)
        if self.error_estimate == 'symmetric':
            coef = coef * 0.5

        gradW, gradBias = [], []

        with torch.no_grad():
            if self.softmax_output:
                if self.error_estimate != 'symmetric':
                    gradW.append(-(1 / batch_size) * torch.mm(torch.transpose(self.rho(s[1]), 0, 1), (s[0] - target)))
                    gradBias.append(-(1 / batch_size) * (s[0] - target).sum(0))
                else:
                    gradW.append(
                        -(0.5 / batch_size) * (torch.mm(torch.transpose(self.rho(s[1]), 0, 1), (s[0] - target)) +
                                               torch.mm(torch.transpose(self.rho(seq[1]), 0, 1),
                                                        (seq[0] - target))))
                    gradBias.append(-(0.5 / batch_size) * (s[0] + seq[0] - 2 * target).sum(0))
            else:
                gradW.append(coef * (torch.mm(torch.transpose(self.rho(s[1]), 0, 1), self.rho(s[0]))
                                     - torch.mm(torch.transpose(self.rho(seq[1]), 0, 1),
                                                self.rho(seq[0]))))
                gradBias.append(coef * (self.rho(s[0]) - self.rho(seq[0])).sum(0))

            for layer in range(1, len(s) - 1):
                gradW.append(coef * (torch.mm(torch.transpose(self.rho(s[layer + 1]), 0, 1), self.rho(s[layer]))
                                     - torch.mm(torch.transpose(self.rho(seq[layer + 1]), 0, 1),
                                                self.rho(seq[layer]))))
                gradBias.append(coef * (self.rho(s[layer]) - self.rho(seq[layer])).sum(0))

        for i in range(len(self.W)):
            self.W[i].grad = -gradW[i]
            self.bias[i].grad = -gradBias[i]

    def init_state(self, data):
        """
        Init the state of the network
        State if a dict, each layer is state["S_layer"]
        Xdata is the last element of the dict
        """
        state = []
        size = data.size(0)
        for layer in range(len(self.fcLayers) - 1):
            state.append(torch.zeros(size, self.fcLayers[layer], requires_grad=False))

        state.append(data.float())

        return state


class Classifier(nn.Module):
    # one layer perceptron does not need to be trained by EP
    def __init__(self, jparams):
        super(Classifier, self).__init__()
        # construct the classifier layer
        self.classifier = torch.nn.Sequential(nn.Dropout(p=float(jparams['class_dropProb'])),
                                              nn.Linear(jparams['fcLayers'][0], jparams['n_class']),
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
