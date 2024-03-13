# coding: utf-8
from funcs.tools import smooth_labels, drop_all_layers, define_unsupervised_target, direct_association, One2one
from funcs.data import generate_n_targets_label
from .network import *


def classify(net, jparams, class_loader, k_select=None):
    # todo do the sum for each batch to save the memory
    net.eval()

    class_record = torch.zeros((jparams['n_class'], jparams['fcLayers'][0]), device=net.device)
    labels_record = torch.zeros((jparams['n_class'], 1), device=net.device)
    for batch_idx, (data, targets) in enumerate(class_loader):

        if net.cuda:
            targets = targets.to(net.device)

        # forward propagation
        output = inference_ep(net, data)

        for i in range(jparams['n_class']):
            indice = (targets == i).nonzero(as_tuple=True)[0]
            labels_record[i] += len(indice)
            class_record[i, :] += torch.sum(output[indice, :], dim=0)

    # take the maximum activation as associated class
    response, k_select_neuron = direct_association(class_record, labels_record, k_select)

    if k_select is None:
        return response
    else:
        return response, k_select_neuron


def classify_network(net, class_net, layer_loader, optimizer, class_smooth=None):
    net.eval()
    class_net.train()

    # define the loss of classification layer
    criterion = torch.nn.CrossEntropyLoss()

    # create the list for training errors
    correct_train = torch.zeros(1, device=net.device).squeeze()
    total_train = torch.zeros(1, device=net.device).squeeze()

    for batch_idx, (data, targets) in enumerate(layer_loader):
        optimizer.zero_grad()

        if net.cuda:
            targets = targets.to(net.device)

        x = inference_ep(net, data)
        output = class_net.forward(x)
        if class_smooth:
            targets = smooth_labels(targets.to(torch.float32), 0.3, 1)

        # calculate the loss
        loss = criterion(output, targets)
        # backpropagation
        loss.backward()
        optimizer.step()

        # calculate the training errors
        prediction = torch.argmax(output, dim=1)
        correct_train += (prediction == torch.argmax(targets, dim=1)).sum().float()
        total_train += targets.size(dim=0)

    # calculate the train error
    train_error = 1 - correct_train / total_train
    return train_error


def test_unsupervised_ep_layer(net, class_net, jparams, test_loader):
    net.eval()
    class_net.eval()

    # create the list for testing errors
    correct_test = torch.zeros(1, device=net.device).squeeze()
    total_test = torch.zeros(1, device=net.device).squeeze()
    loss_test = torch.zeros(1, device=net.device).squeeze()
    total_batch = torch.zeros(1, device=net.device).squeeze()

    for batch_idx, (data, targets) in enumerate(test_loader):
        total_batch += 1
        # record the total test
        total_test += targets.size()[0]
        targets = targets.type(torch.LongTensor)
        if net.cuda:
            targets = targets.to(net.device)

        # forward propagation in classification layer
        x = inference_ep(net, data)
        output = class_net.forward(x)
        # calculate the loss
        if jparams['class_activation'] == 'softmax':
            loss = F.cross_entropy(output, targets)
        else:
            loss = F.mse_loss(output, F.one_hot(targets, num_classes=jparams['n_class']))

        loss_test += loss.item()

        # calculate the training errors
        prediction = torch.argmax(output, dim=1)
        correct_test += (prediction == targets).sum().float()

    # calculate the test error
    test_error = 1 - correct_test / total_test
    loss_test = loss_test / total_batch

    return test_error, loss_test


# def Mlp_Centropy_train_cycle(net, jparams, train_loader, optimizer, Xth=None):
#
#     total_train = torch.zeros(1, device=net.device).squeeze()
#     correct_train = torch.zeros(1, device=net.device).squeeze()
#
#     # Stochastic mode
#     if jparams['batchSize'] == 1:
#         Y_p = torch.zeros(jparams['fcLayers'][0], device=net.device)
#
#     for batch_idx, (data, targets) in enumerate(train_loader):
#         optimizer.zero_grad()
#         # random signed beta: better approximate the gradient
#         net.beta = torch.sign(torch.randn(1)) * jparams['beta']
#         # init the hidden layers
#         h, y = net.initHidden(data)
#
#         if jparams['dropout']:
#             p_distribut, y_distribut = drop_all_layers(h, p=jparams['dropProb'], y=y)
#             del (h)
#             h, y = net.initHidden(data, drop_visible=p_distribut[-1])
#             p_distribut = p_distribut[:-1]
#         else:
#             p_distribut, y_distribut = None, None
#
#         if net.cuda:
#             targets = targets.to(net.device)
#             net.beta = net.beta.to(net.device)
#             h = [item.to(net.device) for item in h]  # no need to put data on the GPU as data is included in s!
#             if jparams['dropout']:
#                 p_distribut = [item.to(net.device) for item in p_distribut]
#                 y_distribut = y_distribut.to(net.device)
#
#         if len(h) <= 1:
#             raise ValueError(
#                 "Cross-entropy loss should be used for more than 1 layer structure" "but got {} layer".format(
#                     len(h)))
#
#         # free phase
#         # TODO change all the rho_y to pre_y
#         h, y, rho_y = net.forward_softmax(h, p_distribut, y_distribut)
#         heq = h.copy()
#         yeq = y.clone()
#
#         if Xth is not None:
#             del(targets)
#             #targets, maxindex = define_unsupervised_target(y, jparams['nudge'], net.device, Xth)
#             targets, maxindex = define_unsupervised_target(rho_y, jparams['nudge'], net.device, Xth)
#             # label smoothing
#             if jparams['smooth']:
#                 targets = smooth_labels(targets.float(), 0.3, jparams['nudge'])
#
#         else:
#             # label smoothing
#             if jparams['smooth']:
#                 targets = smooth_labels(targets.float(), 0.3, 1)
#
#         if jparams['dropout']:
#             targets = y_distribut*targets
#
#         if jparams['error_estimate'] == 'one-sided':
#             # nudging phase
#             h, y, rho_y = net.forward_softmax(h, p_distribut, y_distribut, target=targets, beta=net.beta)
#             # compute the gradients
#             net.computeGradientEP_softmax(h, heq, y, targets)
#             optimizer.step()
#
#         elif jparams['error_estimate'] == 'symmetric':
#             # + beta
#             h, y, rho_y = net.forward_softmax(h, p_distribut, y_distribut, target=targets, beta=net.beta)
#             hplus = h.copy()
#             yplus = y.clone()
#             # -beta
#             h = heq.copy()
#             h, y, rho_y = net.forward_softmax(h, p_distribut, y_distribut, target=targets, beta=-net.beta)
#             hmoins = h.copy()
#             ymoins = y.clone()
#             # update and track the weights of the network
#             net.computeGradientEP_softmax(hplus, hmoins, yplus, targets, ybeta=ymoins)
#             optimizer.step()
#         if Xth is None:
#             # calculate the training error
#             prediction = torch.argmax(yeq.detach(), dim=1)
#             correct_train += (prediction == torch.argmax(targets, dim=1)).sum().float()
#             total_train += targets.size(dim=0)
#         else:
#             if jparams['dropout']:
#                 target_activity = jparams['nudge'] / (jparams['fcLayers'][0] * (
#                         1 - jparams['dropProb'][0]))  # dropout influences the target activity
#             else:
#                 target_activity = jparams['nudge'] / jparams['fcLayers'][0]
#
#             if jparams['batchSize'] == 1:
#                 Y_p = (1 - jparams['eta']) * Y_p + jparams['eta'] * targets[0]
#                 Xth += net.gamma * (Y_p - target_activity)
#             else:
#                 Xth += net.gamma * (torch.mean(targets, axis=0) - target_activity)
#     if Xth is None:
#         return 1 - correct_train / total_train
#     else:
#         return Xth
#


def inference_ep(net, data):
    # no dropout
    s = net.init_state(data)
    if net.cuda:
        s = [item.to(net.device) for item in s]
    # free phase
    s = net.forward(s)
    # we note the last layer as s_output
    output = s[0].clone().detach()

    return output


def nudge_weight_update(net, s, targets, p_distribut, error_estimate, loss):
    seq = s.copy()
    if error_estimate == 'one-sided':
        s = net.forward(s, p_distribut, target=targets, beta=net.beta)
    elif error_estimate == 'symmetric':
        # -beta
        s = net.forward(s, p_distribut, target=targets, beta=-net.beta)
        s_moins = s.copy()
        # + beta
        s = seq.copy()
        s = net.forward(s, p_distribut, target=targets, beta=net.beta)
        del seq
        seq = s_moins
    else:
        raise ValueError("f'{jparams['error_estimate']}' is not defined!")

    if loss == 'MSE':
        net.compute_gradients_ep(s, seq)
    elif loss == 'Cross-entropy':
        net.compute_gradients_ep(s, seq, target=targets)
    else:
        raise ValueError("f'{jparams['loss']}' is not defined!")


def train_supervised_ep(net, jparams, train_loader, optimizer, epoch):
    net.train()
    net.epoch = epoch + 1

    total_train = torch.zeros(1, device=net.device).squeeze()
    correct_train = torch.zeros(1, device=net.device).squeeze()

    for batch_idx, (data, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        # random signed beta: better approximate the gradient
        net.beta = torch.sign(torch.randn(1)) * jparams['beta']
        # init the hidden layers
        s = net.init_state(data)

        if jparams['dropout']:
            p_distribut = drop_all_layers(s, p=jparams['dropProb'])
            s[-1] = s[-1] * p_distribut[-1]
        else:
            p_distribut = None

        if net.cuda:
            targets = targets.to(net.device)
            net.beta = net.beta.to(net.device)
            s = [item.to(net.device) for item in s]
            if jparams['dropout']:
                p_distribut = [item.to(net.device) for item in p_distribut]

        # free phase
        s = net.forward(s, p_distribut)
        num_per_class = jparams['fcLayers'][0] // jparams['n_class']
        prediction = torch.argmax(s[0].detach(), dim=1) // num_per_class

        # consider the situation with multi-neuron representing one class
        if jparams['fcLayers'][0] > jparams['n_class']:
            multi_targets = generate_n_targets_label(torch.argmax(targets, dim=1).tolist(), num_per_class,
                                                     jparams['fcLayers'][0])
            if net.cuda:
                multi_targets = multi_targets.to(net.device)
            # nudging phase
            nudge_weight_update(net, s, multi_targets, p_distribut, jparams['error_estimate'], jparams['loss'])
        else:
            if jparams['smooth']:
                targets = smooth_labels(targets.float(), 0.3, 1)

            if jparams['dropout']:
                targets = p_distribut[0] * targets

            # nudging phase
            nudge_weight_update(net, s, targets, p_distribut, jparams['error_estimate'], jparams['loss'])

        optimizer.step()

        correct_train += (prediction == torch.argmax(targets, dim=1)).sum().float()
        total_train += targets.size(dim=0)

    return 1 - correct_train / total_train


def train_unsupervised_ep(net, jparams, train_loader, optimizer, epoch):
    """
    Function to train the network for 1 epoch
    """

    net.train()
    net.epoch = epoch + 1

    homeo = torch.zeros(jparams['fcLayers'][0], device=net.device)

    Y_p = None
    # Stochastic mode
    if jparams['batchSize'] == 1:
        Y_p = torch.zeros(jparams['fcLayers'][0], device=net.device)

    for batch_idx, (data, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        del targets

        # random signed beta: better approximate the gradient
        net.beta = torch.sign(torch.randn(1)) * jparams['beta']

        # init layers
        s = net.init_state(data)
        if jparams['dropout']:
            p_distribut = drop_all_layers(s, p=jparams['dropProb'])
            s[-1] = s[-1] * p_distribut[-1]
        else:
            p_distribut = None

        if net.cuda:
            net.beta = net.beta.to(net.device)
            s = [item.to(net.device) for item in s]
            if jparams['dropout']:
                p_distribut = [item.to(net.device) for item in p_distribut]

        # free phase
        s = net.forward(s, p_distribut)
        # calculate the output
        if jparams['loss'] == 'Cross-entropy':
            output = torch.mm(net.rho(s[1]), net.W[0]) + net.bias[0]
        else:
            output = s[0].clone()

        targets, maxindex = define_unsupervised_target(output.detach(), jparams['nudge'], net.device, homeo)

        # label smoothing
        if jparams['smooth']:
            targets = smooth_labels(targets.float(), 0.3, jparams['nudge'])
        # dropout on the targets
        if jparams['dropout']:
            targets = p_distribut[0] * targets

        # calculate the weight
        nudge_weight_update(net, s, targets, p_distribut, jparams['error_estimate'], jparams['loss'])
        optimizer.step()

        if jparams['dropout']:
            target_activity = jparams['nudge'] / (jparams['fcLayers'][0] * (
                    1 - jparams['dropProb'][0]))  # dropout influences the target activity
        else:
            target_activity = jparams['nudge'] / jparams['fcLayers'][0]

        if jparams['batchSize'] == 1:
            Y_p = (1 - jparams['eta']) * Y_p + jparams['eta'] * targets[0]
            homeo += net.gamma * (Y_p - target_activity)
        else:
            homeo += net.gamma * (torch.mean(targets, dim=0) - target_activity)

    return homeo


def test_unsupervised_ep(net, jparams, test_loader, response):
    """
        Function to test the network
        """
    net.eval()

    # record total test number
    total_test = torch.zeros(1, device=net.device).squeeze()

    # record unsupervised test error
    correct_av_test = torch.zeros(1, device=net.device).squeeze()
    correct_max_test = torch.zeros(1, device=net.device).squeeze()

    one2one = One2one(jparams['n_class'])

    for batch_idx, (data, targets) in enumerate(test_loader):
        # record the total test
        total_test += targets.size()[0]

        if net.cuda:
            targets = targets.to(net.device)

        output = inference_ep(net, data)

        '''average value'''
        predict_av = one2one.average_predict(output, response)
        correct_av_test += (predict_av == targets).sum().float()

        '''maximum value'''
        # remove the non response neurons
        predict_max = one2one.max_predict(output, response)
        correct_max_test += (predict_max == targets).sum().float()

    # calculate the test error
    test_error_av = 1 - correct_av_test / total_test
    test_error_max = 1 - correct_max_test / total_test

    return test_error_av, test_error_max


def test_supervised_ep(net, class_num, test_loader):
    """
    Function to test the network
    """
    net.eval()

    # record total test number
    total_test = torch.zeros(1, device=net.device).squeeze()

    # record supervised test error
    corrects_supervised = torch.zeros(1, device=net.device).squeeze()

    for batch_idx, (data, targets) in enumerate(test_loader):
        # record the total test
        total_test += targets.size()[0]

        if net.cuda:
            targets = targets.to(net.device)

        output = inference_ep(net, data)

        num_per_class = output.size(1) // class_num
        prediction = torch.argmax(output.detach(), dim=1) // num_per_class
        corrects_supervised += (prediction == targets).sum().float()

    test_error = 1 - corrects_supervised / total_test
    return test_error
