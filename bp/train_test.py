# coding: utf-8
import pandas as pd
import os
from torch.nn import functional as F
from funcs.data import generate_n_targets_label
from funcs.tools import define_loss, define_unsupervised_target, drop_output, smooth_labels, direct_association, One2one
from .network import *


def classify(net, jparams, class_loader, k_select=None):
    # todo do the sum for each batch to save the memory
    net.eval()

    class_record = torch.zeros((jparams['n_class'], jparams['fcLayers'][-1]), device=net.device)
    labels_record = torch.zeros((jparams['n_class'], 1), device=net.device)
    for batch_idx, (data, targets) in enumerate(class_loader):

        if net.cuda:
            data = data.to(net.device)
            targets = targets.to(net.device)

        # forward propagation
        output = net(data.to(torch.float32))

        for i in range(jparams['n_class']):
            indice = (targets == i).nonzero(as_tuple=True)[0]
            labels_record[i] += len(indice)
            class_record[i, :] += torch.sum(output[indice, :], dim=0)

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
            data = data.to(net.device)
            targets = targets.to(net.device)

        # net forward
        with torch.no_grad():
            x = net(data.to(torch.float32))  # without the dropout

        # class_net forward
        output = class_net(x)

        # calculate the prediction
        prediction = torch.argmax(output, dim=1)

        # class label smooth
        if class_smooth:
            targets = smooth_labels(targets.to(torch.float32), 0.2, 1)

        # print("targets dimension is:", targets.dim())
        if targets.dim() == 1:
            loss = criterion(output, targets.to(torch.long))
            correct_train += (prediction == targets).sum().float()
        else:
            loss = criterion(output, targets.to(torch.float32))
            correct_train += (prediction == torch.argmax(targets, dim=1)).sum().float()
        # Ensure the targets are integers representing class indices
        # TODO to support different type of targets, setting the if situation
        # loss = criterion(output, targets.to(torch.long))  # Convert targets to LongTensor (integers)
        # print("target is :", targets)

        # backpropagation
        loss.backward()
        optimizer.step()
        #
        # # calculate the training errors
        # prediction = torch.argmax(output, dim=1)
        # correct_train += (prediction == torch.argmax(targets, dim=1)).sum().float()
        # correct_train += (prediction == targets).sum().float()
        total_train += targets.size(dim=0)

    # calculate the train error
    train_error = 1 - correct_train / total_train
    return train_error


def train_unsupervised(net, jparams, train_loader, epoch, optimizer):
    net.train()
    net.epoch = epoch + 1

    # construct the loss function
    criterion = define_loss(jparams['loss'])

    # initiate the Homeostasis term
    homeo = torch.zeros(jparams['fcLayers'][-1], device=net.device)
    total_batch = torch.zeros(1, device=net.device).squeeze()
    total_loss = torch.zeros(1, device=net.device).squeeze()

    # Stochastic mode
    if jparams['batchSize'] == 1:
        y_p = torch.zeros(jparams['fcLayers'][-1], device=net.device)

    for batch_idx, (data, _) in enumerate(train_loader):
        if net.cuda:
            data = data.to(net.device)
            # target = target.to(net.device)
        optimizer.zero_grad()

        # forward propagation
        output = net(data.to(torch.float32))
        # generate output mask
        output_mask = drop_output(output, p=jparams['dropProb'][-1]).to(net.device)
        output = output_mask * output

        # create the unsupervised target
        unsupervised_targets, n_maxindex = define_unsupervised_target(output, jparams['nudge'], net.device,
                                                                      Homeo=homeo)
        # label smoothing
        if jparams['smooth']:
            unsupervised_targets = smooth_labels(unsupervised_targets, 0.3, jparams['nudge'])
        unsupervised_targets = unsupervised_targets * output_mask
        target_activity = (1 - jparams['dropProb'][-1]) * jparams['nudge'] / jparams['fcLayers'][-1]

        if jparams['batchSize'] == 1:
            y_p = (1 - jparams['eta']) * y_p + jparams['eta'] * unsupervised_targets[0]
            homeo += jparams['gamma'] * (y_p - target_activity)
        else:
            homeo += jparams['gamma'] * (torch.mean(unsupervised_targets, dim=0) - target_activity)

        # calculate the loss on the gpu
        loss = criterion(output, unsupervised_targets.to(torch.float32))
        loss.backward()
        total_batch += 1
        total_loss += loss

        optimizer.step()

    return total_loss / total_batch


def train_bp(net, jparams, train_loader, epoch, optimizer):
    net.train()
    net.epoch = epoch + 1

    # construct the loss function
    criterion = define_loss(jparams['loss'])

    # create the list for training errors and testing errors
    correct_train = torch.zeros(1, device=net.device).squeeze()
    total_train = torch.zeros(1, device=net.device).squeeze()

    # for Homeostasis, initialize the moving average and the target activity
    # TODO consider the situation the batch number is not divisible

    for batch_idx, (data, targets) in enumerate(train_loader):

        if net.cuda:
            data = data.to(net.device)
            targets = targets.to(net.device)  # target here is the extern targets

        optimizer.zero_grad()

        # forward propagation
        output = net(data.to(torch.float32))

        # label smoothing
        if jparams['smooth']:
            targets = smooth_labels(targets.to(torch.float32), 0.2, 1)

        # transform targets
        if jparams['fcLayers'][-1] > jparams['n_class']:
            number_per_class = jparams['fcLayers'][-1] // jparams['n_class']
            multi_targets = generate_n_targets_label(torch.argmax(targets, dim=1).tolist(), number_per_class,
                                                     jparams['fcLayers'][-1])
            if net.cuda:
                multi_targets = multi_targets.to(net.device)

            loss = criterion(output, multi_targets.to(torch.float32))
        else:
            loss = criterion(output, targets.to(torch.float32))

        loss.backward()
        # print('the backward loss is:', net.W[0].weight.grad)
        optimizer.step()
        # count correct times for supervised BP

        # training error
        number_per_class = output.size(1) // jparams['n_class']
        prediction = torch.argmax(output, dim=1) // number_per_class

        correct_train += (prediction == torch.argmax(targets, dim=1)).sum().float()
        total_train += targets.size(dim=0)

    # # update the lr after at the end of each epoch
    # scheduler.step()

    # calculate the train error
    train_error = 1 - correct_train / total_train
    return train_error


def test_unsupervised(net, jparams, test_loader, response, output_record_path=None):
    net.eval()

    # record the number of test examples
    total_test = torch.zeros(1, device=net.device).squeeze()
    # records of errors for unsupervised BP
    correct_av_test = torch.zeros(1, device=net.device).squeeze()
    correct_max_test = torch.zeros(1, device=net.device).squeeze()

    # records of the output values for the test dataset
    if output_record_path is not None:
        records = []

    one2one = One2one(jparams['n_class'])

    for batch_idx, (data, targets) in enumerate(test_loader):

        if net.cuda:
            data = data.to(net.device)
            targets = targets.to(net.device)

        if len(targets.size()) > 1:  # for the training error
            targets = torch.argmax(targets, 1)

        output = net(data.to(torch.float32))

        if output_record_path is not None:
            records.append({'img': output.cpu().tolist(), 'target': targets.cpu().tolist()})

        # calculate the total testing times and record the testing labels
        total_test += targets.size()[0]

        # calculate the one2one_av
        predict_av = one2one.average_predict(output, response)
        correct_av_test += (predict_av == targets).sum().float()

        # calculate the one2one_max
        predict_max = one2one.max_predict(output, response)
        correct_max_test += (predict_max == targets).sum().float()

    # save the output values:
    if output_record_path is not None:
        df = pd.DataFrame(records)
        df.to_pickle(os.path.join(str(output_record_path), 'output_records.pkl'))
        del df

    # calculate the test error
    test_error_av = 1 - correct_av_test / total_test
    test_error_max = 1 - correct_max_test / total_test

    return test_error_av, test_error_max


def test_bp(net, test_loader, class_num, output_record_path=None):
    net.eval()

    # record the total test time
    total_test = torch.zeros(1, device=net.device).squeeze()

    # records of accuracy for supervised BP
    correct_test = torch.zeros(1, device=net.device).squeeze()

    # records of the output values for the test dataset
    if output_record_path is not None:
        records = []

    for batch_idx, (data, targets) in enumerate(test_loader):

        if net.cuda:
            data = data.to(net.device)
            targets = targets.to(net.device)

        output = net(data.to(torch.float32))

        if output_record_path is not None:
            records.append({'img': output.cpu().tolist(), 'target': targets.cpu().tolist()})

        # calculate the total testing times and record the testing labels
        total_test += targets.size()[0]

        # calculate the accuracy
        number_per_class = output.size(1) // class_num
        prediction = torch.argmax(output, dim=1) // number_per_class

        correct_test += (prediction == targets).sum().float()

    # save the output values:
    if output_record_path is not None:
        df = pd.DataFrame(records)
        df.to_pickle(os.path.join(str(output_record_path), 'output_records.pkl'))
        del df

    test_error = 1 - correct_test / total_test
    return test_error


def test_unsupervised_layer(net, class_net, jparams, test_loader):
    net.eval()
    class_net.eval()

    # create the list for testing errors
    correct_test = torch.zeros(1, device=net.device).squeeze()
    total_test = torch.zeros(1, device=net.device).squeeze()
    loss_test = torch.zeros(1, device=net.device).squeeze()
    total_batch = torch.zeros(1, device=net.device).squeeze()

    for batch_idx, (data, targets) in enumerate(test_loader):

        total_batch += 1
        targets = targets.type(torch.LongTensor)
        if net.cuda:
            data = data.to(net.device)
            targets = targets.to(net.device)

        # record the total test
        total_test += targets.size()[0]

        # net forward
        x = net(data.to(torch.float32))
        # class_net forward
        output = class_net(x)
        # calculate the loss
        if jparams['class_activation'] == 'softmax' or jparams['class_activation'] == 'x':
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
