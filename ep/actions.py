import os
import optuna

from tqdm import tqdm
from funcs.tools import update_dataframe
from funcs.tools import init_ep_dataframe as init_dataframe
from .train_test import *

DATAFRAME, PretrainFrame, SEMIFRAME, class_dataframe = None, None, None, None


def define_optimizer(net, lr, optimizer_type, momentum=0, dampening=0):
    net_params = []

    for i in range(len(net.W)):
        net_params += [{'params': [net.W[i]], 'lr': lr[i]}]
        net_params += [{'params': [net.bias[i]], 'lr': lr[i]}]

    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(net_params, momentum=momentum, dampening=dampening)
    elif optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(net_params)
    else:
        raise ValueError("{} optimizer_type of optimizer is not defined ".format(optimizer_type))

    return net_params, optimizer


def define_optimizer_classlayer(net, lr, optimizer_type, momentum=0, dampening=0):
    # Define optimizer parameters directly from the network's named parameters
    parameters = []
    lr_layer = None
    for idx, (name, param) in enumerate(net.named_parameters()):
        if param.requires_grad:  # Check if the parameter requires gradient computation
            # Update learning rate based on the index
            lr_layer = lr[int(idx / 2)] if idx % 2 == 0 else lr_layer  # Maintain previous lr_layer value if idx is odd
            # Append layer parameters with the corresponding learning rate
            parameters.append({'params': [param], 'lr': lr_layer})

    # Construct the optimizer
    # TODO changer optimizer to ADAM
    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(parameters, momentum=momentum, dampening=dampening)
    elif optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(parameters)
    else:
        raise ValueError('This optimizer optimizer_type has not been defined!')
    return parameters, optimizer


def supervised_ep(net, jparams, train_loader, test_loader, BASE_PATH=None, trial=None):
    global DATAFRAME
    # define optimizer
    params, optimizer = define_optimizer(net, jparams['lr'], jparams['optimizer'])

    train_error_list, test_error_list = [], []

    if BASE_PATH is not None:
        DATAFRAME = init_dataframe(BASE_PATH, method='supervised')

    test_error_epoch = None
    for epoch in tqdm(range(jparams['epochs'])):
        # train
        train_error_epoch = train_supervised_ep(net, jparams, train_loader, optimizer, epoch)
        # test
        test_error_epoch = test_supervised_ep(net, jparams['n_class'], test_loader)

        # add optuna pruning process
        if trial is not None:
            trial.report(test_error_epoch, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if BASE_PATH is not None:
            train_error_list.append(train_error_epoch.item())
            test_error_list.append(test_error_epoch.item())

            DATAFRAME = update_dataframe(BASE_PATH, DATAFRAME, train_error_list, test_error_list)

            # save the entire model
            with open(os.path.join(BASE_PATH, 'model_entire.pt'), 'wb') as f:
                torch.jit.save(net, f)

    if trial is not None:
        return test_error_epoch


def unsupervised_ep(net, jparams, train_loader, class_loader, test_loader, layer_loader, BASE_PATH=None, trial=None):
    # define optimizer
    global DATAFRAME
    params, optimizer = define_optimizer(net, jparams['lr'], jparams['optimizer'])

    # define scheduler
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-3,
                                                  total_iters=jparams['epochs'])

    # define the record lists
    test_error_list_av, test_error_list_max = [], []
    error_av_epoch = None

    if BASE_PATH is not None:
        DATAFRAME = init_dataframe(BASE_PATH, method='unsupervised')

    for eph in tqdm(range(jparams['epochs'])):
        # train
        train_unsupervised_ep(net, jparams, train_loader, optimizer, eph)
        # class association
        response = classify(net, jparams, class_loader)
        # test
        error_av_epoch, error_max_epoch = test_unsupervised_ep(net, jparams, test_loader, response)
        # scheduler
        scheduler.step()

        # add optuna pruning process
        if trial is not None:
            trial.report(error_av_epoch, eph)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if BASE_PATH is not None:
            test_error_list_av.append(error_av_epoch.item())
            test_error_list_max.append(error_max_epoch.item())

            DATAFRAME = update_dataframe(BASE_PATH, DATAFRAME, test_error_list_av, test_error_list_max)

            # save the entire model
            with open(os.path.join(BASE_PATH, 'model_entire.pt'), 'wb') as f:
                torch.jit.save(net, f)

    if trial is not None:
        return error_av_epoch

    if jparams['epochs'] == 0 and BASE_PATH is not None:
        response = classify(net, jparams, class_loader)
        error_av_epoch, error_max_epoch = test_unsupervised_ep(net, jparams, test_loader, response)
        test_error_list_av.append(error_av_epoch.item())
        test_error_list_max.append(error_max_epoch.item())
        DATAFRAME = update_dataframe(BASE_PATH, DATAFRAME, test_error_list_av, test_error_list_max)

    # we create the layer for classfication
    train_class_layer(net, jparams, layer_loader, test_loader, trained_path=None, BASE_PATH=BASE_PATH, trial=None)


def pre_supervised_ep(net, jparams, supervised_loader, test_loader, BASE_PATH=None, trial=None):
    global PretrainFrame

    # define pre_train optimizer
    pretrain_params, pretrain_optimizer = define_optimizer(net, jparams['pre_lr'], jparams['optimizer'])
    # define pre_train scheduler
    pretrain_scheduler = torch.optim.lr_scheduler.ExponentialLR(pretrain_optimizer, 0.96)
    # define record lists
    pretrain_error_list, pretest_error_list = [], []
    pretest_error_epoch = None

    if BASE_PATH is not None:
        PretrainFrame = init_dataframe(BASE_PATH, method='supervised', dataframe_to_init='pre_supervised.csv')

    for eph in tqdm(range(jparams['pre_epochs'])):
        # train
        pretrain_error_epoch = train_supervised_ep(net, jparams, supervised_loader, pretrain_optimizer, eph)
        # test
        pretest_error_epoch = test_supervised_ep(net, jparams['n_class'], test_loader)
        # scheduler
        pretrain_scheduler.step()

        if trial is not None:
            trial.report(pretest_error_epoch, eph)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if BASE_PATH is not None:
            pretrain_error_list.append(pretrain_error_epoch.item())
            pretest_error_list.append(pretest_error_epoch.item())
            PretrainFrame = update_dataframe(BASE_PATH, PretrainFrame, pretrain_error_list, pretest_error_list,
                                             'pre_supervised.csv')
            # save the entire model
            with open(os.path.join(BASE_PATH, 'model_pre_supervised_entire.pt'), 'wb') as f:
                torch.jit.save(net, f)

    return pretest_error_epoch


def semi_supervised_ep(net, jparams, supervised_loader, unsupervised_loader, test_loader, BASE_PATH=None, trial=None):
    global SEMIFRAME

    # Pretrain with labeled data
    initial_pretrain_err = pre_supervised_ep(net, jparams, supervised_loader, test_loader, BASE_PATH=BASE_PATH)

    # define the supervised and unsupervised optimizer
    unsupervised_params, unsupervised_optimizer = define_optimizer(net, jparams['lr'], jparams['optimizer'])
    supervised_params, supervised_optimizer = define_optimizer(net, jparams['lr'], jparams['optimizer'])

    # define the supervised and unsupervised scheduler
    unsupervised_scheduler = torch.optim.lr_scheduler.LinearLR(unsupervised_optimizer,
                                                               start_factor=0.001,
                                                               end_factor=0.1,
                                                               total_iters=jparams['epochs'])
    supervised_scheduler = torch.optim.lr_scheduler.LinearLR(supervised_optimizer,
                                                             start_factor=0.2,
                                                             end_factor=0.03,
                                                             total_iters=jparams['epochs'])

    # define record lists
    supervised_test_error_list, entire_test_error_list = [], []
    supervised_test_epoch = None

    if BASE_PATH is not None:
        SEMIFRAME = init_dataframe(BASE_PATH, method='semi-supervised', dataframe_to_init='semi-supervised.csv')

    for epoch in tqdm(range(jparams['epochs'])):
        # unsupervised train
        train_unsupervised_ep(net, jparams, unsupervised_loader, unsupervised_optimizer, epoch)
        # test
        entire_test_epoch = test_supervised_ep(net, jparams['n_class'], test_loader)
        # unsupervised scheduler
        unsupervised_scheduler.step()

        # supervised train
        train_supervised_ep(net, jparams, supervised_loader, supervised_optimizer, epoch)
        # test
        supervised_test_epoch = test_supervised_ep(net, jparams['n_class'], test_loader)
        # supervised scheduler
        supervised_scheduler.step()

        if trial is not None:
            trial.report(supervised_test_epoch, epoch)
            if entire_test_epoch > initial_pretrain_err:
                raise optuna.TrialPruned()
            if trial.should_prune():
                raise optuna.TrialPruned()

        if BASE_PATH is not None:
            supervised_test_error_list.append(supervised_test_epoch.item())
            entire_test_error_list.append(entire_test_epoch.item())
            SEMIFRAME = update_dataframe(BASE_PATH, SEMIFRAME, entire_test_error_list, supervised_test_error_list,
                                         'semi-supervised.csv')
            with open(os.path.join(BASE_PATH, 'model_semi_entire.pt'), 'wb') as f:
                torch.jit.save(net, f)
    if trial is not None:
        return supervised_test_epoch


def train_class_layer(net, jparams, layer_loader, test_loader, trained_path=None, BASE_PATH=None, trial=None):
    # load the pre-trianed network
    global class_dataframe
    if trained_path is not None:
        with open(trained_path, 'rb') as f:
            loaded_net = torch.jit.load(f, map_location=net.device)
            net.W = loaded_net.W.copy()
            net.bias = loaded_net.bias.copy()

    # create the classification layer
    class_net = Classifier(jparams)

    # define optimizer
    class_params, class_optimizer = define_optimizer_classlayer(class_net, jparams['class_lr'],
                                                                jparams['class_optimizer'])

    # define scheduler
    class_scheduler = torch.optim.lr_scheduler.LinearLR(class_optimizer, start_factor=1,
                                                        end_factor=0.9, total_iters=jparams['class_epoch'])
    # Create record lists
    class_train_error_list, final_test_error_list, final_loss_error_list = [], [], []
    final_test_error_epoch = None

    if BASE_PATH is not None:
        # create dataframe for classification layer
        class_dataframe = init_dataframe(BASE_PATH, method='classification_layer',
                                         dataframe_to_init='classification_layer.csv')

    for epoch in tqdm(range(jparams['class_epoch'])):
        # train
        class_train_error_epoch = classify_network(net, class_net, layer_loader, class_optimizer,
                                                   jparams['class_smooth'])
        # test
        final_test_error_epoch, final_loss_epoch = test_unsupervised_ep_layer(net, class_net, jparams, test_loader)
        # scheduler
        class_scheduler.step()

        if trial is not None:
            trial.report(final_test_error_epoch, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if BASE_PATH is not None:
            class_train_error_list.append(class_train_error_epoch.item())
            final_test_error_list.append(final_test_error_epoch.item())
            final_loss_error_list.append(final_loss_epoch.item())
            class_dataframe = update_dataframe(BASE_PATH, class_dataframe, class_train_error_list,
                                               final_test_error_list,
                                               filename='classification_layer.csv', loss=final_loss_error_list)
            # save the trained class_net
            torch.save(class_net.state_dict(), os.path.join(BASE_PATH, 'class_model_state_dict.pt'))

    if trial is not None:
        return final_test_error_epoch
