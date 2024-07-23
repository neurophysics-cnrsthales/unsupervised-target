import optuna
from funcs.tools import init_bp_dataframe as init_dataframe
from funcs.tools import update_dataframe, CustomStepLR
from .train_test import *

from tqdm import tqdm

DATAFRAME, PretrainFrame, SEMIFRAME, class_dataframe = None, None, None, None


def define_optimizer(net, lr, optim_type, momentum=0, dampening=0):
    parameters = []

    for idx, (name, param) in enumerate(net.named_parameters()):
        if param.requires_grad:
            lr_layer = lr[idx // 2]  # Integer division by 2 to use the same lr for consecutive parameters (weight/bias)
            parameters.append({'params': param, 'lr': lr_layer})

    # construct the optimizer

    # TODO changer optimizer to ADAM
    if optim_type == 'SGD':
        optimizer = torch.optim.SGD(parameters, momentum=momentum, dampening=dampening)
    elif optim_type == 'Adam':
        optimizer = torch.optim.Adam(parameters)
    else:
        raise ValueError('This optimizer_type of optimizer is not defined!')
    return parameters, optimizer


# def update_momentum(optimizer, epoch, start_factor, end_factor):
#     if epoch < 500:
#         factor = (epoch/500)*end_factor + (1-epoch/500)*start_factor
#     else:
#         factor = end_factor
#
#     optimizer.momentum = factor
#     optimizer.dampening = factor

# todo use pre-defined scheduler for each training process
# def defineScheduler(optimizer, optimizer_type, decay_factor, decay_epoch, exponential_factor):
#     # linear
#     if optimizer_type == 'linear':
#         scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1,
#                                                            end_factor=decay_factor,
#                                                            total_iters=decay_epoch)
#     # exponential
#     elif optimizer_type == 'exponential':
#         scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, exponential_factor)
#     # step
#     elif optimizer_type == 'step':
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_epoch, gamma=decay_factor)
#     # combine cosine
#     elif optimizer_type == 'cosine':
#         scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=decay_factor,
#                                                          total_iters=decay_epoch)
#         scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_epoch)
#         scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])
#
#     return scheduler


def pre_supervised_bp(net, jparams, supervised_loader, test_loader, base_path=None, trial=None):
    # define the pre-supervised training optimizer
    global PretrainFrame
    pretrain_params, pretrain_optimizer = define_optimizer(net, jparams['pre_lr'], jparams['optimizer'])

    # define the pre-supervised scheduler
    pretrain_scheduler = torch.optim.lr_scheduler.ExponentialLR(pretrain_optimizer, 0.98)

    # list to save error
    pretrain_error_list, pretest_error_list = [], []
    pretest_error_epoch = None

    # save the initial network
    if base_path is not None:
        # init Dataframe
        PretrainFrame = init_dataframe(base_path, method='bp', dataframe_to_init='pre_supervised.csv')

    # training process
    for epoch in tqdm(range(jparams['pre_epochs'])):
        # train
        pretrain_error_epoch = train_bp(net, jparams, supervised_loader, epoch, pretrain_optimizer)
        # test
        pretest_error_epoch = test_bp(net, test_loader, jparams['n_class'])
        pretrain_scheduler.step()

        if trial is not None:
            trial.report(pretest_error_epoch, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if base_path is not None:
            pretrain_error_list.append(pretrain_error_epoch.item())
            pretest_error_list.append(pretest_error_epoch.item())
            # write the error in csv
            PretrainFrame = update_dataframe(base_path, PretrainFrame,
                                             pretrain_error_list, pretest_error_list, 'pre_supervised.csv')
            # save the entire model
            torch.save(net.state_dict(), os.path.join(base_path, 'model_pre_supervised_state_dict.pt'))

    return pretest_error_epoch


def supervised_bp(net, jparams, train_loader, test_loader, base_path=None, trial=None):
    global DATAFRAME
    # define the optimizer
    params, optimizer = define_optimizer(net, jparams['lr'], jparams['optimizer'])

    # define the scheduler
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1,
                                                  end_factor=5e-4, total_iters=jparams['epochs'])

    train_error, test_error = [], []
    test_error_epoch = None

    if base_path is not None:
        DATAFRAME = init_dataframe(base_path, method='bp')
        print(DATAFRAME)

    for epoch in tqdm(range(jparams['epochs'])):
        train_error_epoch = train_bp(net, jparams, train_loader, epoch, optimizer)
        test_error_epoch = test_bp(net, test_loader, jparams['n_class'])
        scheduler.step()

        if trial is not None:
            trial.report(test_error_epoch, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if base_path is not None:
            train_error.append(train_error_epoch.item())
            test_error.append(test_error_epoch.item())
            DATAFRAME = update_dataframe(base_path, DATAFRAME, train_error, test_error)
            torch.save(net.state_dict(), os.path.join(base_path, 'model_state_dict.pt'))

    if trial is not None:
        return test_error_epoch


def unsupervised_bp(net, jparams, train_loader, class_loader, test_loader, layer_loader, base_path=None, trial=None):
    global DATAFRAME

    # define optimizer
    params, optimizer = define_optimizer(net, jparams['lr'], jparams['optimizer'])

    # define scheduler
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1,
                                                  end_factor=5e-4, total_iters=jparams['epochs'])

    test_error_av, test_error_max = [], []
    test_error_av_epoch = None

    if base_path is not None:
        DATAFRAME = init_dataframe(base_path, method='unsupervised_bp')
        print(DATAFRAME)

    for epoch in tqdm(range(jparams['epochs'])):

        train_unsupervised(net, jparams, train_loader, epoch, optimizer)

        # direct association
        response = classify(net, jparams, class_loader)
        # testing process
        test_error_av_epoch, test_error_max_epoch = test_unsupervised(net, jparams, test_loader, response=response)
        scheduler.step()

        if trial is not None:
            trial.report(test_error_av_epoch, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if base_path is not None:
            test_error_av.append(test_error_av_epoch.item())
            test_error_max.append(test_error_max_epoch.item())
            DATAFRAME = update_dataframe(base_path, DATAFRAME, test_error_av, test_error_max)

            # at each epoch, we update the model parameters
            torch.save(net.state_dict(), os.path.join(base_path, 'model_state_dict.pt'))

    if trial is not None:
        return test_error_av_epoch

    if jparams['epochs'] == 0 and base_path is not None:
        # final direct association
        response = classify(net, jparams, class_loader)
        # testing process
        test_error_av_epoch, test_error_max_epoch = test_unsupervised(net, jparams, test_loader, response=response)
        test_error_av.append(test_error_av_epoch.item())
        test_error_max.append(test_error_max_epoch.item())
        DATAFRAME = update_dataframe(base_path, DATAFRAME, test_error_av, test_error_max)

    # train linear classifer
    train_class_layer(net, jparams, layer_loader, test_loader, base_path=base_path)


def unsupervsed_bp_cnn(net, jparams, train_loader, test_loader, layer_loader, base_path=None, trial=None):
    # define optimizer
    _, optimizer = define_optimizer(net, jparams['lr'], jparams['optimizer'])

    # define scheduler
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=1e-2, total_iters=100)

    # create the classification layer
    class_net = Classifier(jparams)

    # define optimizer
    _, class_optimizer = define_optimizer(class_net, jparams['class_lr'], jparams['class_optimizer'])

    # define scheduler
    class_scheduler = CustomStepLR(class_optimizer, jparams['class_epoch'])

    if base_path is not None:
        # create dataframe for classification layer
        class_dataframe = init_dataframe(base_path, method='classification_layer',
                                         dataframe_to_init='classification_layer.csv')
        class_train_error_list = []
        final_test_error_list = []
        final_loss_error_list = []

    for epoch in tqdm(range(jparams['epochs'])):
        # train unsupervised network
        train_unsupervised(net, jparams, train_loader, epoch, optimizer)
        scheduler.step()

        # train classifier
        class_train_error_epoch = classify_network(net, class_net, layer_loader, class_optimizer,
                                                   jparams['class_smooth'])
        # test
        final_test_error_epoch, final_loss_epoch = test_unsupervised_layer(net, class_net, jparams, test_loader)
        # scheduler
        class_scheduler.step()

        if trial is not None:
            trial.report(final_test_error_epoch, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if base_path is not None:
            class_train_error_list.append(class_train_error_epoch.item())
            final_test_error_list.append(final_test_error_epoch.item())
            final_loss_error_list.append(final_loss_epoch.item())
            class_dataframe = update_dataframe(base_path, class_dataframe, class_train_error_list,
                                               final_test_error_list,
                                               filename='classification_layer.csv', loss=final_loss_error_list)
            # Save the trained classifier model
            torch.save(class_net.state_dict(), os.path.join(base_path, 'class_model_state_dict.pt'))

    if trial is not None:
        return final_test_error_epoch

    # train linear classifer
    train_class_layer(net, jparams, layer_loader, test_loader, base_path=base_path)


def train_class_layer(net, jparams, layer_loader, test_loader, trained_path=None, base_path=None, trial=None):
    """
    Train the added linear classifier. Take the output of unsupervised network as input data and real labels as targets.
    This function is used exclusively in unsupervised Training.
    """

    global class_dataframe

    # Load the pre-trained network
    if trained_path is not None:
        net.load_state_dict(torch.load(trained_path))

    # Create the classification layer
    class_net = Classifier(jparams)
    # Define optimizer
    # TODO to see whether change the classifier scheduler

    class_params, class_optimizer = define_optimizer(class_net, jparams['class_lr'], jparams['class_optimizer'])
    # Define scheduler
    if jparams['cnn']:
        class_scheduler = CustomStepLR(class_optimizer, jparams['class_epoch'])
    else:
        class_scheduler = torch.optim.lr_scheduler.ExponentialLR(class_optimizer, 0.9)
    # Define the record lists
    class_train_error_list, final_test_error_list, final_loss_error_list = [], [], []
    # Define the return error
    final_test_error_epoch = None

    if base_path is not None:
        # create dataframe for classification layer
        class_dataframe = init_dataframe(base_path, method='classification_layer',
                                         dataframe_to_init='classification_layer.csv')

    for epoch in tqdm(range(jparams['class_epoch'])):
        # Train
        class_train_error_epoch = classify_network(net, class_net, layer_loader, class_optimizer, jparams['class_smooth'])
        # Test
        final_test_error_epoch, final_loss_epoch = test_unsupervised_layer(net, class_net, jparams, test_loader)
        # Scheduler
        class_scheduler.step()

        if trial is not None:
            trial.report(final_test_error_epoch, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if base_path is not None:
            class_train_error_list.append(class_train_error_epoch.item())
            final_test_error_list.append(final_test_error_epoch.item())
            final_loss_error_list.append(final_loss_epoch.item())
            class_dataframe = update_dataframe(base_path, class_dataframe, class_train_error_list,
                                               final_test_error_list,
                                               filename='classification_layer.csv', loss=final_loss_error_list)
            # Save the trained classifier model
            torch.save(class_net.state_dict(), os.path.join(base_path, 'class_model_state_dict.pt'))

    if trial is not None:
        return final_test_error_epoch


def semi_supervised_bp(net, jparams, supervised_loader, unsupervised_loader, test_loader, base_path=None, trial=None):
    global SEMIFRAME

    # train in the supervised way
    initial_pretrain_err = pre_supervised_bp(net, jparams, supervised_loader,
                                             test_loader, base_path=base_path, trial=None)

    # define the supervised and unsupervised optimizer
    unsupervised_params, unsupervised_optimizer = define_optimizer(net, jparams['lr'], jparams['optimizer'])
    supervised_params, supervised_optimizer = define_optimizer(net, jparams['lr'], jparams['optimizer'])

    # define the supervised and unsupervised scheduler
    unsupervised_scheduler = torch.optim.lr_scheduler.LinearLR(unsupervised_optimizer,
                                                               start_factor=0.001,
                                                               end_factor=0.18, total_iters=jparams['epochs'])
    supervised_scheduler = torch.optim.lr_scheduler.LinearLR(supervised_optimizer,
                                                             start_factor=0.72, end_factor=0.05,
                                                             total_iters=jparams['epochs'] - 50)
    # define the list for saving training error rates
    supervised_test_error_list, entire_test_error_list = [], []
    # define the return variable
    supervised_test_epoch = None

    if base_path is not None:
        # init the semi-supervised frame
        SEMIFRAME = init_dataframe(base_path, method='semi_supervised_bp', dataframe_to_init='semi_supervised.csv')

    for epoch in tqdm(range(jparams['epochs'])):
        # unsupervised training
        train_unsupervised(net, jparams, unsupervised_loader, epoch, unsupervised_optimizer)
        entire_test_epoch = test_bp(net, test_loader, jparams['n_class'])
        unsupervised_scheduler.step()

        # supervised reminder
        train_bp(net, jparams, supervised_loader, epoch, supervised_optimizer)
        supervised_test_epoch = test_bp(net, test_loader, jparams['n_class'])
        supervised_scheduler.step()

        if trial is not None:
            trial.report(supervised_test_epoch, epoch)
            if entire_test_epoch > initial_pretrain_err:
                raise optuna.TrialPruned()
            if trial.should_prune():
                raise optuna.TrialPruned()

        if base_path is not None:
            entire_test_error_list.append(entire_test_epoch.item())
            supervised_test_error_list.append(supervised_test_epoch.item())
            SEMIFRAME = update_dataframe(base_path, SEMIFRAME, entire_test_error_list, supervised_test_error_list,
                                         'semi_supervised.csv')
            # save the entire model
            torch.save(net.state_dict(), os.path.join(base_path, 'model_semi_state_dict.pt'))

    if trial is not None:
        return supervised_test_epoch
