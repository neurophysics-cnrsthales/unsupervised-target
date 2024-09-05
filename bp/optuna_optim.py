import argparse
import json
import copy as cp
from funcs.data import *
from funcs.tools import PathCreator
from .actions import *
import logging
import sys

# load the parameters in optuna_config
parser = argparse.ArgumentParser(description='Path of json file')
parser.add_argument(
    '--json_path',
    type=str,
    default=r'./bp',
    help='path of json configuration'
)
parser.add_argument(
    '--trained_path',
    type=str,
    # default=r'.',
    default=r'./pretrain_file',
    help='path of model_dict_state_file'
)

args = parser.parse_args()

# load the parameters in optuna_config
with open(args.json_path + '/optuna_config.json')as f:
    pre_config = json.load(f)


def jparams_create(config, trial):
    jparams = cp.deepcopy(config)

    if jparams["action"] == 'bp':
        # hyperparameters not used
        jparams["eta"] = 0.5
        jparams["gamma"] = 0.5
        jparams["nudge"] = 1
        jparams["pre_batchSize"] = 32
        # optimizing hyperparameters
        jparams["batchSize"] = 256
        # jparams["batchSize"] = trial.suggest_categorical("batchSize", [64, 128, 256, 512])
        lr = []
        if jparams["cnn"]:
            for i in range(len(jparams["fcLayers"]) + 2):
                lr.append(trial.suggest_float("lr_cnn" + str(i), 1e-5, 10, log=True))

        else:
            for i in range(len(jparams["fcLayers"]) - 1):
                lr_i = trial.suggest_float("lr" + str(i), 1e-5, 10, log=True)
                lr.append(lr_i)

        jparams["lr"] = lr.copy()

    elif jparams["action"] == 'unsupervised_bp':

        # hyperparameters not used
        jparams["pre_batchSize"] = 32

        # optimizing hyperparameters
        if jparams['mode'] == 'batch':
            jparams["batchSize"] = trial.suggest_categorical("batchSize", [8, 16, 32])
            jparams["eta"] = 0.5
        else:
            jparams["batchSize"] = 1,
            jparams["eta"] = trial.suggest_float("eta", 0.01, 1, log=True)

        jparams["gamma"] = trial.suggest_float("gamma", 0.1, 1, log=True)
        jparams["nudge"] = trial.suggest_int("nudge", 1, jparams["nudge_max"])

        lr = []
        if jparams["cnn"]:
            for i in range(len(jparams["fcLayers"]) + 2):
                lr.append(trial.suggest_float("lr_cnn" + str(i), 1e-9, 1e-3, log=True))

        else:
            for i in range(len(jparams["fcLayers"]) - 1):
                lr_i = trial.suggest_float("lr" + str(i), 1e-5, 10, log=True)
                lr.append(lr_i)

        jparams["lr"] = lr.copy()

    elif jparams["action"] == 'semi_supervised_bp':
        # hyperparameters not used
        jparams["eta"] = 0.5
        jparams["gamma"] = 0.2
        jparams["nudge"] = 1

        # optimizing hyperparameters
        jparams["pre_batchSize"] = trial.suggest_categorical("pre_batchSize", [16, 32, 64])
        jparams["batchSize"] = trial.suggest_categorical("batchSize", [128, 256])
        pre_lr = []
        for i in range(len(jparams["fcLayers"]) - 1):
            lr_i = trial.suggest_float("pre_lr" + str(i), 1e-7, 1e-1, log=True)
            pre_lr.append(lr_i)
        jparams["pre_lr"] = pre_lr.copy()

        lr = []
        for i in range(len(jparams["fcLayers"]) - 1):
            lr_i = trial.suggest_float("lr" + str(i), 1e-5, 1e-1, log=True)
            lr.append(lr_i)
        jparams["lr"] = lr.copy()

    elif jparams["action"] == 'train_class_layer':
        # hyperparameters not used
        jparams["batchSize"] = 256
        jparams["pre_batchSize"] = 32
        jparams["eta"] = 0.5
        jparams["gamma"] = 0.5
        jparams["lr"] = [0, 0]
        jparams["class_optimizer"] = 'Adam'

        # optimizing hyperparameters
        jparams["class_activation"] = trial.suggest_categorical("class_activation", ['sigmoid', 'x', 'hardsigm'])
        jparams["class_lr"] = [trial.suggest_float("class_lr", 1e-4, 10, log=True)]

    return jparams


def objective(trial, config):
    # design the hyperparameters to be optimized
    jparams = jparams_create(config, trial)

    (train_loader, test_loader, validation_loader,
    class_loader, layer_loader, supervised_loader, unsupervised_loader) = get_dataset(jparams, validation=True)

    # create the model
    # TODO considering the CNN model
    if jparams["cnn"]:
        net = CNN(jparams)
        net.prune_network(amount=jparams["cnn_prune"])
    else:
        net = MLP(jparams)

    # load the trained unsupervised network when we train classification layer
    if jparams["action"] == 'bp':
        print("Hyperparameters optimization for supervised BP")
        final_err = supervised_bp(net, jparams, train_loader, validation_loader, base_path=None, trial=trial)

    elif jparams["action"] == 'unsupervised_bp':
        print("Hyperparameters optimization for unsupervised BP")
        if jparams['cnn']:
            final_err = unsupervsed_bp_cnn(net, jparams, train_loader, validation_loader, layer_loader, trial=trial)
        else:
            final_err = unsupervised_bp(net, jparams, train_loader, class_loader, validation_loader, layer_loader,
                                    base_path=None, trial=trial)

    elif jparams["action"] == 'semi_supervised_bp':
        print("Hyperparameters optimization for semi-supervised BP")
        final_err = semi_supervised_bp(net, jparams, supervised_loader, unsupervised_loader, validation_loader,
                                       base_path=None, trial=trial)
    elif jparams["action"] == 'train_class_layer':
        print("Hyperparameters optimization for linear classifier on top of a BP trained network")
        trained_path = str(args.trained_path)+'/model_state_dict.pt'
        final_err = train_class_layer(net, jparams, layer_loader, validation_loader, trained_path=trained_path,
                                      trial=trial)
    else:
        raise ValueError(f"'{jparams['action']}' is not defined in hyper optimization!")

    df = study.trials_dataframe()
    df.to_csv(filePath)
    del jparams

    return final_err


if __name__ == '__main__':

    # define the dataframe
    optuna_path_creator = PathCreator('optuna-')
    BASE_PATH, name = optuna_path_creator.create_path()

    # save the optuna configuration
    with open(os.path.join(BASE_PATH, "optuna_config.json"), "w") as outfile:
        json.dump(pre_config, outfile)

    # create the filepath for saving the optuna trails
    filePath = os.path.join(BASE_PATH, "bp_test.csv")

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler(),
                                pruner=optuna.pruners.PercentilePruner(20, n_startup_trials=2, n_warmup_steps=3))

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study.optimize(lambda trial: objective(trial, pre_config), n_trials=200)
    trails = study.get_trials()
