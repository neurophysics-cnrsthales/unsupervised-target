import copy as CP
import argparse
import json
import logging
import sys
from funcs.data import *
from funcs.tools import PathCreator
from .actions import *

parser = argparse.ArgumentParser(description='Path of json file')
parser.add_argument(
    '--json_path',
    type=str,
    default=r'./ep',
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
with open(str(args.json_path) + '/optuna_config.json') as f:
    pre_config = json.load(f)

# define the activation function
if pre_config['activation_function'] == 'sigm':
    def rho(x):
        return 1 / (1 + torch.exp(-(4 * (x - 0.5))))

    def rhop(x):
        return 4 * torch.mul(rho(x), 1 - rho(x))

elif pre_config['activation_function'] == 'hardsigm':
    def rho(x):
        return x.clamp(min=0).clamp(max=1)

    def rhop(x):
        return (x >= 0) & (x <= 1)

elif pre_config['activation_function'] == 'half_hardsigm':
    def rho(x):
        return (1 + F.hardtanh(x - 1)) * 0.5

    def rhop(x):
        return ((x >= 0) & (x <= 2)) * 0.5

elif pre_config['activation_function'] == 'tanh':
    def rho(x):
        return torch.tanh(x)

    def rhop(x):
        return 1 - torch.tanh(x) ** 2


def jparams_create(config, trial):
    jparams = CP.deepcopy(config)

    if jparams["action"] == 'ep':

        # not used parameters
        jparams["eta"] = 0.5
        jparams["gamma"] = 0.5
        jparams["nudge"] = 1
        jparams["pre_batchSize"] = 32

        # optimizing hyperparameters
        jparams["batchSize"] = trial.suggest_categorical("batchSize", [64, 128, 256])
        jparams["beta"] = trial.suggest_float("beta", 0.05, 0.5)
        lr = []
        for i in range(len(jparams["fcLayers"]) - 1):
            lr_i = trial.suggest_float("lr" + str(i), 1e-6, 1e-1, log=True)
            lr.append(lr_i)
        jparams["lr"] = lr.copy()
        jparams["lr"].reverse()

    elif jparams["action"] == 'unsupervised_ep':

        # hyperparameters not used
        jparams["pre_batchSize"] = 32

        # optimizing hyperparameters
        jparams["beta"] = trial.suggest_float("beta", 0.05, 0.5)
        if jparams['mode'] == 'batch':
            jparams["batchSize"] = trial.suggest_categorical("batchSize", [16, 32, 64])
            jparams["eta"] = 0.5
        else:
            jparams["batchSize"] = 1,
            jparams["eta"] = trial.suggest_float("eta", 0.01, 1, log=True)

        # test parameters
        jparams["gamma"] = trial.suggest_float("gamma", 0.1, 1, log=True)
        jparams["nudge"] = trial.suggest_int("nudge", 1, jparams["nudge_max"])

        lr = []
        for i in range(len(jparams["fcLayers"]) - 1):
            lr_i = trial.suggest_float("lr" + str(i), 1e-6, 1, log=True)
            lr.append(lr_i)
        jparams["lr"] = lr.copy()
        jparams['lr'].reverse()

    elif jparams["action"] == 'semi_supervised_ep':
        # hyperparameters not used
        jparams["eta"] = 0.5
        jparams["gamma"] = 0.1
        jparams["beta"] = 0.45
        jparams["nudge"] = 1

        # optimizing hyperparameters
        jparams["pre_batchSize"] = trial.suggest_categorical("pre_batchSize", [16, 32, 64])
        jparams["batchSize"] = trial.suggest_categorical("batchSize", [128, 256])
        pre_lr = []
        for i in range(len(jparams["fcLayers"]) - 1):
            lr_i = trial.suggest_float("pre_lr" + str(i), 1e-7, 1e-1, log=True)
            pre_lr.append(lr_i)
        jparams["pre_lr"] = pre_lr.copy()
        jparams["pre_lr"].reverse()

        lr = []
        for i in range(len(jparams["fcLayers"]) - 1):
            lr_i = trial.suggest_float("lr" + str(i), 1e-6, 1e-2, log=True)
            # to verify whether we need to change the name of lr_i
            lr.append(lr_i)
        jparams["lr"] = lr.copy()
        jparams["lr"].reverse()

    elif jparams["action"] == 'train_class_layer':
        # non-used parameters
        jparams["pre_batchSize"] = 32
        jparams["batchSize"] = 256
        jparams["beta"] = 0.5
        jparams["eta"] = 0.5
        jparams["gamma"] = 0.5
        jparams["lr"] = [0, 0]
        jparams["nudge"] = 1
        # unchanged parameters
        jparams["class_optimizer"] = trial.suggest_categorical("class_optimizer", ['SGD', 'Adam'])

        # class
        jparams["class_smooth"] = trial.suggest_categorical("class_smooth", [True, False])
        jparams["class_activation"] = trial.suggest_categorical("class_activation", ['sigmoid', 'x', 'hardsigm'])
        jparams["class_lr"] = [trial.suggest_float("class_lr", 1e-4, 1, log=True)]

    return jparams


def objective(trial, config):
    # design the hyperparameters to be optimized
    jparams = jparams_create(config, trial)

    # create the dataset
    (train_loader, test_loader, validation_loader,
     class_loader, layer_loader, supervised_loader, unsupervised_loader) = get_dataset(jparams, validation=True)

    # reverse the layer
    jparams['fcLayers'].reverse()  # we put in the other side, output first, input last
    jparams['dropProb'].reverse()
    # create the model
    net = torch.jit.script(MlpEP(jparams, rho, rhop))

    if jparams["action"] == 'ep':
        print("Hyperparameters optimization for supervised EP")
        final_err = supervised_ep(net, jparams, train_loader, test_loader, trial=trial)
    elif jparams["action"] == 'unsupervised_ep':
        print("Hyperparameters optimization for unsupervised EP")
        final_err = unsupervised_ep(net, jparams, train_loader, class_loader, test_loader, layer_loader, trial=trial)
    elif jparams["action"] == 'semi_supervised_ep':
        print("Hyperparameters optimization for semi-supervised EP")
        final_err = semi_supervised_ep(net, jparams, supervised_loader, unsupervised_loader, test_loader, trial=trial)
    elif jparams["action"] == 'train_class_layer':
        print("Hyperparameters optimization for linear classifier on top of a EP trained network")
        trained_path = str(args.trained_path) + '/model_entire.pt'
        final_err = train_class_layer(net, jparams, layer_loader, test_loader, trained_path=trained_path, trial=trial)
    else:
        raise ValueError(f"'{jparams['action']}' is not defined in hyper optimization!")

    # record trials
    df = study.trials_dataframe()
    df.to_csv(filePath)
    del jparams

    return final_err


if __name__ == '__main__':

    # Cuda problem
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # define the dataframe
    optuna_path_creator = PathCreator('optuna-')
    BASE_PATH, name = optuna_path_creator.create_path()

    # save the optuna configuration
    with open(os.path.join(BASE_PATH, "optuna_config.json"), "w") as outfile:
        json.dump(pre_config, outfile)

    # create the filepath for saving the optuna trails
    filePath = os.path.join(BASE_PATH, "ep_test.csv")

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler(),
                                pruner=optuna.pruners.PercentilePruner(20, n_startup_trials=2, n_warmup_steps=3))

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study.optimize(lambda trial: objective(trial, pre_config), n_trials=200)

    trails = study.get_trials()
