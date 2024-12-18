import argparse  # this can be removed
import json
import sys
from pathlib import Path

from .actions import *
from funcs.data import *
from funcs.tools import PathCreator

sys.path.append(str(Path(__file__).resolve().parent.parent))

parser = argparse.ArgumentParser(description='Path of json file')
parser.add_argument(
    '--json_path',
    type=str,
    default=r'./ep',
    help='path of json configuration'
)


args = parser.parse_args()

with open(args.json_path + '/config.json') as f:
    jparams = json.load(f)


# define the two batch sizes
batch_size = jparams['batchSize']
batch_size_test = jparams['test_batchSize']

# get data loader
(train_loader, test_loader, validation_loader,
 class_loader, layer_loader, supervised_loader, unsupervised_loader) = get_dataset(jparams)

# define the activation function
if jparams['activation_function'] == 'sigm':
    def rho(x):
        return 1 / (1 + torch.exp(-(4 * (x - 0.5))))

    def rhop(x):
        return 4 * torch.mul(rho(x), 1 - rho(x))

elif jparams['activation_function'] == 'hardsigm':
    def rho(x):
        return x.clamp(min=0).clamp(max=1)

    def rhop(x):
        return (x >= 0) & (x <= 1)

elif jparams['activation_function'] == 'half_hardsigm':
    def rho(x):
        return (1 + F.hardtanh(x - 1)) * 0.5

    def rhop(x):
        return ((x >= 0) & (x <= 2)) * 0.5

elif jparams['activation_function'] == 'tanh':
    def rho(x):
        return torch.tanh(x)

    def rhop(x):
        return 1 - torch.tanh(x) ** 2

elif jparams['activation_function'] == 'relu':
    def rho(x):
        return x.clamp(min=0)

    def rhop(x):
        return x >= 0

else:
    raise ValueError(f"{jparams['activation_function']} activation function is not defined!")

if __name__ == '__main__':

    jparams['fcLayers'].reverse()  # we put in the other side, output first, input last
    jparams['lr'].reverse()
    jparams['pre_lr'].reverse()
    jparams['dropProb'].reverse()

    ep_path_creator = PathCreator('ep-')
    BASE_PATH, name = ep_path_creator.create_path()

    # save hyper-parameters as json file
    with open(os.path.join(BASE_PATH, "config.json"), "w") as outfile:
        json.dump(jparams, outfile)

    # Cuda problem
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # we create the network and define the  parameters
    net = torch.jit.script(MlpEP(jparams, rho, rhop))

    if jparams['action'] == 'ep':
        print("Training the model with supervised ep")
        supervised_ep(net, jparams, train_loader, test_loader, BASE_PATH=BASE_PATH)

    elif jparams['action'] == 'unsupervised_ep':
        print("Training the model with unsupervised ep")
        unsupervised_ep(net, jparams, train_loader, class_loader, test_loader, layer_loader, BASE_PATH=BASE_PATH)

    elif jparams['action'] == 'semi_supervised_ep':
        print("Training the model with semi-supervised learning")
        semi_supervised_ep(net, jparams, supervised_loader, unsupervised_loader, test_loader, BASE_PATH=BASE_PATH)

    elif jparams['action'] == 'pretrain_ep':
        print("Training the model with little dataset with supervised ep")
        pre_supervised_ep(net, jparams, supervised_loader, test_loader, BASE_PATH=BASE_PATH)

    else:
        raise ValueError(f"f'{jparams['action']}' action is not defined!")

