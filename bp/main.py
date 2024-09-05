import sys
import argparse
import json
from pathlib import Path
from .actions import *
from funcs.data import *
from funcs.tools import PathCreator

sys.path.append(str(Path(__file__).resolve().parent.parent))

parser = argparse.ArgumentParser(description='Path of json file')
parser.add_argument(
    '--json_path',
    type=str,
    default=r'./bp/',
    help='path of json configuration'
)
parser.add_argument(
    '--trained_path',
    type=str,
    default=None,
    help='path of json configuration'
)

args = parser.parse_args()

with open(args.json_path + '/config.json') as f:
    jparams = json.load(f)


# define the two batch sizes
batch_size = jparams['batchSize']
batch_size_test = jparams['test_batchSize']

# if jparams["dataset"] == "cifar10":
#     train_loader, test_loader, class_loader, layer_loader = returnFastCIFAR10(jparams, validation=False)
# else:
# get data loader
(train_loader, test_loader, validation_loader,
class_loader, layer_loader, supervised_loader, unsupervised_loader) = get_dataset(jparams)

if __name__ == '__main__':

    # create the network
    if jparams['cnn']:
        net = CNN(jparams)
        net.prune_network(amount=jparams["cnn_prune"])
    else:
        net = MLP(jparams)

    # Cuda problem
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # save hyper-parameters as json file
    bp_path_creator = PathCreator('bp-')
    BASE_PATH, name = bp_path_creator.create_path()

    with open(os.path.join(BASE_PATH, "config.json"), "w") as outfile:
        json.dump(jparams, outfile)

    if jparams['action'] == 'bp':
        print("Training the supervised bp network")
        supervised_bp(net, jparams, train_loader, test_loader, base_path=BASE_PATH)

    elif jparams['action'] == 'unsupervised_bp':
        print("Training the unsupervised bp network")
        if jparams['cnn']:
            unsupervsed_bp_cnn(net, jparams, train_loader, test_loader, layer_loader, base_path=BASE_PATH)
        else:
            unsupervised_bp(net, jparams, train_loader, class_loader, test_loader, layer_loader, base_path=BASE_PATH)

    elif jparams['action'] == 'semi_supervised_bp':
        print("Training the semi_supervised bp network")
        semi_supervised_bp(net, jparams, supervised_loader, unsupervised_loader, test_loader, base_path=BASE_PATH)

    elif jparams['action'] == 'pretrain_bp':
        print("Training the supervised bp network with little labeled data")
        pre_supervised_bp(net, jparams, supervised_loader, test_loader, base_path=BASE_PATH)

    elif jparams['action'] == 'train_class_layer':
        print("We train only the linear classifer")
        if args.trained_path is None:
            trained_path = None
        else:
            trained_path = str(args.trained_path) + '/model_state_dict.pt'
        train_class_layer(net, jparams, layer_loader, test_loader, trained_path=trained_path, base_path=BASE_PATH)

    else:
        raise ValueError(f"f'{jparams['action']}' action is not defined!")
