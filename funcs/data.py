import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class ReshapeTransformTarget:
    def __init__(self, number_classes):
        self.number_classes = number_classes

    def __call__(self, target):
        if torch.is_tensor(target):
            target = target.unsqueeze(0).unsqueeze(1)
        else:
            target = torch.tensor(target).unsqueeze(0).unsqueeze(1)
        target_onehot = torch.zeros((1, self.number_classes))

        return target_onehot.scatter_(1, target.long(), 1).squeeze(0)


class MyDataset(Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
        self.data = images
        self.targets = labels
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        data = self.data[i, :].numpy()
        target = self.targets[i]
        # data, label = self.data[item].numpy(), self.targets[item]
        data = Image.fromarray(data)

        if self.transform:
            data = self.transform(data)

        if self.target_transform:
            target = self.target_transform(target)

        if self.targets is not None:
            return data, target
        else:
            return data

    def __len__(self):
        return len(self.data)


class SplitClass(Dataset):
    def __init__(self, x, y, split_ratio, seed, transform=None, target_transform=None):

        class_set_data, rest_data, \
            class_set_targets, rest_targets = train_test_split(x, y, train_size=split_ratio, random_state=seed,
                                                               stratify=y)

        del (rest_data, rest_targets)

        self.data = class_set_data
        self.transform = transform
        self.targets = class_set_targets
        self.target_transform = target_transform

    def __getitem__(self, item):
        img, label = self.data[item].numpy(), self.targets[item].numpy()
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.targets)


def generate_n_targets_label(targets, number_per_class, output_neurons):
    multi_targets = list(map(lambda x: np.asarray(range(number_per_class * x, number_per_class * (x + 1))), targets))
    mlb = MultiLabelBinarizer(classes=range(output_neurons))
    n_targets = mlb.fit_transform(multi_targets)

    return torch.from_numpy(n_targets)


def semi_supervised_dataset(train_set, targets, output_neurons, n_class, labeled_number, transform, seed=1):
    fraction = labeled_number / len(targets)
    # we split the dataset for supervised training and unsupervised training
    X_super, X_unsuper, Y_super, Y_unsuper = train_test_split(train_set, targets, test_size=1 - fraction,
                                                              train_size=fraction, random_state=seed,
                                                              stratify=targets)
    number_per_class = int(output_neurons / n_class)

    # we define the target of supervised learning considering the number of output neurons
    if number_per_class > 1:
        N_Y_super = generate_n_targets_label(Y_super, number_per_class, output_neurons)
    else:
        N_Y_super = torch.nn.functional.one_hot(Y_super, num_classes=-1)

    # we load the target
    dataset_super = MyDataset(X_super, N_Y_super, transform=transform, target_transform=None)
    dataset_unsuper = MyDataset(X_unsuper, Y_unsuper, transform=transform, target_transform=None)  # no one-hot coding
    # dataset_super = torch.utils.data.TensorDataset(transform(X_super), N_Y_super)
    # dataset_unsuper = torch.utils.data.TensorDataset(transform(X_unsuper), Y_unsuper) # no one-hot coding

    return dataset_super, dataset_unsuper


def return_mnist(jparams, validation=False, fashion=False):
    # Define the Transform
    transforms_type = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      # torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                                      ReshapeTransform((-1,))])

    # Train set
    if fashion:
        train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True,
                                                      transform=transforms_type,
                                                      target_transform=ReshapeTransformTarget(10))
    else:
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                               transform=transforms_type,
                                               target_transform=ReshapeTransformTarget(10))

    # Validation set
    if validation:
        (X_train, X_validation,
         Y_train, Y_validation) = train_test_split(train_set.data, train_set.targets,
                                                   test_size=0.1, random_state=34, stratify=train_set.targets)

        train_set = MyDataset(X_train, Y_train, transform=transforms_type, target_transform=ReshapeTransformTarget(10))
        validation_set = MyDataset(X_validation, Y_validation, transform=transforms_type, target_transform=None)
    else:
        validation_set = None

    # Class set and Layer set
    if jparams['class_label_percentage'] == 1:
        class_set = MyDataset(train_set.data, train_set.targets, transform=transforms_type, target_transform=None)
        layer_set = train_set
    else:
        class_set = SplitClass(train_set.data, train_set.targets, jparams['class_label_percentage'], seed=34,
                               transform=transforms_type)
        layer_set = SplitClass(train_set.data, train_set.targets, jparams['class_label_percentage'], seed=34,
                               transform=transforms_type,
                               target_transform=ReshapeTransformTarget(10))

    # Supervised set and Unsupervised set
    if jparams['semi_seed'] < 0:
        semi_seed = None
    else:
        semi_seed = jparams['semi_seed']

    supervised_dataset, unsupervised_dataset = semi_supervised_dataset(train_set.data, train_set.targets,
                                                                       jparams['fcLayers'][-1], jparams['n_class'],
                                                                       jparams['train_label_number'],
                                                                       transform=transforms_type, seed=semi_seed)
    # Test set
    if fashion:
        test_set = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True,
                                                     transform=transforms_type)
    else:
        test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms_type)

    return train_set, test_set, validation_set, class_set, layer_set, supervised_dataset, unsupervised_dataset


def return_cifar10(jparams, validation=False):

    # Define the Transform
    train_transform_type = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ReshapeTransform((-1,))])

    # Train set
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=train_transform_type,
                                             target_transform=ReshapeTransformTarget(10))
    # Validation set
    if validation:
        (X_train, X_validation,
         Y_train, Y_validation) = train_test_split(torch.tensor(train_set.data), torch.tensor(train_set.targets),
                                                   test_size=0.1, random_state=34, stratify=train_set.targets)

        train_set = MyDataset(X_train, Y_train, transform=train_transform_type,
                              target_transform=ReshapeTransformTarget(10))
        validation_set = MyDataset(X_validation, Y_validation, transform=train_transform_type, target_transform=None)
    else:
        validation_set = None

    # Class set and Layer set
    if jparams['class_label_percentage'] == 1:
        class_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False,
                                                 transform=train_transform_type)
        layer_set = train_set
    else:
        class_set = SplitClass(torch.tensor(train_set.data), torch.tensor(train_set.targets),
                               jparams['class_label_percentage'], seed=34,
                               transform=train_transform_type)

        layer_set = SplitClass(torch.tensor(train_set.data), torch.tensor(train_set.targets),
                               jparams['class_label_percentage'], seed=34,
                               transform=train_transform_type,
                               target_transform=ReshapeTransformTarget(10))

    # Supervised set and Unsupervised set
    if jparams['semi_seed'] < 0:
        semi_seed = None
    else:
        semi_seed = jparams['semi_seed']

    supervised_dataset, unsupervised_dataset = semi_supervised_dataset(torch.tensor(train_set.data),
                                                                       torch.tensor(train_set.targets),
                                                                       jparams['fcLayers'][-1], jparams['n_class'],
                                                                       jparams['train_label_number'],
                                                                       transform=train_transform_type, seed=semi_seed)
    # Test set
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=train_transform_type)

    return train_set, test_set, validation_set, class_set, layer_set, supervised_dataset, unsupervised_dataset


def get_dataset(jparams, validation=False):
    if jparams['dataset'] == 'mnist':
        print('We use the MNIST Dataset')
        (train_set, test_set, validation_set,
         class_set, layer_set,
         supervised_dataset, unsupervised_dataset) = return_mnist(jparams, validation=validation)

    elif jparams['dataset'] == 'fashionMnist':
        print('We use the Fashion MNIST Dataset')
        (train_set, test_set, validation_set,
         class_set, layer_set,
         supervised_dataset, unsupervised_dataset) = return_mnist(jparams, validation=validation, fashion=True)

    elif jparams['dataset'] == 'cifar10':
        print('We use the CIFAR10 dataset')
        (train_set, test_set, validation_set,
         class_set, layer_set,
         supervised_dataset, unsupervised_dataset) = return_cifar10(jparams, validation=validation)
    else:
        raise ValueError(f'{jparams["dataset"]} dataset is not defined!')

    # load dataset

    (train_loader, test_loader, validation_loader,
     class_loader, layer_loader, supervised_loader, unsupervised_loader) = load_dataset(train_set, test_set,
                                                                                        validation_set,
                                                                                        class_set, layer_set,
                                                                                        supervised_dataset,
                                                                                        unsupervised_dataset,
                                                                                        jparams['batchSize'],
                                                                                        jparams['test_batchSize'],
                                                                                        jparams['pre_batchSize'])

    return (train_loader, test_loader, validation_loader,
            class_loader, layer_loader, supervised_loader, unsupervised_loader)


def load_dataset(train_set, test_set, validation_set, class_set, layer_set,
                 supervised_dataset, unsupervised_dataset, batchSize, test_batchSize, pre_batchSize):
    # load the dataset
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchSize, shuffle=True)

    if validation_set is not None:
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=test_batchSize,
                                                        shuffle=False)
    else:
        validation_loader = None

    class_loader = torch.utils.data.DataLoader(class_set, batch_size=test_batchSize, shuffle=False)
    layer_loader = torch.utils.data.DataLoader(layer_set, batch_size=test_batchSize, shuffle=True)
    supervised_loader = torch.utils.data.DataLoader(supervised_dataset, batch_size=pre_batchSize,
                                                    shuffle=True)
    unsupervised_loader = torch.utils.data.DataLoader(unsupervised_dataset, batch_size=batchSize,
                                                      shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batchSize, shuffle=False)

    return (train_loader, test_loader, validation_loader,
            class_loader, layer_loader, supervised_loader, unsupervised_loader)
