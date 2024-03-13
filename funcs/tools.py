import datetime
import torch
import os
import os.path
import pandas as pd


def define_loss(loss='MSE'):
    if loss == 'MSE':
        criterion = torch.nn.MSELoss()
    elif loss == 'Cross-entropy':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f'{loss} loss is not defined!')
    return criterion


def smooth_labels(labels, smooth_factor, nudge_N):
    assert len(labels.shape) == 2, 'input should be a batch of one-hot-encoded data'
    assert 0 <= smooth_factor <= 1, 'smooth_factor should be between 0 and 1'

    if 0 <= smooth_factor <= 1:
        with torch.no_grad():
            # label smoothing
            labels *= 1 - smooth_factor
            labels += (nudge_N * smooth_factor) / labels.shape[1]
            # labels = drop_output(labels)
    else:
        raise ValueError('Invalid label smoothing factor: ' + str(smooth_factor))
    return labels


def drop_output(x, p):
    if p < 0 or p > 1:
        raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
    if p == 0:
        p_distribut = torch.ones(x.size())
    else:
        binomial = torch.distributions.binomial.Binomial(probs=torch.tensor(1 - p))
        p_distribut = binomial.sample(x.size())
    return p_distribut


def drop_all_layers(s, p):
    p_distribut = []
    for layer in range(len(s)):
        if p[layer] < 0 or p[layer] > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p[layer]))
        if p[layer] == 0:
            p_distribut.append(torch.ones(s[layer].size()))
        else:
            binomial = torch.distributions.binomial.Binomial(probs=torch.tensor(1 - p[layer]))
            p_distribut.append(binomial.sample(s[layer].size()))
    return p_distribut


def define_unsupervised_target(output, N, device, Homeo=None):
    with torch.no_grad():

        # define unsupervised target
        unsupervised_targets = torch.zeros(output.size(), device=device)

        # N_maxindex
        if Homeo is not None:
            n_maxindex = torch.topk(output.detach() - Homeo, N).indices
        else:
            n_maxindex = torch.topk(output.detach(), N).indices

        unsupervised_targets.scatter_(1, n_maxindex, torch.ones(output.size(), device=device))  # WTA definition

    return unsupervised_targets, n_maxindex


def direct_association(class_record, labels_record, k_select):
    # take the maximum activation as associated class
    class_moyenne = torch.div(class_record, labels_record)
    response = torch.argmax(class_moyenne, 0)
    # remove the unlearned neuron
    max0_indice = (torch.max(class_moyenne, 0).values == 0).nonzero(as_tuple=True)[0]
    response[max0_indice] = -1

    if k_select is None:
        return response, None
    else:
        k_select_neuron = torch.topk(class_moyenne, k_select, dim=1).indices.flatten()
        return response, k_select_neuron


class One2one:
    def __init__(self, class_number):
        self.class_number = class_number

    def average_predict(self, output, response):

        classvalue = torch.zeros(output.size(0), self.class_number, device=output.device)

        for i in range(self.class_number):
            indice = (response == i).nonzero(as_tuple=True)[0]
            if len(indice) == 0:
                classvalue[:, i] = -1
            else:
                classvalue[:, i] = torch.mean(output[:, indice], 1)

        return torch.argmax(classvalue, 1)

    @staticmethod
    def max_predict(output, response):

        non_response_indice = (response == -1).nonzero(as_tuple=True)[0]  # remove the non response neurons
        output[:, non_response_indice] = -1

        maxindex_output = torch.argmax(output, 1)
        predict_max = response[maxindex_output]

        return predict_max


class PathCreator:
    def __init__(self, folder_prefix='bp-'):
        self.folder_prefix = folder_prefix

    def create_path(self):
        base_path = os.path.join(os.getcwd(), 'DATA-0', datetime.datetime.now().strftime("%Y-%m-%d"))

        # Ensure the base directory exists
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        print(f"len(BASE_PATH)={len(base_path)}")

        files = os.listdir(base_path)

        # Find the next directory with the specified prefix
        max_index = 0
        for file in files:
            if file.startswith(self.folder_prefix):
                try:
                    index = int(file.split('-')[1])
                    max_index = max(max_index, index)
                except ValueError:
                    continue

        # Create and return the new directory path
        new_path = os.path.join(base_path, f'{self.folder_prefix}{max_index + 1}')
        os.mkdir(new_path)
        name = os.path.basename(new_path)

        return new_path, name


def init_bp_dataframe(path, method='bp', dataframe_to_init='results.csv'):
    # Initiate a dataframe to save the Backpropagation training results
    if os.path.isfile(path + dataframe_to_init):
        dataframe = pd.read_csv(path + dataframe_to_init, sep=',', index_col=0)
    else:
        if method == 'bp':
            columns_header = ['Train_Error', 'Min_Train_Error', 'Test_Error', 'Min_Test_Error']
        elif method == 'unsupervised_bp':
            columns_header = ['One2one_av_Error', 'Min_One2one_av', 'One2one_max_Error', 'Min_One2one_max_Error']
        elif method == 'semi_supervised_bp':
            columns_header = ['Unsupervised_Test_Error', 'Min_Unsupervised_Test_Error', 'Supervised_Test_Error',
                              'Min_Supervised_Test_Error']
        elif method == 'classification_layer':
            columns_header = ['Train_Class_Error', 'Min_Train_Class_Error', 'Final_Test_Error', 'Min_Final_Test_Error',
                              'Final_Test_Loss', 'Min_Final_Test_Loss']
        else:
            raise ValueError(f'{method} frame type is not defined!')

        dataframe = pd.DataFrame({}, columns=columns_header)
        dataframe.to_csv(os.path.join(path, dataframe_to_init))

    return dataframe


def init_ep_dataframe(path, method='supervised', dataframe_to_init='results.csv'):
    # Initiate a dataframe to save the Backpropagation training results
    if os.path.isfile(path + dataframe_to_init):
        dataframe = pd.read_csv(path + dataframe_to_init, sep=',', index_col=0)
    else:
        if method == 'supervised':
            columns_header = ['Train_Error', 'Min_Train_Error', 'Test_Error', 'Min_Test_Error']
        elif method == 'unsupervised':
            columns_header = ['One2one_av_Error', 'Min_One2one_av', 'One2one_max_Error', 'Min_One2one_max_Error']
        elif method == 'semi-supervised':
            # TODO maybe to be changed
            columns_header = ['Unsupervised_Test_Error', 'Min_Unsupervised_Test_Error',
                              'Supervised_Test_Error', 'Min_Supervised_Test_Error']
        elif method == 'classification_layer':
            columns_header = ['Train_Class_Error', 'Min_Train_Class_Error', 'Final_Test_Error', 'Min_Final_Test_Error',
                              'Final_Test_Loss', 'Min_Final_Test_Loss']
        else:
            raise ValueError(f'{method} frame type is not defined!')

        dataframe = pd.DataFrame({}, columns=columns_header)
        dataframe.to_csv(os.path.join(path, dataframe_to_init))
    return dataframe


def update_dataframe(BASE_PATH, dataframe, error1, error2, filename='results.csv', loss=None):
    # update the dataframe
    if loss is None:
        data = [error1[-1], min(error1), error2[-1], min(error2)]
    else:
        data = [error1[-1], min(error1), error2[-1], min(error2), loss[-1], min(loss)]

    new_data = pd.DataFrame([data], index=[1], columns=dataframe.columns)
    dataframe = pd.concat([dataframe, new_data], axis=0)

    try:
        dataframe.to_csv(os.path.join(BASE_PATH, filename))
    except PermissionError:
        input("Close the {} and press any key.".format(filename))

    return dataframe
