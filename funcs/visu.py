import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pylab as P
import random


def plot_output(x, out_nb, xlabel, ylabel, path, prefix):
    fig, ax = plt.subplots()
    label_name = range(out_nb)
    x = x.cpu()
    for out in label_name:
        ax.plot(x[:, out].numpy(), label=f'Output{out}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('The evolution of ' + ylabel + ' with respect to the ' + xlabel)
    ax.legend(label_name, loc='upper right', bbox_to_anchor=(1.12, 1.15), ncol=1, fontsize=6, frameon=False)
    plt.savefig(str(path) + prefix + ylabel + '.svg', format='svg', dpi=300)


def plot_distribution(x, out_nb, file_name, path, prefix):
    fig = plt.figure()
    colors = cm.rainbow(np.linspace(0, 1, out_nb))
    n, _, patches = P.hist(x.transpose(), 10, density=1, histtype='bar',
                           color=colors, label=range(out_nb), stacked=True)
    plt.legend(loc='upper right', bbox_to_anchor=(1.12, 1.15), ncol=1, fontsize=6, frameon=False)
    fig.savefig(str(path) + prefix + file_name + '.svg', format='svg', dpi=300)


def plot_imshow(x, out_nb, display, imShape, figName, path, prefix, random_select=True, responses=None):
    fig, axes = plt.subplots(display[0], display[1])
    fig.set_size_inches(9, 6)
    # take the random selected weights
    if random_select:
        range_of_ints = range(out_nb)
        selected_ints = random.sample(range_of_ints, display[0] * display[1])

    else:
        selected_ints = range(out_nb)

    for i, ax in zip(selected_ints, axes.flat):
        plot = ax.imshow(x[i, :].reshape(imShape[0], imShape[1]), cmap=cm.binary)
        # ax.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if responses is not None:
            ax.set_title(str(responses[i].item()))

    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    cax = plt.axes([0.92, 0.1, 0.02, 0.8])
    cb = fig.colorbar(plot, cax=cax)
    cb.ax.tick_params(labelsize=8)
    fig.suptitle('Imshow of ' + figName + ' neurons', fontsize=10)
    plt.savefig(str(path) + prefix + figName + '.svg', format='svg', dpi=300)


def plot_receptive_field(weights, display, imShape, figName, path, prefix):
    fig, axes = plt.subplots(display[0], display[1])
    fig.set_size_inches(9, 6)

    for i, ax in zip(range(weights.size(0)), axes.flat):
        plot = ax.imshow(weights[i, :].reshape(imShape[0], imShape[1]), cmap=cm.binary)
        # ax.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    cax = plt.axes([0.92, 0.1, 0.02, 0.8])
    cb = fig.colorbar(plot, cax=cax)
    cb.ax.tick_params(labelsize=8)
    fig.suptitle('Receptive field ' + figName, fontsize=10)
    plt.savefig(str(path) + prefix + figName + '.svg', format='svg', dpi=300)


def plot_one_class(x, display, imShape, indices, figName, path, prefix):
    fig, axes = plt.subplots(display[0], display[1])

    range_index = min(len(indices), display[0] * display[1])
    np.random.shuffle(indices)

    for i, ax in zip(indices[0:range_index], axes.flat[0:range_index]):
        plot = ax.imshow(x[i, :].reshape(imShape[0], imShape[1]), cmap=cm.binary)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_title('neuron ' + str(i))

    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    cax = plt.axes([0.92, 0.1, 0.02, 0.8])
    cb = fig.colorbar(plot, cax=cax)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(figName, fontsize=10)
    plt.savefig(str(path) + prefix + figName + '.svg', format='svg', dpi=300)


def plot_neach_class(x, n_class, N, display, imShape, responces, figName, path, prefix):
    # fig, axes = plt.subplots(n_class, N)
    fig, axes = plt.subplots(display[0], display[1])

    # fig, axes = plt.subplots(4, 5)
    fig.set_size_inches(9, 6)
    indx_neurons = []

    for i in range(n_class):
        index_i = (responces.cpu() == i).nonzero(as_tuple=True)[0].numpy()
        np.random.shuffle(index_i)

        range_index = min(len(index_i), N)
        indx_neurons.extend(index_i[0:range_index])

    for i, ax in zip(range(display[0] * display[1]), axes.flat):
        plot = ax.imshow(x[indx_neurons[i], :].reshape(imShape[0], imShape[1]), cmap=cm.binary)
        # ax.axis('off')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # for ind, j in zip(index_i[0:range_index], range(range_index)):
        #     plot = axes[i, j].imshow(x[ind, :].reshape(imShape[0], imShape[1]), cmap=cm.coolwarm)
        #     axes[i, j].get_xaxis().set_visible(False)
        #     axes[i, j].get_yaxis().set_visible(False)
        #     #axes[i, j].set_title(str(i.item()))

    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    cax = plt.axes([0.92, 0.1, 0.02, 0.8])
    cb = fig.colorbar(plot, cax=cax)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(figName, fontsize=10)
    plt.savefig(str(path) + prefix + figName + '.svg', format='svg', dpi=300)


def plot_spike(spike, figName, path, prefix):
    fig, ax = plt.subplots()
    ax.imshow(spike, cmap=cm.binary)
    ax.set_ylabel('Label')
    ax.set_xlabel(figName)
    ax.set_title(figName + ' for different labels')
    if spike.size(0) == 10 and spike.size(1) == 10:
        for i in range(spike.size(0)):
            for j in range(spike.size(1)):
                ax.text(j, i, spike[i, j].item(), ha="center", va="center", color="w")
    plt.savefig(str(path) + prefix + figName + '.svg', format='svg', dpi=300)
