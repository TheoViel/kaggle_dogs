import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def update_average(model_old, model_new, beta):
    """
    Updates weights using exponential moving average
    :param model_old: Model to update
    :param model_new: Model to updat with
    :param beta: EMA coefficient
    :return:
    """
    def toggle_grad(model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)

    toggle_grad(model_old, False)
    toggle_grad(model_new, False)

    param_dict_new = dict(model_new.named_parameters())
    for param_name, param_old in model_old.named_parameters():
        param_new = param_dict_new[param_name]
        assert (param_old is not param_new)
        param_old.copy_(beta * param_old + (1. - beta) * param_new)

    toggle_grad(model_old, True)
    toggle_grad(model_new, True)


def save_model_weights(model, filename, verbose=1):
    if verbose:
        print(f'-> Saving weights to {filename}')
    torch.save(model.state_dict(), filename)


def load_model_weights(model, filename, verbose=1):
    if verbose:
        print(f'-> Loading weights from {filename}')
    model.load_state_dict(torch.load(filename))
    return model


def plot_infos(infos, num_epochs):

    fig, ax1 = plt.subplots(figsize=(12, 8))

    plt.plot(infos['discriminator_loss'], label='discriminator loss', c='darkorange')
    ax1.set_ylabel('Discriminator Loss', color='darkorange', size=14)
    ax1.tick_params(axis='y', colors='darkorange')
    ax1.set_xlabel('Epochs', size=14)
    plt.grid(True)
    plt.legend(loc=(0, 1.01))

    ax2 = ax1.twinx()
    plt.plot(infos['generator_loss'], label='generator loss', c='dodgerblue')
    ax2.set_ylabel('Generator Loss', color='dodgerblue', size=14)
    ax2.tick_params(axis='y', colors='dodgerblue')
    plt.legend(loc=(0.84, 1.01))

    res = 4
    for epoch in np.cumsum(num_epochs[:-1]):
        plt.axvline(epoch, c='r', alpha=0.5)
        plt.text(x=epoch-10, y=np.max(infos['generator_loss']), s=f'{res}x{res}', bbox=dict(facecolor='red', alpha=0.25))
        res *= 2

    plt.title('Loss evolution', size=15)
    plt.show()
