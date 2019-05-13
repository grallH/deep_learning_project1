#!/usr/bin/env python

######################################################################

import torch
import math

from torch import optim
from torch import Tensor
from torch import nn

######################################################################

def generate_disc_set(nb):
    input = Tensor(nb, 2).uniform_(-1, 1)

    target1 = input.pow(2).sum(1).sub(2 / math.pi).sign().add(1).div(2).long()
    target = torch.empty(input.size())
    target[:, 0] = 1 - target1
    target[:, 1] = target1
    return input, target


######################################################################

def train_model(model, train_input, train_target, nb_epochs, mini_batch_size, test_input, test_target):
    torch.set_grad_enabled(True)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.1)

    acc_loss_list = []
    per_train_error_list = []


    for e in range(nb_epochs):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
        acc_loss_list.append(sum_loss)
        per_train_error_list.append(compute_nb_errors(model, train_input, train_target,mini_batch_size) / train_input.size(0))

    test_error = compute_nb_errors(model, test_input, test_target,mini_batch_size) / test_input.size(0)

    return (acc_loss_list, per_train_error_list, test_error)

######################################################################

def compute_nb_errors(model, data_input, data_target,mini_batch_size):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output, 1)
        for k in range(mini_batch_size):
            if data_target[b + k, 1] != predicted_classes[k].float():
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors

######################################################################

def create_model():
    return nn.Sequential(
        nn.Linear(2, 25),
        nn.ReLU(),
        nn.Linear(25, 25),
        nn.ReLU(),
        nn.Linear(25, 25),
        nn.ReLU(),
        nn.Linear(25, 25),
        nn.ReLU(),
        nn.Linear(25, 2)

    )

######################################################################
if __name__ == "__main__":

    train_input, train_target = generate_disc_set(1000)
    test_input, test_target = generate_disc_set(1000)

    mean, std = train_input.mean(), train_input.std()

    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)

    nb_epoch = 20
    mini_batch_size = 100

    model = create_model()

    (acc_loss_list, per_train_error_list, test_error) = train_model(model, train_input, train_target, nb_epoch, mini_batch_size, test_input, test_target)

    print('{:s} train_error {:.02f}% test_error {:.02f}%'.format(
        'model',
        compute_nb_errors(model, train_input, train_target,mini_batch_size) / train_input.size(0) * 100,
        compute_nb_errors(model, test_input, test_target),mini_batch_size / test_input.size(0) * 100
    )
    )