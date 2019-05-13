#!/usr/bin/env python

######################################################################

import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

import dlc_practical_prologue as prologue

train_input, train_target, test_input, test_target = \
    prologue.load_data(one_hot_labels = True, normalize = True, flatten = False)

N = 1000
tri, trt, trc, tei, tet, tec = \
prologue.generate_pair_sets(N)

def convert_to_one_hot_labels(input, target):
    tmp = input.new_zeros(target.size(0), target.max() + 1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp

test_target_label = tet
trt = convert_to_one_hot_labels(tri, trt)
tet = convert_to_one_hot_labels(tei, tet)

tri = tri/255
tei = tei/255
print(trt.shape)



######################################################################

class Net(nn.Module):
    def __init__(self, nb_hidden):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(512, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 1)

    def forward(self, x):
        x1 = x.narrow(1,0,1)
        x2 = x.narrow(1,1,1)

        x1 = F.relu(F.max_pool2d(self.conv1(x1), kernel_size=2, stride=2))
        x1 = F.relu(F.max_pool2d(self.conv2(x1), kernel_size=2, stride=2))

        x2 = F.relu(F.max_pool2d(self.conv1(x2), kernel_size=2, stride=2))
        x2 = F.relu(F.max_pool2d(self.conv2(x2), kernel_size=2, stride=2))

        x = torch.cat((x1, x2) ,1)
        x = F.relu(self.fc1(x.view(-1, 512)))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

######################################################################

def train_model(model, train_input, train_target, mini_batch_size):
    print(train_target[0])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    eta = 1e-1

    for e in range(25):
        sum_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            optimizer.zero_grad()
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size)[:, 1])

            #model.zero_grad()
            loss.backward()
            optimizer.step()
            sum_loss = sum_loss + loss.item()
           # for p in model.parameters():
           #     p.data.sub_(eta * p.grad.data)
        print(e, sum_loss)

def compute_nb_errors(model, input, target, mini_batch_size):
    nb_errors = 0

    for b in range(0, input.size(0), mini_batch_size):
        output = model(input.narrow(0, b, mini_batch_size))
        #_, predicted_classes = output.data.max(1)
        predicted_classes = output.data > 0.5


        for k in range(mini_batch_size):
            true_classes = target.data[b + k]
            if true_classes != predicted_classes[k].long():
                nb_errors = nb_errors + 1

    return nb_errors


mini_batch_size = 100

for k in range(10):
    model = Net(200)
    train_model(model, tri, trt, mini_batch_size)
    nb_test_errors = compute_nb_errors(model, tei, test_target_label, mini_batch_size)
    print('test error Net {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) / test_input.size(0),
                                                      nb_test_errors, test_input.size(0)))


