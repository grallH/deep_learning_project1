import math
import torch

import dlc_practical_prologue as prologue

######################################################################


class Linear :

    def __init__(self, w, b):

        self.w = w
        self.b = b
        self.dl_dw = torch.zeros(w.size())
        self.dl_db = torch.zeros(b.size())

    def forward ( self , x):

        return self.w.mv(x) + self.b

    def backward ( self , x, dl_dx):
        self.dl_dw.add_(dl_dx.view(-1, 1).mm(x.view(1, -1)))
        self.dl_db.add_(dl_dx)
        return self.w.t().mv(dl_dx)

    def gradient_step(self, eta):
        self.w = self.w - eta*self.dl_dw
        self.b = self.b - eta * self.dl_db

    def set_dl_zero(self):
        self.dl_dw = torch.zeros(self.w.size())
        self.dl_db = torch.zeros(self.b.size())

    def param ( self ) :
        return [self.w, self.b]



class Sigma :

    def forward ( self , x):
        return x.tanh()

    def backward ( self , x, dl_dx):
        return (4 * (x.exp() + x.mul(-1).exp()).pow(-2))*dl_dx

    def param ( self ):
        return []


class Relu :


    def forward ( self , x):
        bool = (x > 0).type(x.type())
        return bool * x

    def backward ( self , x, dl_dx):
        return ((x>0).type(x.type()))*dl_dx

    def param ( self ) :
        return []


class Loss :


    def forward ( self , v, t):
        return (v - t).pow(2).sum()

    def backward ( self , v, t):
        return 2 * (v - t)

    def param ( self ) :
        return []



train_input, train_target, test_input, test_target = prologue.load_data(one_hot_labels=True, normalize=True)

nb_classes = train_target.size(1)
nb_train_samples = train_input.size(0)

zeta = 0.90

train_input = train_input * zeta
test_input = test_input * zeta

nb_hidden = 50
eta = 1e-1 / nb_train_samples
epsilon = 1e-6

w1 = torch.empty(nb_hidden, train_input.size(1)).normal_(0, epsilon)
b1 = torch.empty(nb_hidden).normal_(0, epsilon)
w2 = torch.empty(nb_classes, nb_hidden).normal_(0, epsilon)
b2 = torch.empty(nb_classes).normal_(0, epsilon)

lin1 = Linear(w1, b1)
sigma = Sigma()
lin2 = Linear(w2, b2)
relu = Relu()
loss = Loss()

model = [lin1, sigma, lin2, relu]


def forward_pass(model, x):

    xlist = [x]

    for m in model:
        x = m.forward(x)
        xlist.append(x)

    return xlist


def backward_pass(model, loss, x, t):

    xlist = forward_pass(model, x)

    dl_dx = loss.backward(xlist[-1], t)


    for i in range( len(model) - 1, -1, -1) :
       dl_dx = model[i].backward(xlist[i], dl_dx)


for k in range(100):

    # Back-prop

    acc_loss = 0
    nb_train_errors = 0

    lin1.set_dl_zero()
    lin2.set_dl_zero()

    for n in range(nb_train_samples):

        x_list = forward_pass(model, train_input[n])

        pred = x_list[-1].max(0)[1].item()

        if train_target[n, pred] < 0.5: nb_train_errors = nb_train_errors + 1
        acc_loss = acc_loss + loss.forward(x_list[-1], train_target[n])

        backward_pass(model, loss, train_input[n], train_target[n])

    # Gradient step

    lin1.gradient_step(eta)
    lin2.gradient_step(eta)

    # Test error

    nb_test_errors = 0

    for n in range(test_input.size(0)):
        xlist = forward_pass(model, test_input[n])
        x2 = xlist[-1]

        pred = x2.max(0)[1].item()
        if test_target[n, pred] < 0.5: nb_test_errors = nb_test_errors + 1

    print('{:d} acc_train_loss {:.02f} acc_train_error {:.02f}% test_error {:.02f}%'
          .format(k,
                  acc_loss,
                  (100 * nb_train_errors) / train_input.size(0),
                  (100 * nb_test_errors) / test_input.size(0)))





