import math
import torch
from torch import Tensor

######################################################################

class Linear :

    def __init__(self, ni, no):
        epsilon_ = math.sqrt(2 / (ni + no))

        self.w = torch.empty(no, ni).normal_(0, epsilon_)
        self.b = torch.empty(no).normal_(0, epsilon_)
        self.dl_dw = torch.zeros(self.w.size())
        self.dl_db = torch.zeros(self.b.size())

    def forward ( self , x):

        return self.w.mv(x) + self.b

    def backward ( self , x, dl_dx):
        self.dl_dw.add_(dl_dx.view(-1, 1).mm(x.view(1, -1)))
        self.dl_db.add_(dl_dx)
        return self.w.t().mv(dl_dx)

    def gradient_step(self, eta):
        self.w = self.w - eta*self.dl_dw
        self.b = self.b - eta * self.dl_db

    def zero_grad(self):
        self.dl_dw = torch.zeros(self.w.size())
        self.dl_db = torch.zeros(self.b.size())

    def param ( self ) :
        return [self.w, self.b, self.dl_dw, self.dl_db]



class Sigma :

    def forward ( self , x):
        return x.exp()/(1 + x.exp())

    def backward ( self , x, dl_dx):
        return self.forward(x) * (1-self.forward(x))*dl_dx
        #return (x.exp()/(1 + x.exp()))*(1 - x.exp()/(1 + x.exp()))*dl_dx

    def gradient_step(self, eta):
        return

    def zero_grad(self):
        return

    def param ( self ):
        return []

class Tanh :

    def forward ( self , x):
        return x.tanh()

    def backward ( self , x, dl_dx):
        return (1-self.forward(x).pow(2)) * dl_dx
        #return (4 * (x.exp() + x.mul(-1).exp()).pow(-2))*dl_dx

    def gradient_step(self, eta):
        return

    def zero_grad(self):
        return

    def param ( self ):
        return []

class Relu :


    def forward ( self , x):
        bool = (x > 0).type(x.type())
        return bool * x

    def backward ( self , x, dl_dx):
        return ((x>0).type(x.type()))*dl_dx

    def gradient_step(self, eta):
        return

    def zero_grad(self):
        return

    def param ( self ) :
        return []


class Sequential:
    """

    """
    def __init__(self, module):
        self.model = module

    def forward_pass(self, x):

        xlist = [x]

        for m in self.model:
            x = m.forward(x)
            xlist.append(x)

        return xlist

    def backward_pass(self, loss, x, t):

        xlist = self.forward_pass(x)

        dl_dx = loss.backward(xlist[-1], t)

        for i in range(len(self.model) - 1, -1, -1):
            dl_dx = self.model[i].backward(xlist[i], dl_dx)

    def gradient_step(self, eta):
        for m in self.model:
            m.gradient_step(eta)

    def zero_grad(self):
        for m in self.model:
            m.zero_grad()


class LossMSE :


    def forward ( self , v, t):
        return (v - t).pow(2).mean(0)

    def backward ( self , v, t):
        return 2 * (v - t) / v.size(0)

    def param ( self ) :
        return []


def generate_disc_set(nb):
    input = Tensor(nb, 2).uniform_(-1, 1)
    target1 = input.pow(2).sum(1).sub(2 / math.pi).sign().add(1).div(2).long()
    target = torch.empty(input.size())
    target[:, 0] = 1 - target1
    target[:, 1] = target1

    return input, target
######################################################################
def run(seq, train_input, test_input, train_target, test_target,loss,eta,Nepoch, mini_batch_size):
    torch.set_grad_enabled(False)

    acc_loss_list = []
    per_train_error_list = []
    per_test_error_list = []

    for k in range(Nepoch):

        # Compute error
        nb_test_errors = 0
        nb_train_errors = 0

        for n in range(test_input.size(0)):
            # Test
            xlist = seq.forward_pass(test_input[n])
            pred = xlist[-1].max(0)[1].item()
            if test_target[n, pred] < 0.5: nb_test_errors = nb_test_errors + 1
            # Train 
            xlist = seq.forward_pass(train_input[n])
            pred = xlist[-1].max(0)[1].item()
            if train_target[n, pred] < 0.5: nb_train_errors = nb_train_errors + 1
        
        acc_loss = 0
        # Back-prop, train loss and train error
        for b in range(0, train_input.size(0), mini_batch_size):
            seq.zero_grad()
            
            for n in range(mini_batch_size):
                x_list = seq.forward_pass(train_input[b + n])
                acc_loss = acc_loss + loss.forward(x_list[-1], train_target[b + n])
                seq.backward_pass(loss, train_input[b + n], train_target[b + n])

            # Gradient step
            seq.gradient_step(eta/mini_batch_size)

        print('{:d} acc_train_loss {:.02f} acc_train_error {:.02f}% test_error {:.02f}%'
              .format(k,
                      acc_loss/train_input.size(0),
                      (100 * nb_train_errors) / train_input.size(0),
                      (100 * nb_test_errors) / test_input.size(0)))

        acc_loss_list.append(acc_loss/train_input.size(0))
        per_train_error_list.append(nb_train_errors / train_input.size(0))
        per_test_error_list.append(nb_test_errors / test_input.size(0))


    return (acc_loss_list, per_train_error_list, per_test_error_list)
######################################################################
if __name__ == "__main__":
	nb_train_samples = 1000

	train_input, train_target = generate_disc_set(nb_train_samples)
	test_input, test_target = generate_disc_set(nb_train_samples)

	mean, std = train_input.mean(), train_input.std()

	# normalize samples
	train_input.sub_(mean).div_(std)
	test_input.sub_(mean).div_(std)

	# fixed learning rate
	eta = 0.05

	# instance fully connected layers, relu and loss
	lin1 = Linear(2, 25)
	lin2 = Linear(25, 25)
	lin3 = Linear(25, 25)
	lin4 = Linear(25, 25)
	lin5 = Linear(25, 2)
	relu = Relu()
	loss = LossMSE()

	# model is a list of ordered modules
	model = [lin1, relu, lin2, relu, lin3, relu, lin4, relu, lin5]

	# instance sequential class
	seq = Sequential(model)
	run(seq, train_input, test_input, 20, 100)
