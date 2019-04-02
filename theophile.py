
######################################################################
#
# This is free and unencumbered software released into the public domain.
# 
# Anyone is free to copy, modify, publish, use, compile, sell, or
# distribute this software, either in source code form or as a compiled
# binary, for any purpose, commercial or non-commercial, and by any
# means.
# 
# In jurisdictions that recognize copyright laws, the author or authors
# of this software dedicate any and all copyright interest in the
# software to the public domain. We make this dedication for the benefit
# of the public at large and to the detriment of our heirs and
# successors. We intend this dedication to be an overt act of
# relinquishment in perpetuity of all present and future rights to this
# software under copyright law.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
# 
# For more information, please refer to <http://unlicense.org/>
#
######################################################################

import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

import dlc_practical_prologue as prologue

class Net(nn.Module):
    def __init__(self, nbhidden):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(256, nbhidden)
        self.fc2 = nn.Linear(nbhidden, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=3))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x
        
class Net2(nn.Module):
    def __init__(self, nbhidden):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(512, nbhidden)
        self.fc2 = nn.Linear(nbhidden, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(-1, 512)))
        x = self.fc2(x)
        return x


def train_model(model, train_input, train_target, mini_batch_size):
	train_input, train_target = Variable(train_input), Variable(train_target)
	criterion = nn.MSELoss()
	eta = 1e-1
	for e in range(0, 25):
	    sum_loss = 0
  	  # We do this with mini-batches
	    for b in range(0, train_input.size(0), mini_batch_size):
	        output = model(train_input.narrow(0, b, mini_batch_size))
	        loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
	        sum_loss = sum_loss + loss.item()
	        model.zero_grad()
	        loss.backward()
	        for p in model.parameters():
	            p.data.sub_(eta * p.grad.data)
	    # print(e, sum_loss)

def compute_nb_errors(model, input, target, mini_batch_size):
	res = torch.zeros(target.size())
	
	for b in range(0, input.size(0), mini_batch_size):
		output = model(input.narrow(0, b, mini_batch_size))
		res[torch.arange(b,b+mini_batch_size,dtype=torch.int64),output.argmax(1)] = 1
	return (target != res).sum()/2
		
train_input, train_target, test_input, test_target = \
    prologue.load_data(one_hot_labels = True, normalize = True, flatten = False)


if __name__ == "__main__":
	mini_batch_size = 100
	
	train_input, train_target, train_class, test_input, test_target, test_class = \
		prologue.generate_pair_sets(self.nb)
	
	train_target = prologue.convert_to_one_hot_labels(train_input, train_target)
	train_class[:,0] = 
	

	"""for k,nhidden in enumerate(hidden):
		for j in range(0,n_iter):
			model = Net2(nhidden.item())
			train_model(model, train_input, train_target, mini_batch_size)
			nb_errors_j[j] = compute_nb_errors(model, test_input, test_target, mini_batch_size)
			print("Progress: " + str((k*n_iter + j)*100/(n_iter*hidden.size(0))) + "%")
		nb_errors[k] = nb_errors_j.mean()*100/test_target.size(0)
		print("n_hidden: " + str(nhidden) + " / Error: " + str(nb_errors[k]))"""

