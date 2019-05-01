
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
#import hugo_test as h

import dlc_practical_prologue as prologue

class Net1(nn.Module):
	def __init__(self, nb_hidden,nb_output):
		super(Net1, self).__init__()
		self.nb_hidden = nb_hidden
		self.nb_output = nb_output
		self.conv1 = nn.Conv2d(1, 64, kernel_size=(3,3))
		self.conv2 = nn.Conv2d(64, 128, kernel_size=(3,3))
		self.conv3 = nn.Conv2d(128, 256, kernel_size=(2,2))
		self.fc1 = nn.Linear(256, self.nb_hidden)
		self.fc2 = nn.Linear(self.nb_hidden, self.nb_output)

	def forward(self, x):
		#print('[step0] : {0}'.format(x.size()))
		x = x.view(-1, 1, x.size(2), x.size(3)) # uncomment for weight sharing
		x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=(3,3), stride=(3,3)))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = F.relu(self.fc1(x.view(x.size(0),-1)))
		x = self.fc2(x)
		#print('[step1] : {0}'.format(x.size()))
		x = x.view(-1,2,x.size(1))
		return x
		
class Net2(nn.Module): # without weight sharing
	def __init__(self, nb_hidden,nb_output):
		super(Net2, self).__init__()
		self.nb_hidden = nb_hidden
		self.nb_output = nb_output
		self.conv1a = nn.Conv2d(1, 64, kernel_size=(3,3))
		self.conv1b = nn.Conv2d(1, 64, kernel_size=(3,3))
		self.conv2a = nn.Conv2d(64, 128, kernel_size=(3,3))
		self.conv2b = nn.Conv2d(64, 128, kernel_size=(3,3))
		self.conv3a = nn.Conv2d(128, 256, kernel_size=(2,2))
		self.conv3b = nn.Conv2d(128, 256, kernel_size=(2,2))
		self.fc1a = nn.Linear(256, self.nb_hidden)
		self.fc1b = nn.Linear(256, self.nb_hidden)
		self.fc2a = nn.Linear(self.nb_hidden, self.nb_output)
		self.fc2b = nn.Linear(self.nb_hidden, self.nb_output)

	def forward(self, x):
		x = x.view(-1, 1, x.size(2), x.size(3))
		#print('[x] : {0}'.format(x.size()))
		xa = x[0:x.size(0)//2,:,:,:]
		xb = x[x.size(0)//2:,:,:,:]
		#print('[xa] : {0}'.format(xa.size()))
		#print('[xb] : {0}'.format(xb.size()))
		xa = F.relu(F.max_pool2d(self.conv1a(xa), kernel_size=(3,3), stride=(3,3)))
		xb = F.relu(F.max_pool2d(self.conv1b(xb), kernel_size=(3,3), stride=(3,3)))
		xa = F.relu(self.conv2a(xa))
		xb = F.relu(self.conv2b(xb))
		xa = F.relu(self.conv3a(xa))
		xb = F.relu(self.conv3b(xb))
		xa = F.relu(self.fc1a(xa.view(xa.size(0),-1)))
		xb = F.relu(self.fc1b(xb.view(xb.size(0),-1)))
		xa = self.fc2a(xa)
		xb = self.fc2b(xb)
		x = torch.cat([xa,xb])
		#print('[x] : {0}'.format(x.size()))
		x = x.view(-1,2,x.size(1))
		return x

def class_to_target(output,onehot):
	output = (output.transpose(2,0) - output.transpose(2,0).min(0)[0]).transpose(0,2)
	output = output**6
	class1 = output.narrow(1,0,1).transpose(1,2)
	class2 = output.narrow(1,1,1)
	class1 = F.normalize(class1,p=1,dim=1)
	class2 = F.normalize(class2,p=1,dim=2)
	output = class1.matmul(class2)*torch.ones(output.size(2),output.size(2)).triu()
	output = output.sum(2).sum(1)
	if(onehot):
		output = torch.cat((1-output.view(-1,1),output.view(-1,1)),1)
	return output

def train_model(model, train_input, train_target, mini_batch_size, criterion, objective, n_iter):
	train_input, train_target = Variable(train_input), Variable(train_target)
	eta = 1e-1
	for e in range(0, n_iter):
	    sum_loss = 0
  	  # We do this with mini-batches
	    for b in range(0, train_input.size(0), mini_batch_size):
	        output = model(train_input.narrow(0, b, mini_batch_size))
	        target = train_target.narrow(0,b,mini_batch_size)
	        if objective == "target":
	        	output = class_to_target(output,True)
	        elif objective == "class":
	        	output = output.view(-1,output.size(2))
	        	if type(criterion) is nn.MSELoss:
		        	target = target.view(-1,target.size(2))
		        elif type(criterion) is nn.CrossEntropyLoss:
		        	target = target.view(-1)
	        loss = criterion(output, target)
	        sum_loss = sum_loss + loss.item()
	        model.zero_grad()
	        loss.backward()
	        for p in model.parameters():
	            p.data.sub_(eta * p.grad.data)
	    #print(e, sum_loss)

def compute_nb_errors(model, input, target, mini_batch_size, onehot):
    target = target.float()
    if (onehot):
        target = target[:,1]
    nb_errors = 0
    for b in range(0, input.size(0), mini_batch_size):
        output = model(input.narrow(0, b, mini_batch_size))
        #print(output.mean())
        output = class_to_target(output,False)
        output = (output>0.5).float() # Threshold
        for k in range(mini_batch_size):
            if target[b+k] != output[k]:
                nb_errors = nb_errors + 1

    return nb_errors

def to_one_hot(input,label):
	if label.dim() == 2:
		nb = train_class.unique().size(0)
		lab = torch.zeros(train_target.size(0), 2, nb)
		lab[:,0,:] = prologue.convert_to_one_hot_labels(input, label[:,0])
		lab[:,1,:] = prologue.convert_to_one_hot_labels(input, label[:,1])
	elif label.dim() == 1:
		lab = prologue.convert_to_one_hot_labels(input, label)
	else:
		error("Error: Incorrect label shape, impossible to convert to one hot encoding")
	return(lab)

def test():
	mini_batch_size = 100
	m = 1000;
	#label_type = "target"
	label_type = "class"
	nb_class = 10
	nb_hidden = 100
	n_iter = 25
	#criterion = nn.MSELoss()
	criterion = nn.CrossEntropyLoss()
	onehot = type(criterion) is nn.MSELoss
		
	train_input, train_target, train_class, test_input, test_target, test_class = \
		prologue.generate_pair_sets(m)
	# Size of input is 1000,2,14,14
	
	# Normalize input
	train_input.sub_(train_input.mean()).div_(train_input.std())
	test_input.sub_(train_input.mean()).div_(train_input.std())
	
	test_label = test_target
	# Select train labels
	if label_type == "target":
		train_label = train_target
	elif label_type == "class":
		train_label = train_class
	else:
		error("Incorrect label type")
	
	# Convert to one hot labels if needed
	if onehot:
		train_label = to_one_hot(train_input, train_label)
		test_label = to_one_hot(test_input, test_label)
	
	model = Net1(nb_hidden,nb_class) # weight sharing
	#model = Net2(nb_hidden,nb_class) # no weight sharing
	train_model(model, train_input, train_label, mini_batch_size, criterion, label_type, n_iter)
	nb_errors = compute_nb_errors(model, test_input, test_label, mini_batch_size, onehot)
	#print('[Nb errors] : {0}'.format(nb_errors))
	#print('[% errors] : {0}'.format(100*nb_errors/m))
	return(100*nb_errors/m)
	
if __name__ == "__main__":
	n_test = 10
	error_mean = 0.
	for j in range(0, n_test):
		error_perc = test()
		print('Test {0} [% errors] : {1}'.format(j, error_perc))
		error_mean += error_perc
	error_mean /= n_test
	print('Average [% errors] : {0}'.format(error_mean))
		