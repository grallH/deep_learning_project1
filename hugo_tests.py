#!/usr/bin/env python

######################################################################

import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

import dlc_practical_prologue as prologue
######################################################################
class Net_conv3d(nn.Module):
	def __init__(self, nb_hidden):
		super(Net_conv3d, self).__init__()
		self.conv1 = nn.Conv3d(1, 64, kernel_size=(1,3,3))
		self.conv2 = nn.Conv3d(64, 128, kernel_size=(1,3,3))
		self.conv3 = nn.Conv3d(128, 256, kernel_size=(1,2,2))
		self.fc1 = nn.Linear(256, nb_hidden)
		self.fc2 = nn.Linear(nb_hidden, nb_classes)
		self.nb_hidden = nb_hidden

	def forward(self, x):
		#print('[step0] : {0}'.format(x.size()))
		x = F.relu(F.max_pool3d(self.conv1(x), kernel_size=(1,3,3), stride=(1,3,3)))
		#print('[step1] : {0}'.format(x.size()))
		x = F.relu(F.max_pool3d(self.conv2(x), kernel_size=(1,1,1), stride=(1,1,1)))
		#print('[step2] : {0}'.format(x.size()))
		x = F.relu(F.max_pool3d(self.conv3(x), kernel_size=(1,1,1), stride=(1,1,1)))
		# weight sharing manually implemented
		x1 = torch.zeros(x.size(0),2,self.nb_hidden)
		for j in range(x.size(2)):
			x1[:,j,:] = F.relu(self.fc1(x[:,:,j,:,:].reshape(100,-1)))
		#print('[step3] : {0}'.format(x1.size()))
		x2 = torch.zeros(x.size(0),2,nb_classes)
		for j in range(x.size(2)):
			x2[:,j,:] = self.fc2(x1[:,j,:].view(-1,self.nb_hidden))
		#print('[step4] : {0}'.format(x2.size()))
		return x2
######################################################################

def train_model(model, train_input, train_target, mini_batch_size):
	criterion = nn.CrossEntropyLoss()
	#criterion = nn.MSELoss()
	eta = 10e-1

	for e in range(25):
		sum_loss = 0
		for b in range(0, train_input.size(0), mini_batch_size):
			temp1 = train_input.narrow(0, b, mini_batch_size)
			#print('[train_input] : {0} [temp] : {1}'.format(train_input.size(), temp1.size()))
			output = model(train_input.narrow(0, b, mini_batch_size))
			#print('[output] : {0} [temp] : {1}'.format(output.size(), temp.size()))
			loss1 = criterion(output[:,0,:],train_target.narrow(0, b, mini_batch_size)[:,0])
			loss2 = criterion(output[:,1,:],train_target.narrow(0, b, mini_batch_size)[:,1])
			loss = loss1 + loss2
			model.zero_grad()
			loss.backward()
			sum_loss = sum_loss + loss.item()
			for p in model.parameters():
				p.data.sub_(eta * p.grad.data)
		print(e, sum_loss)

def compute_nb_errors(model, input, target, mini_batch_size):
	nb_errors = 0

	for b in range(0, input.size(0), mini_batch_size):
		output = model(input.narrow(0, b, mini_batch_size))
		predicted_classes = torch.zeros(mini_batch_size,target.size(1))
		for j in range(target.size(1)):
			_, predicted_classes[:,j] = output[:,j,:].view(-1,nb_classes).data.max(1)
		for k in range(mini_batch_size):
			for j in range (target.size(1)):
				if target.data[b + k,j] != predicted_classes[k,j].long():
					nb_errors = nb_errors + 1

	return nb_errors

######################################################################
def new_practical4():
	train_input, train_class = Variable(my_train_input.reshape(nb,1,2,14,14)), Variable(my_train_class)
	test_input, test_class = Variable(my_test_input.reshape(nb,1,2,14,14)), Variable(my_test_class)

	mini_batch_size = 100

######################################################################
	# Question 3

	for nh in [ 10,50,100,500]:
		model = Net_conv3d(nh)
		train_model(model, train_input, train_class, mini_batch_size)
		nb_test_errors = compute_nb_errors(model, test_input, test_class, mini_batch_size)
		print('test error Net nh={:d} {:0.2f}%% {:d}/{:d}'.format(nh,
																  (100 * nb_test_errors) /( test_input.size(0)*2),
																  nb_test_errors, (test_input.size(0))*2))

######################################################################
class data_set():
	def __init__(self,nb = 1000, nb_classes = 10,normalized = True, one_hot_labels = False):
		self.nb = nb
		self.nb_classes = nb_classes
		self.train_input, self.train_target, self.train_class, self.test_input, self.test_target, self.test_class = \
		prologue.generate_pair_sets(nb)
		# Normalization
		if normalized:
			self._normalize_data()
			self.isNormalized = True
		else:
			self.isNormalized = False
		# One hot labels
		if one_hot_labels:
			self._get_one_hot_labels()
			self.hasHot = True
		else:
			self.hasHot = False

	def _compute_to_one_label(self,input,target,class_label = False):
		if class_label:
			tmp = input.new_zeros(target.size(0), target.size(1),target.max() + 1)
			for j in range(target.size(1)):
				tmp[:,j,:].scatter_(1, target[:,j].view(-1, 1), 1.0)
		else:
			tmp = input.new_zeros(target.size(0), target.max() + 1)
			tmp.scatter_(1, target.view(-1, 1), 1.0)
		return tmp

	def _get_one_hot_labels(self):
		# Train_class one_hot_Labels
		target_train = self.train_class
		self.train_class_hot = self._compute_to_one_label(self.train_input, target_train, class_label = True)
		# Test_class one_hot_Labels
		target_test = self.test_class
		self.test_class_hot = self._compute_to_one_label(self.test_input, target_test, class_label = True)
		# Train_target one_hot_Labels
		target_train = self.train_target
		self.train_target_hot = self._compute_to_one_label(self.train_input, target_train)
		# Test_target one_hot_Labels
		target_test = self.test_target
		self.test_target_hot = self._compute_to_one_label(self.test_input, target_test)

	def _normalize_data(self):
		mu, std = self.train_input.mean(), self.train_input.std()
		self.train_input.sub_(mu).div_(std)
		self.test_input.sub_(mu).div_(std)
######################################################################

def mini_projet1():
	# Load dataset
	data1 = data_set(nb = 1000, nb_classes = 10, normalized = True, one_hot_labels = True)
	print("Mini Project 1: Load data success")
	
	#new_practical4()


if __name__ == "__main__":
	#practical4()
	mini_projet1()