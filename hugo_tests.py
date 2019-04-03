#!/usr/bin/env python

######################################################################

import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

import dlc_practical_prologue as prologue
######################################################################
class Net_conv3d(nn.Module):
	def __init__(self, nb_hidden,nb_output):
		super(Net_conv3d, self).__init__()
		self.nb_hidden = nb_hidden
		self.nb_output = nb_output
		self.conv1 = nn.Conv3d(1, 64, kernel_size=(1,3,3))
		self.conv2 = nn.Conv3d(64, 128, kernel_size=(1,3,3))
		self.conv3 = nn.Conv3d(128, 256, kernel_size=(1,2,2))
		self.fc1 = nn.Linear(256, self.nb_hidden)
		self.fc2 = nn.Linear(self.nb_hidden, self.nb_output)

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
		x2 = torch.zeros(x.size(0),2,self.nb_output)
		for j in range(x.size(2)):
			x2[:,j,:] = self.fc2(x1[:,j,:].view(-1,self.nb_hidden))
		#print('[step4] : {0}'.format(x2.size()))
		return x2

	def freeze_features(self, q):
		for p in self.parameters():
		# q = True means that it is frozen and we do NOT need the gradient
			p.requires_grad = not q
######################################################################
class Net_conv3dbis(nn.Module):
	def __init__(self, nb_hidden,nb_output):
		super(Net_conv3dbis, self).__init__()
		self.nb_hidden = nb_hidden
		self.nb_output = nb_output
		self.conv1 = nn.Conv3d(1, 64, kernel_size=(1,3,3))
		self.conv2 = nn.Conv3d(64, 128, kernel_size=(1,3,3))
		self.fc1 = nn.Linear(128, self.nb_hidden)
		self.fc2 = nn.Linear(self.nb_hidden, self.nb_output)

	def forward(self, x):
		#print('[step0] : {0}'.format(x.size()))
		x = F.relu(F.max_pool3d(self.conv1(x), kernel_size=(1,3,3), stride=(1,3,3)))
		#print('[step1] : {0}'.format(x.size()))
		x = F.relu(F.max_pool3d(self.conv2(x), kernel_size=(1,2,2), stride=(1,2,2)))
		#print('[step2] : {0}'.format(x.size()))
		# weight sharing manually implemented
		x1 = torch.zeros(x.size(0),2,self.nb_hidden)
		for j in range(x.size(2)):
			x1[:,j,:] = F.relu(self.fc1(x[:,:,j,:,:].reshape(100,-1)))
		#print('[step3] : {0}'.format(x1.size()))
		x2 = torch.zeros(x.size(0),2,self.nb_output)
		for j in range(x.size(2)):
			x2[:,j,:] = self.fc2(x1[:,j,:].view(-1,self.nb_hidden))
		#print('[step4] : {0}'.format(x2.size()))
		return x2

	def freeze_features(self, q):
		for p in self.parameters():
		# q = True means that it is frozen and we do NOT need the gradient
			p.requires_grad = not q
######################################################################
class Net_targetLayer(nn.Module):
	def __init__(self,m,nb_hidden):
		super(Net_targetLayer,self).__init__()
		self.model_f = m.forward
		self.fc1 = nn.Linear(2*10,nb_hidden)
		self.fc2 = nn.Linear(nb_hidden,2)
	def forward(self,x):
		x = self.model_f(x)
		x = F.relu(self.fc1(x.view(-1,2*10)))
		x = self.fc2(x)
		return x
######################################################################
def train_model(data, model, criterion = nn.MSELoss(), param = {"mini_batch_size" : 100,"eta" : 1e-1, "epoch" : 25, "label_target" : "class","verbose" : True}):
	# Get parameters
	mini_batch_size = param["mini_batch_size"]
	eta = param["eta"]
	epoch = param["epoch"]
	label_target = param["label_target"]
	verbose = param["verbose"]
	
	for e in range(epoch):
		sum_loss = 0
		for b in range(0, data.nb, mini_batch_size):
			temp = data.train_input.narrow(0, b, mini_batch_size)
			#print('[train_input] : {0} [temp] : {1}'.format(train_input.size(), temp1.size()))
			output = model(temp)
			#print('[output] : {0} [temp] : {1}'.format(output.size(), temp.size()))
			loss1 = loss2 = torch.zeros(1)
			if(type(criterion) is nn.MSELoss):
				#Check if there is hot point labels
				if(~data.hasHot):
					data.get_one_labels()
				if(label_target == "class"):
					target = data.train_class_hot
					# Compute Loss for hot points labels
					loss1 = criterion(output[:,0,:],target.narrow(0, b, mini_batch_size)[:,0,:])
					loss2 = criterion(output[:,1,:],target.narrow(0, b, mini_batch_size)[:,1,:])
					loss = loss1 + loss2
				elif(label_target == "target"):
					target = data.train_target_hot
					#print("[output] : {} [target] : {}".format(output.size(),target.size()))
					loss = criterion(output,target.narrow(0, b, mini_batch_size))
				else:
					if(verbose):print("Error: " + label_target + " is not a valide option... Chose: label_target or class")
				
			elif(type(criterion) is nn.CrossEntropyLoss):
				if(label_target == "class"):
					target = data.train_class
					# Compute Loss
					loss1 = criterion(output[:,0,:],target.narrow(0, b, mini_batch_size)[:,0])
					loss2 = criterion(output[:,1,:],target.narrow(0, b, mini_batch_size)[:,1])
					loss = loss1 + loss2
				elif(label_target == "target"):
					target = data.train_target
					#print("[output] : {} [target] : {}".format(output.size(),target.size()))
					loss = criterion(output,target.narrow(0, b, mini_batch_size))
				else:
					if(verbose):print("Error: " + label_target + " is not a valide option... Chose: label_target or class")

			else:
				if(verbose): print("Error: your criterion is not valide... Chose: nn.MSELoss or nn.CrossEntropyLoss")
			model.zero_grad()
			loss.backward()
			sum_loss = sum_loss + loss.item()
			for p in model.parameters():
				p.data.sub_(eta * p.grad.data)
		if(verbose): print(e, sum_loss)

def compute_nb_errors(data, model, param = {"mini_batch_size" : 100,"label_target" : "class","verbose" : True}):
	nb_errors = 0
	mini_batch_size = param["mini_batch_size"]
	label_target = param["label_target"]
	verbose = param["verbose"]
	
	for b in range(0, data.nb, mini_batch_size):
		output = model(data.test_input.narrow(0, b, mini_batch_size))
		if(label_target == "class"):
			nb_target = data.test_class.size(1)
		else:
			nb_target = 1
		predicted_classes = torch.zeros(mini_batch_size,nb_target)
		if(label_target == "class"):
			target = data.test_class
			for j in range(target.size(1)):
				_, predicted_classes[:,j] = output[:,j,:].view(-1,data.nb_classes).data.max(1)
			for k in range(mini_batch_size):
				for j in range (target.size(1)):
					if target.data[b + k,j] != predicted_classes[k,j].long():
						nb_errors = nb_errors + 1
		elif(label_target == "target"):
			target = data.test_target
			_, predicted_classes = output.data.max(1)
			for k in range(mini_batch_size):
				if target.data[b + k]!= predicted_classes[k].long():
					nb_errors = nb_errors + 1
		else:
			if(verbose):print("Error: " + label_target + " is not a valide option... Chose: label_target or class")
	if(verbose) : print('Test error Net :{:0.2f}%% {:d}/{:d}'.format((100 * nb_errors) /(data.nb * nb_target),
																  nb_errors, (data.nb * nb_target)))
	return nb_errors
######################################################################
class data_set():
	""" 
	member variables:
		self.nb : nb training/testing points
		self.nb_classes: nb classes (10 differents numbers)
		self.isNormalized: {True, False} tell if the data points have been normalized or not
		self.hasHot: {True, False} tell if there is a set of hot points labels
		self.isConv3D: {True,False} tell if the data points are compatibles with conv3D
	------------------------------------------------------------------------------------------
	private functions:
		_compute_to_one_label:
		_get_conv3D_data:
		_get_one_hot_labels:
		_normalize_data:"""
	def __init__(self,param = {"nb" : 1000, "nb_classes" : 10,"normalized" : True, "one_hot_labels" : False, "conv3D" : False}):
		self.nb = param["nb"]
		self.nb_classes = param["nb_classes"]
		self.train_input, self.train_target, self.train_class, self.test_input, self.test_target, self.test_class = \
		prologue.generate_pair_sets(self.nb)
		# Normalization
		if param["normalized"]:
			self._normalize_data()
			self.isNormalized = True
		else:
			self.isNormalized = False
		# One hot labels
		if param["one_hot_labels"]:
			self._get_one_hot_labels()
			self.hasHot = True
		else:
			self.hasHot = False
		# Conv3D reshape dataset
		if param["conv3D"]:
			self._get_conv3D_data()
			self.isConv3D = True
		else:
			self.isConv3D = False
	
	def get_one_labels(self):
		if(~self.hasHot):
			self._get_one_hot_labels()
			self.hasHot = True

	def _compute_to_one_label(self,input,target,class_label = False):
		if class_label:
			tmp = input.new_zeros(target.size(0), target.size(1),target.max() + 1)
			for j in range(target.size(1)):
				tmp[:,j,:].scatter_(1, target[:,j].view(-1, 1), 1.0)
		else:
			tmp = input.new_zeros(target.size(0), target.max() + 1)
			tmp.scatter_(1, target.view(-1, 1), 1.0)
		return tmp

	def _get_conv3D_data(self):
		self.train_input = self.train_input.view(-1,1,2,14,14)
		self.test_input = self.test_input.view(-1,1,2,14,14)
		
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
	param = {
	"nb" : 1000, 
	"nb_classes" : 10, 
	"normalized" : True, 
	"one_hot_labels" : True,
	"conv3D" : True
	}
	data1 = data_set(param)
	print("Mini Project 1: Load data success")
	## Example:
	# 1) Choose model:
	nb_hidden = 200
	nb_output = data1.nb_classes
	model = Net_conv3dbis(nb_hidden,nb_output) #Net_conv3d(nb_hidden,nb_output) #Net_conv3dbis(nb_hidden,nb_output)
	# 2) Choose criterion:
	criterion = nn.CrossEntropyLoss() #nn.CrossEntropyLoss() #nn.MSELoss()
	# 3) Choose Parameters
	param = {
	"mini_batch_size" : 100,
	"eta" : 1e-1,
	"epoch" : 25, 
	"label_target" : "class",
	"verbose" : True
	}
	# train the model
	print("Mini Project 1: Training begins")
	train_model(data1, model, criterion, param)
	print("Mini Project 1: Training finishes")
	# test the model
	param1 = {
	"mini_batch_size" : 100,
	"label_target" : "class",
	"verbose" : True
	}
	nb_test_errors = compute_nb_errors(data1, model, param1)
	
	#PHASE 2
	
	## Example:
	# 1) Choose model:
	nb_hidden = 100
	model.freeze_features(True) # Disable the gradient
	model2 = Net_targetLayer(model,nb_hidden)
	# 2) Choose criterion:
	criterion = nn.CrossEntropyLoss() #nn.CrossEntropyLoss() #nn.MSELoss()
	# 3) Choose Parameters
	param = {
	"mini_batch_size" : 100,
	"eta" : 2e-1,
	"epoch" : 25, 
	"label_target" : "target",
	"verbose" : True
	}
	# train the model
	print("Mini Project 1: Training begins")
	train_model(data1, model2, criterion, param)
	print("Mini Project 1: Training finishes")
	# test the model
	param1 = {
	"mini_batch_size" : 100,
	"label_target" : "target",
	"verbose" : True
	}
	nb_test_errors = compute_nb_errors(data1, model2, param1)
																  
	#new_practical4()


if __name__ == "__main__":
	#practical4()
	mini_projet1()