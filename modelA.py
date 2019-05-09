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
		self.nb_hidden = nb_hidden
		self.nb_class = 10
		self.nb_output = 2
		self.conv1 = nn.Conv3d(1, 64, kernel_size=(1,3,3))
		self.conv2 = nn.Conv3d(64, 128, kernel_size=(1,3,3))
		self.conv3 = nn.Conv3d(128, 256, kernel_size=(1,2,2))
		self.fc1 = nn.Linear(256, self.nb_hidden)
		self.fc2 = nn.Linear(self.nb_hidden, self.nb_class)
		self.fc3 = nn.Linear(self.nb_output*self.nb_class,self.nb_hidden)
		self.fc4 = nn.Linear(self.nb_hidden,self.nb_output)
	def forward(self, x):
		x = F.relu(F.max_pool3d(self.conv1(x), kernel_size=(1,3,3), stride=(1,3,3)))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x1 = torch.zeros(x.size(0),2,self.nb_hidden)
		for j in range(x.size(2)):
			x1[:,j,:] = F.relu(self.fc1(x[:,:,j,:,:].reshape(self.nb_hidden,-1)))
		x_class = torch.zeros(x.size(0),2,self.nb_class)
		for j in range(x.size(2)):
			x_class[:,j,:] = self.fc2(x1[:,j,:].view(-1,self.nb_hidden))
		x3 = F.relu(self.fc3(x_class.view(-1,self.nb_output*self.nb_class)))
		x_target = self.fc4(x3)
		return x_target, x_class
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
def train_model(data, model, criterion = nn.MSELoss(), param = {"mini_batch_size" : 100,"eta" : 1e-1, "epoch" : 25, "label_target" : "class","verbose" : True}):
	# Get parameters
	mini_batch_size = param["mini_batch_size"]
	eta = param["eta"]
	epoch = param["epoch"]
	label_target = param["label_target"]
	verbose = param["verbose"]
	loss_array = torch.zeros(epoch,2)
	error_array = torch.zeros(epoch)
	param["verbose"] = False
	# Main loop
	for e in range(epoch):
		sum_loss_t = sum_loss_c = 0
		for b in range(0, data.nb, mini_batch_size):
			temp = data.train_input.narrow(0, b, mini_batch_size)
			output_target, output_class = model(temp)
			#output_target, output_class = model(temp)
			loss1 = loss2 = loss_c = loss_t = torch.zeros(1)
			if(type(criterion) is nn.MSELoss):
				#Check if there is hot point labels
				if(~data.hasHot):
					data.get_one_labels()
				target = data.train_target_hot
				loss_t = criterion(output_target,target.narrow(0, b, mini_batch_size))
				if(label_target == "auxiliary"):
						target = data.train_class_hot
						# Compute Class Losses
						loss1 = criterion(output_class[:,0,:],target.narrow(0, b, mini_batch_size)[:,0,:])
						loss2 = criterion(output_class[:,1,:],target.narrow(0, b, mini_batch_size)[:,1,:])
						loss_c = loss1 + loss2
			elif(type(criterion) is nn.CrossEntropyLoss):
					target = data.train_target
					loss_t = criterion(output_target,target.narrow(0, b, mini_batch_size))
					if(label_target == "auxiliary"):
						target = data.train_class
						# Compute Class Losses
						loss1 = criterion(output_class[:,0,:],target.narrow(0, b, mini_batch_size)[:,0])
						loss2 = criterion(output_class[:,1,:],target.narrow(0, b, mini_batch_size)[:,1])
						loss_c = loss1 + loss2
			else:
				if(verbose): print("Error: your criterion is not valid... Chose: nn.MSELoss or nn.CrossEntropyLoss")
			model.zero_grad()
			loss = loss_c +loss_t
			loss.backward()
			sum_loss_c = sum_loss_c + loss_c.item()
			sum_loss_t = sum_loss_t + loss_t.item()
			sum_loss = sum_loss_c + sum_loss_t
			with torch.no_grad():
				for p in model.parameters():
					p.data.sub_(eta * p.grad.data)
		loss_array[e,0] = sum_loss_t
		loss_array[e,1] = sum_loss_c
		error_array[e]  = compute_nb_errors(data, model,"train", param)/data.nb
		if(verbose): print(e, round(sum_loss,2), round(error_array[e].item(),2))
	return loss_array,error_array
######################################################################
def compute_nb_errors(data, model, type,param = {"mini_batch_size" : 100,"label_target" : "class","verbose" : True}):
	nb_errors = 0
	mini_batch_size = param["mini_batch_size"]
	label_target = param["label_target"]
	verbose = param["verbose"]
	if(type == "test"):
		input = data.test_input
		target = data.test_target
	else:
		input = data.train_input
		target = data.train_target
	
	for b in range(0, data.nb, mini_batch_size):
		output_target,_ = model(input.narrow(0, b, mini_batch_size))
		_, predicted_classes = output_target.data.max(1)
		for k in range(mini_batch_size):
			if target.data[b + k]!= predicted_classes[k].long():
				nb_errors = nb_errors + 1
	if(verbose) : print('Test error Net :{:0.2f}% {:d}/{:d}'.format((100 * nb_errors) /(data.nb),
																  nb_errors, (data.nb )))
	return nb_errors
######################################################################

def run(criterion, label_type, e, weightsharing):
	# Load dataset
	nb = 1000
	nb_class = 10
	eta = 1e-1
	mini_batch_size = 100
	param = {
	"nb" : nb, 
	"nb_classes" : nb_class, 
	"normalized" : True, 
	"one_hot_labels" : True,
	"conv3D" : True,
	"mini_batch_size" : mini_batch_size,
	"eta" : eta,
	"epoch" : e, 
	"label_target" : label_type,
	"verbose" : False
	}
	data = data_set(param)
	# Creat Model
	nb_hidden1 = 100
	model = Net_conv3d(nb_hidden1)
	#nb_hidden2 = 100
	#model = Net_targetLayer(model_class,nb_hidden2)
	# Metric
	if(criterion == "CrossEntropyLoss"):
		metric = nn.CrossEntropyLoss()
	else:
		metric = nn.MSELoss()
	# train the model
	loss,train_error = train_model(data, model, metric, param)
	test_errors = compute_nb_errors(data, model,"test", param) 
	
	return(loss, train_error, test_errors/nb)