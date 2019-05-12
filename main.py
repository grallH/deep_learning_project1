#--------------------------------------------------------------------#
#								Main
#--------------------------------------------------------------------#

######################################################################
import torch
import my_plot	as mp # Custom file: creat and save plots (errorbars)
import modelA	      # Custom file: 												To DO : add a small explaination
import modelB	      # Custom file: 												To DO : add a small explaination
######################################################################

# Variables
N = 1; # Number of folds
e = 25; # Number of epochs for the trainning

if __name__ == "__main__":
#--------------------------------------------------------------------#
	#Project 1 : Classification, weight sharing, auxiliary losses
#--------------------------------------------------------------------#
	# 1) Weight sharing :
	ws_train_error_rate = torch.zeros(N,e,2)
	ws_test_error_rate  = torch.zeros(N,2)
	ws_loss			 = torch.zeros(N,e,2,2) # Losses (class and target)
	# Params
	criterion		= "MSELoss" 
	label			= "target"
	weight_sharing1	= True
	weight_sharing2 = False
	# Multiple runs with different initializations
	for n in range(0,N):
		print(n)
		ws_loss[n,:,:,0] , ws_train_error_rate[n,:,0], ws_test_error_rate[n,0] = modelB.run(criterion, label, e, weight_sharing1)
		print(round(ws_test_error_rate[n,0].item(),3))
		ws_loss[n,:,:,1] , ws_train_error_rate[n,:,1], ws_test_error_rate[n,1] = modelB.run(criterion, label, e, weight_sharing2)
		print(round(ws_test_error_rate[n,1].item(),3))
	# Plots
	mp.plot(e,torch.Tensor.view(ws_loss[:,:,0,:],N,e,-1), "ws","loss")
	mp.plot(e,ws_train_error_rate, "ws","train")
	# Print test_error_rate mean + std
	error_mean = ws_test_error_rate.mean(0)
	error_std  = ws_test_error_rate.std(0)
	print("[Test]\n" + "[with weight sharing] : " + "error_rate: mean  = " + str(round(error_mean[0].item(),3)) + " std : " + str(round(error_std[0].item(),3)))
	print("[without weight sharing] : " + "error_rate: mean  = " + str(round(error_mean[1].item(),3)) + " std : " + str(round(error_std[1].item(),3)))

#--------------------------------------------------------------------#
	# 2) auxiliary Losses :
	ax_train_error_rate = torch.zeros(N,e,2)
	ax_test_error_rate  = torch.zeros(N,2)
	ax_loss			 = torch.zeros(N,e,2,2) # Losses (class and target)
	# Params
	criterion		= "CrossEntropyLoss"#"MSELoss" 
	label1 	= "target"
	label2		= "auxiliary"
	weight_sharing	= True
	# Multiple runs with different initializations
	for n in range(0,N):
		print(n)
		ax_loss[n,:,:,0] , ax_train_error_rate[n,:,0], ax_test_error_rate[n,0] = modelA.run(criterion, label1, e, weight_sharing)
		print(round(ax_test_error_rate[n,0].item(),3))
		ax_loss[n,:,:,1] , ax_train_error_rate[n,:,1], ax_test_error_rate[n,1] = modelA.run(criterion, label2, e, weight_sharing)
		print(round(ax_test_error_rate[n,1].item(),3))
	# Plots
	temp = torch.zeros(N,e,4)
	temp[:,:,0] = torch.Tensor.view(ax_loss[:,:,0,0],N,e)
	temp[:,:,1:3] = torch.Tensor.view(ax_loss[:,:,:,1],N,e,-1)
	temp[:,:,3] = temp[:,:,1] + temp[:,:,2]
	mp.plot(e,temp,"ax","loss")
	mp.plot(e,ax_train_error_rate, "ax","train")
	# Print test_error_rate mean + 
	error_mean = ax_test_error_rate.mean(0)
	error_std  = ax_test_error_rate.std(0)
	print("[Test]\n" + "[target] : " + "error_rate: mean  = " + str(round(error_mean[0].item(),3)) + " std : " + str(round(error_std[0].item(),3)))
	print("[with auxiliary] : " + "error_rate: mean  = " + str(round(error_mean[1].item(),3)) + " std : " + str(round(error_std[1].item(),3)))
#--------------------------------------------------------------------#
	# 3) Engigneered problem :
	ip_train_error_rate	= torch.zeros(N,e,3)
	ip_test_error_rate	= torch.zeros(N,3)
	ip_loss				= torch.zeros(N,e,2,3) # Losses (class and target)
	# Params
	criterion		= "CrossEntropyLoss" # chang here for the testing the metric
	label1			= "auxiliary"
	label2			= "target"
	label3			= "class"
	weight_sharing	= True
	# Multiple runs with different initializations
	for n in range(0,N):
		print(n)
		ip_loss[n,:,:,0] , ip_train_error_rate[n,:,0], ip_test_error_rate[n,0] = modelA.run(criterion, label1, e, weight_sharing)
		print(round(ip_test_error_rate[n,0].item(),3))
		ip_loss[n,:,:,1] , ip_train_error_rate[n,:,1], ip_test_error_rate[n,1] = modelB.run(criterion, label2, e, weight_sharing)
		print(round(ip_test_error_rate[n,1].item(),3))
		ip_loss[n,:,:,2] , ip_train_error_rate[n,:,2], ip_test_error_rate[n,2] = modelB.run(criterion, label3, e, weight_sharing)
		print(round(ip_test_error_rate[n,2].item(),3))
	# Plots
	temp = torch.zeros(N,e,3)
	temp[:,:,0] = torch.Tensor.view(ip_loss[:,:,0,0],N,e) + torch.Tensor.view(ip_loss[:,:,1,0],N,e)
	temp[:,:,1] = torch.Tensor.view(ip_loss[:,:,0,1],N,e)
	temp[:,:,2] = torch.Tensor.view(ip_loss[:,:,0,2],N,e)
	
	mp.plot(e,temp, "ip","loss")
	mp.plot(e,ip_train_error_rate, "ip","train")
	# Print test_error_rate mean + std
	error_mean = ip_test_error_rate.mean(0)
	error_std  = ip_test_error_rate.std(0)
	print("[Test]\n" + "[all trained] : " + "error_rate: mean  = " + str(round(error_mean[0].item(),3)) + " std : " + str(round(error_std[0].item(),3)))
	print("[hard coded] : target " + "error_rate: mean  = " + str(round(error_mean[1].item(),3))  + " std : " + str(round(error_std[1].item(),3)))
	print("[hard coded] : class " + "error_rate: mean  = " + str(round(error_mean[2].item(),3))  + " std : " + str(round(error_std[2].item(),3)))
#--------------------------------------------------------------------#
			#Project 2 : Mini deep-learning framework
#--------------------------------------------------------------------#
	# To do: Costa add your part