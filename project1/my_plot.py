#!/usr/bin/env python

######################################################################
import matplotlib.pyplot as plt

######################################################################
# Plot param
flag_save_figure = False
folder_name = "picture/"
xLabel = {"ws_loss" : "Different epochs" , "ws_train" : "Different epochs", "ax_loss" : "Different epochs" , "ax_train" : "Different epochs", "ip_loss" : "Different epochs" , "ip_train" : "Different epochs"}
yLabel = {"ws_loss" : "MSELoss " , "ws_train" : "Error rate","ax_loss" : "CrossEntropyLoss " , "ax_train" : "Error rate","ip_loss" : "Loss " , "ip_train" : "Error rate"}
legend = {"ws" : ("With weight sharing", "Without weight sharing"), "ax" : ("Without auxiliary", "With auxiliary : target", "With auxiliary : class", "With auxiliary : target + class"), "ip" : ("all trained", "hard coded : target", "hard coded : class")}
title1  = {"ws" : "Weight Sharing : ","ax" : "Auxiliary Loss : ", "ip" : "Engineered problem : "}
title2 = {"loss" : "Loss Vs Epochs", "train": "Error rate Vs Epochs"}
title_font = {'fontname':'Arial', 'size':'20', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'}
label_font = {'fontname':'Arial', 'size':'14', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'}
name_figure = {"ws_train" : "ws_train", "ax_train" : "ax_train", "ip_train" : "ip_train","ws_loss" : "ws_loss", "ax_loss" : "ax_loss", "ip_loss" : "ip_loss"}

def plot(e,data,type1, type2):
	x_label = xLabel[type1+ "_" + type2]
	y_label = yLabel[type1+ "_" + type2]
	legends = legend[type1]
	my_title1 = title1[type1]
	my_title2 = title2[type2]
	my_name = name_figure[type1+ "_" + type2]
	plt.figure()
	for t in range(0,data.size(2)):
		plt.errorbar(range(1,e + 1),data[:,:,t].mean(0).numpy(),yerr = data[:,:,t].std(0).numpy()) # Plot the mean + std of Loss for the class label 
	plt.grid()
	plt.legend(legends, fontsize = 14)
	plt.title(my_title1 + my_title2, **title_font)
	plt.xlim(0.9, e+0.1)
	plt.ylabel(y_label,**label_font)
	plt.xlabel(x_label,labelpad = 20, **label_font)
	if(flag_save_figure):
		plt.savefig(folder_name + my_name + ".pdf")