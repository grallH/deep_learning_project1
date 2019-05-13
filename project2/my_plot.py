#!/usr/bin/env python

######################################################################
import matplotlib.pyplot as plt

######################################################################
# Plot param
flag_save_figure = False
folder_name = "picture/"
xLabel = {"p2_loss" : "Different epochs" , "p2_train" : "Different epochs", "p2_test" : "Different epochs"}
yLabel = {"p2_loss" : "Loss " , "p2_train" : "Error rate","p2_test" : "Error rate"}
legend = {"p2" : "Custom framework"}
title_font = {'fontname':'Arial', 'size':'20', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'}
label_font = {'fontname':'Arial', 'size':'14', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'}
name_figure = {"p2_train" : "p2_train", "p2_test" : "p2_test", "p2_loss" : "p2_loss"}

def plot(e,data,type1, type2):
	x_label = xLabel[type1+ "_" + type2]
	y_label = yLabel[type1+ "_" + type2]
	legends = legend[type1]
	#my_title1 = title1[type1]
	#my_title2 = title2[type2]
	my_name = name_figure[type1+ "_" + type2]
	plt.figure()
	for t in range(0,data.size(2)):
		plt.errorbar(range(1,e + 1),data[:,:,t].mean(0).numpy(),yerr = data[:,:,t].std(0).numpy()) # Plot the mean + std of Loss for the class label 
	plt.grid()
	plt.legend(legends, fontsize = 14)
	plt.title("test", **title_font)
	plt.xlim(0.9, e+0.1)
	plt.ylabel(y_label,**label_font)
	plt.xlabel(x_label,labelpad = 20, **label_font)
	if(flag_save_figure):
		plt.savefig(folder_name + my_name + ".pdf")