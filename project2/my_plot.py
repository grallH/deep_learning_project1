#!/usr/bin/env python

######################################################################
import matplotlib.pyplot as plt
######################################################################
# Plot param
flag_save_figure = False
folder_name = "Figures/"
xLabel = {"p2_loss" : "Different epochs" , "p2_train" : "Different epochs", "p2_test" : "Different epochs","data" : "x1"}
yLabel = {"p2_loss" : "Loss " , "p2_train" : "Error rate","p2_test" : "Error rate","data" : "x2"}
legend = {"p2" : ("Custom framework","PyTorch framework"), "data": ("Class 0","Class 1")}
title = {"test" : "Error rate : Test Vs Epochs", "train" : "Error rate : Train Vs Epochs", "loss": "Loss : Train Vs Epochs","data" : "Data Set"}
title_font = {'fontname':'Arial', 'size':'20', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'}
label_font = {'fontname':'Arial', 'size':'14', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'}
name_figure = {"p2_train" : "p2_train", "p2_test" : "p2_test", "p2_loss" : "p2_loss", "data" : "data_set" }

def plot(e,data,type1, type2):
	x_label = xLabel[type1+ "_" + type2]
	y_label = yLabel[type1+ "_" + type2]
	legends = legend[type1]
	my_title = title[type2]
	my_name = name_figure[type1+ "_" + type2]
	plt.figure()
	for t in range(0,data.size(2)):
		m = data[:,:,t].mean(0).numpy()
		sigma = data[:,:,t].std(0).numpy()
		plt.fill_between(range(1,e+1), m-sigma, m+sigma,alpha=0.3)
		plt.plot(range(1,e + 1),data[:,:,t].mean(0).numpy())
		#plt.errorbar(range(1,e + 1),data[:,:,t].mean(0).numpy(),yerr = data[:,:,t].std(0).numpy()) # Plot the mean + std of Loss for the class label 
	plt.grid()
	plt.legend(legends, fontsize = 14)
	plt.title(my_title, **title_font)
	plt.xlim(0.9, e+0.1)
	plt.ylabel(y_label,**label_font)
	plt.xlabel(x_label,labelpad = 20, **label_font)
	if(flag_save_figure):
		plt.savefig(folder_name + my_name + ".pdf")

def plot_data(x,label,type1):
    x_label = xLabel[type1]
    y_label = yLabel[type1]
    legends = legend[type1]
    my_title = title[type1]
    my_name = name_figure[type1]
    plt.scatter(x[label.argmax(0),0],x[label.argmax(0),1])
    plt.scatter(x[label.argmin(0),0],x[label.argmin(0),1])
    plt.grid()
    plt.legend(legends, fontsize = 14)
    plt.title(my_title, **title_font)
    plt.ylabel(y_label,**label_font)
    plt.xlabel(x_label,labelpad = 20, **label_font)
    if(flag_save_figure):
        plt.savefig(folder_name + my_name + ".pdf")
    