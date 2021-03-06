#--------------------------------------------------------------------#
#								Main
#--------------------------------------------------------------------#

######################################################################
import torch
import my_plot	as mp # Custom file: creat and save plots (errorbars)
import project2 as p2
import project2_withAutoGrad as p2_pytorch
#import cProfile, pstats,io

if __name__ == "__main__":
#--------------------------------------------------------------------#
	# Project 2 : Mini deep-learning framework
#--------------------------------------------------------------------#
   # Parameters:
   nb_train_samples = 1000
   e = 20                      # Number of epochs
   mini_batch_size = 100
   eta =  0.1                   # fixed learning rate
   N = 10                      # Number of fold
   
   # Array for plots
   p2_train_error_rate = torch.zeros(N,e,2)
   p2_test_error_rate  = torch.zeros(N,e,2)
   p2_loss = torch.zeros(N,e,2) # Losses (class and target)
   
#   pr = cProfile.Profile()

   for n in range(0,N):
       print(n)
#       pr.enable()

       train_input, train_target = p2.generate_disc_set(nb_train_samples)
       test_input, test_target = p2.generate_disc_set(nb_train_samples)
       mean, std = train_input.mean(), train_input.std()
    
    	# normalize samples
       train_input.sub_(mean).div_(std)
       test_input.sub_(mean).div_(std)

    	# instance fully connected layers, relu and loss
       lin1 = p2.Linear(2, 25)
       lin2 = p2.Linear(25, 25)
       lin3 = p2.Linear(25, 25)
       lin4 = p2.Linear(25, 25)
       lin5 = p2.Linear(25, 2)
       relu = p2.Relu()
       loss = p2.LossMSE()
    
    	# model is a list of ordered modules
       model = [lin1, relu, lin2, relu, lin3, relu, lin4, relu, lin5]

    	# instance sequential class
       seq = p2.Sequential(model)
       acc_loss_list, per_train_error_list, per_test_error_list,output = p2.run(seq, train_input, test_input,train_target, test_target ,loss,eta,e, mini_batch_size)
       if(n == 0):
           mp.plot_data(test_input, test_target,"data")
           mp.plot_data(test_input, output[:,:,0],"data_init")
           mp.plot_data(test_input, output[:,:,1],"data_out")
           
       for i,e_ in enumerate(acc_loss_list):
           p2_loss[n,i,0] = e_.item()
           p2_train_error_rate[n,i,0] = per_train_error_list[i]
           p2_test_error_rate[n,i,0] = per_test_error_list[i]
           
#        pr.disable()
#       s = io.StringIO()
#       ps = pstats.Stats(pr, stream=s)
#       ps.print_stats()
#       print(s.getvalue())
#

       model1 = p2_pytorch.create_model()
      
       acc_loss_list, per_train_error_list, per_test_error_list = p2_pytorch.train_model(model1, train_input, train_target, e, mini_batch_size, eta, test_input, test_target)   
       for i,e_ in enumerate(acc_loss_list):
           p2_loss[n,i,1] = e_
           p2_train_error_rate[n,i,1] = per_train_error_list[i]
           p2_test_error_rate[n,i,1] = per_test_error_list[i]

   mp.plot(e,p2_train_error_rate, "p2","train")
   mp.plot(e,p2_test_error_rate, "p2","test")
   mp.plot(e,p2_loss, "p2","loss")

   # Final performances
   print('Custom Framework\n')
   print('[Error_rate] : \nTrain : Mean : {:.02f}% : STD : {:.02f}% : \nTest : Mean : {:.02f}% : STD : {:.02f}%\n'.format(p2_train_error_rate[:,-1,0].mean()*100,p2_train_error_rate[:,-1,0].std()*100,p2_test_error_rate[:,-1,0].mean()*100,p2_test_error_rate[:,-1,0].std()*100))
   print('Pytorch Framework\n')
   print('[Error_rate] : \nTrain : Mean : {:.02f}% : STD : {:.02f}% : \nTest : Mean : {:.02f}% : STD : {:.02f}%\n'
              .format(p2_train_error_rate[:,-1,1].mean()*100,p2_train_error_rate[:,-1,1].std()*100,p2_test_error_rate[:,-1,1].mean()*100,p2_test_error_rate[:,-1,1].std()*100))
       
