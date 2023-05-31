import numpy as np
import matplotlib.pyplot as plt 

import torch
from torch._C import _multiprocessing_init
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

class Net(nn.Module):
    
    def __init__(self, d, nNodes, nHidden, device ):
        super(Net, self).__init__()

        self.prop_in_to_h = nn.Linear( d, nNodes).to(device)
        self.prop_h_to_h = nn.ModuleList([nn.Linear(nNodes, nNodes).to(device) for i in range(nHidden-1)])
        self.prop_h_to_out = nn.Linear(nNodes, 1).to(device)
        
        self.silu = nn.SiLU()
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        
        # input into  hidden layer
        h = self.silu(self.prop_in_to_h(x))
        
        # hidden to hidden 
        for i in range(len(self.prop_h_to_h)):
            h = self.silu(self.prop_h_to_h[i](h))
        
        # hidden layer to output layer
        y = self.prop_h_to_out(h)
                
        return y       
    
class conditional_median():
    
    
    def __init__(self, data, nNodes=20, nHidden=5, lr=0.005):    
        
        if torch.cuda.device_count() > 0:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        print(self.device)

        self.data = torch.zeros( (len(data[0]), 3)).float().to(self.device)
        for i in range(len(data)):
            self.data[:,i] = torch.tensor(np.array(data[i], float)).to(self.device)

        self.d = 2
        
        # the ANN for the conditional median mu(x,y) : = Med[ Z | X=x, Y=y  ]
        self.mu = Net(self.d, nNodes, nHidden, self.device)
        
        self.optimizer = optim.Adam(self.mu.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=500, gamma=0.95)
        
        self.loss_hist = []
        
        
    def Loss(self, X, Z):
        
        alpha = 0.5
        mu = self.mu(X)
        score = (1*(Z.reshape(-1,1) <= mu) - alpha)*(mu-Z.reshape(-1,1))
        
        # return torch.mean( (self.mu(X) - Z)**2  )
        
        return torch.mean( score )
        

    
    def Grab_Mini_Batch(self, mini_batch_size=256):

        idx = torch.randint(self.data.shape[0], [mini_batch_size])

        return self.data[idx,:2], self.data[idx,2]

    def Run_Epoch(self, mini_batch_size=256):
        
        
        # compute score for each element in the batch
        self.optimizer.zero_grad()

        # simulate minibatch of samples from distribution
        x, z = self.Grab_Mini_Batch(mini_batch_size)
        
        loss = self.Loss(x,z)
            
        loss.backward()
        
        self.optimizer.step()
        self.scheduler.step()
        
        self.loss_hist.append(loss.item())   
        
    def Estimate(self, n_iter=1_000, n_print=100, mini_batch_size=256):
        
        for epoch in tqdm(range(n_iter)):
            
            self.Run_Epoch(mini_batch_size=mini_batch_size)
            
            if np.mod(epoch+1, n_print) == 0:
                
                fig = self.Plot_mu('conditional_mean_' + str(int((epoch+1)/n_print)) )
                # print('Epoch:', epoch,'LR:', '{0:0.5f}'.format(self.scheduler.get_last_lr()[0]))
                
        fig = self.Plot_mu()
        
        return fig        
        
    def Plot_mu(self, filename=None):
        
        Nx = 101
        x = torch.zeros((Nx, self.d )).to(self.device)
        x[:,0] = torch.linspace(0,10,Nx)
        x[:,1] = torch.linspace(-0.5,0.6,Nx)
        
        x1, x2 = torch.meshgrid( x[:, 0], x[:, 1])
        
        xm = torch.cat((x1.unsqueeze(2),x2.unsqueeze(2)),axis=2)
        
        mu = self.mu( xm ).squeeze().detach()

        fig = plt.figure(figsize=(10,4))
        ax = fig.add_subplot(1,3,1)

        ax.plot(self.loss_hist)
        ax.set_yscale('log')

        ax= fig.add_subplot(1, 3, 2, projection='3d')
        ax.plot_surface(x1.cpu().numpy(), x2.cpu().numpy(), mu.cpu().numpy(), linewidth=0, antialiased=False, cmap='winter')
        
        ax.set_xlabel(r"$Age$")
        ax.set_ylabel(r"$Fe/H$")        
        
        ax = fig.add_subplot(1,3,3)
        
        Nx = 101
        x = torch.linspace(-0.5,0.6,Nx).to(self.device)
        for i in range(0,10):
                
            x1, x2 = torch.meshgrid( (i+1)*torch.ones(1).to(self.device), x)
            xm = torch.cat((x1.unsqueeze(2),x2.unsqueeze(2)),axis=2)
            mu = self.mu( xm ).squeeze().detach()
            
            ax.plot(x.cpu().numpy(), mu.cpu().numpy(), label=str(i))
        
        ax.scatter(self.data[:,1].cpu().numpy(),self.data[:,2].cpu().numpy(),s=1,alpha=0.2,c=self.data[:,0].cpu().numpy())        
        
        ax.set_ylim(-0.1, 0.3)
        ax.set_xlim(-0.5, 0.6)
        ax.set_xlabel(r"$Fe/H$")
        ax.set_ylabel(r"$Alpha/Fe$") 
        ax.legend(fontsize=8)
        
        plt.tight_layout(pad=1.5)

        if filename is not None:
            plt.savefig(filename)
            
        plt.show()
        
