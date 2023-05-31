# -*- coding: utf-8 -*-
"""
Author(s):
Sebastian Jaimungal
Aarya Patil (modifications only)
"""

import numpy as np
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm


class Net(nn.Module):
    
    def __init__(self, d_in, nNodes, nHidden, device, out_activation=None ):
        super(Net, self).__init__()

        self.prop_in_to_h = nn.Linear(d_in, nNodes).to(device)
        self.prop_h_to_h = nn.ModuleList([nn.Linear(nNodes, nNodes).to(device) for i in range(nHidden-1)])
        self.prop_h_to_out = nn.Linear(nNodes, 1).to(device)
        
        self.silu = nn.SiLU()
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()
        
        self.out_activation = out_activation
        
    def forward(self, x):
        
        # input into  hidden layer
        h = self.silu(self.prop_in_to_h(x.float()))
        
        #h = self.dropout(h)
        
        # hidden to hidden 
        for i in range(len(self.prop_h_to_h)):
            h = self.silu(self.prop_h_to_h[i](h))
        
        # hidden layer to output layer
        y = self.prop_h_to_out(h)
        
        if self.out_activation == 'softplus':
            y = self.softplus(y)
        elif self.out_activation == 'tanh':
            y = torch.tanh(y)
                
        return y       
    
class conditional_correlation():
    
    
    def __init__(self, data, nNodes=20, nHidden=5, lr=0.005):    
        
        if torch.cuda.device_count() > 0:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        self.data = torch.zeros( (len(data[0]), 4)).float().to(self.device)
        for i in range(len(data)):
            self.data[:,i] = torch.tensor(np.array(data[i], float)).to(self.device)
        
        # Galactocentric radius R and distance to midplane z
        self.d = 2
        
        # the ANN for the conditional expectation E[ Y=y | Z=z  ]
        self.mu_X = Net(self.d, nNodes, nHidden, self.device)
        self.mu_Y = Net(self.d, nNodes, nHidden, self.device)
        self.rho = Net(self.d, nNodes, nHidden, self.device, 'tanh')
        self.Var_X = Net(self.d, nNodes, nHidden, self.device, 'softplus')
        self.Var_Y = Net(self.d, nNodes, nHidden, self.device, 'softplus')
        
        self.optimizer = {'X' : optim.AdamW(self.mu_X.parameters(), lr=lr),
                          'Y' : optim.AdamW(self.mu_Y.parameters(), lr=lr),
                          'rho' : optim.AdamW(self.rho.parameters(), lr=lr),
                          'Var_X' : optim.AdamW(self.Var_X.parameters(), lr=lr),
                          'Var_Y' : optim.AdamW(self.Var_Y.parameters(), lr=lr)}
        
        self.scheduler = {}
        for key, optimizer in self.optimizer.items():
            self.scheduler[key] = StepLR(optimizer, step_size=500, gamma=0.95)
        
        self.loss = []
        self.lr_update = []
        
    def Loss(self, x1, x2, x3, x4, x5, x, y):
        # mean and variance of U_x
        score = (x1**2 - 2.0*x2 - 2.0*x1*x + x**2)/x2**2
        # mean and variance of U_y
        score += (x3**2 - 2.0*x4 - 2.0*x3*y + y**2)/x4**2
        # mean of U_x,y
        score += (x5-x*y)**2
        
        return torch.mean(score)        
        
    def Grab_Mini_Batch(self, mini_batch_size=256):

        idx = torch.randint(self.data.shape[0], [mini_batch_size])

        return self.data[idx,0].reshape(-1,1), self.data[idx,1].reshape(-1,1), self.data[idx,2:]

    def Run_Epoch(self, epoch, mini_batch_size=256):

        # compute score for each element in the batch
        for key, optimizer in self.optimizer.items():
            optimizer.zero_grad()

        # simulate minibatch of samples from distribution
        x, y, z = self.Grab_Mini_Batch(mini_batch_size)
        z[:, 1] = np.abs(z[:, 1])

        Var_X = self.Var_X(z)
        mu_X = self.mu_X(z)
        mu_Y = self.mu_Y(z)
        Var_Y = self.Var_Y(z)
        mu_XY = torch.sqrt(Var_X*Var_Y)*self.rho(z) + mu_X*mu_Y        
        
        loss = self.Loss(mu_X, Var_X, mu_Y, Var_Y, mu_XY, x, y)
        loss.backward()
        
        for key, optimizer in self.optimizer.items():
            optimizer.step()
            self.scheduler[key].step()
        
        self.loss.append(loss.item())   
        
        
    def Estimate(self, n_iter=1_000, n_print=100, mini_batch_size=256, prefix = "", labels=None):
        for epoch in tqdm(range(n_iter)):
            
            self.Run_Epoch(epoch, mini_batch_size=mini_batch_size)
            
            self.lr_update.append(self.scheduler['rho'].get_last_lr()[0])
            
            if np.mod(epoch+1, n_print) == 0:
                
                fig = self.Plot(prefix + str(int((epoch+1)/n_print)), labels=labels)
                for key, optimizer in self.optimizer.items():
                    print('Epoch:', epoch,'LR:', '{0:0.5f}'.format(self.scheduler[key].get_last_lr()[0]))
                
        fig = self.Plot(labels=labels)
        
        return fig        
        
    def Plot(self, filename=None, labels=None):
        
        N = 101
        gal = np.zeros(shape=(N, 2))
        gal[:, 0] = np.linspace(0, 20, N)
        gal = torch.tensor(gal).to(self.device)
        
        rho = self.rho( gal ).squeeze().detach().cpu().numpy()
        
        gal = gal.cpu().numpy()
        
        plt.plot(gal[:, 0], rho)
        if labels is not None:
            plt.xlabel(f"{labels[0]}")
            plt.ylabel(f"Correlation [{labels[1]}, {labels[2]}] | {labels[0]}")        
        
        if filename is not None:
            plt.savefig(filename)        

        plt.show()

