# -*- coding: utf-8 -*-
"""
Author(s):
Sebastian Jaimungal
"""

import numpy as np
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt

class copula():

    def __init__(self, data):
        self.data = data
        self.F, self.U = self.__Extract_Empirical_Copula__()
        
    def Generate_Copula_KDE(self):
        
        self.u_grid, self.kde_grid, self.kde_data = self.__Copula_KDE__()


    def PlotCopula(self):

        # Winsorize the data
        tails = np.quantile(self.kde_data[~np.isnan(self.kde_data)], [0.005, 0.995])
        mask = (self.kde_data>tails[0]) & (self.kde_data<tails[1])
    
        qtl = np.quantile(self.kde_data[mask],[0.05, 0.25, 0.5, 0.75, 0.95])
    
        plt.contourf(self.u_grid[0], self.u_grid[1], self.kde_grid, levels=qtl, alpha=0.25)
        plt.scatter(self.U[0], self.U[1],c=self.kde_data, s=0.1,alpha=0.9)
        plt.clim(qtl[0],qtl[-1])
        plt.show()

    # obtain the empirical marginal distribution function and copula from the data
    def __Extract_Empirical_Copula__(self):
    
        # compute the ECDF and transform each dimension marginally to a uniform
        F = []
        U = []
        for X in self.data:
            F.append(ECDF(X))
            U.append(F[-1](X))
    
        return F, U

    def __Copula_KDE__( self ):
    
        # 2-d grid on which to evaluate the KDE
        u1, u2 = np.meshgrid(np.linspace(0.001,0.999,51), np.linspace(0.001, 0.999, 51))
        x1 = norm.ppf(u1)
        x2 = norm.ppf(u2)
    
        # number of data points
        d = len(self.U)
        N = len(self.U[0])
        nrm_fac = 1.0/float(N)
    
        # estimate bandwidth -- this may need tuning
        mask = np.zeros(len(self.U[0]), dtype=bool)
        X = [norm.ppf(self.U[i]) for i in range(d)]
        for i in range(d):
            mask = mask | np.isinf(X[i])
        
        h = [ 1.069*np.std(X[i][~mask], axis=0) * N**(-1/5) for i in range(d)]
    
        X = [ X[i][~mask] for i in range(d)]
    
        # stores KDE
        self.kde_grid = np.zeros(u1.shape)
    
        # one-dimensional Gaussian kernel transformed onto uniform interval
        f = lambda x, X, h :  norm.pdf((x - X)/h )/(norm.pdf(x)*h)
    
        # loop over two dimesions, parallelize over data
        print('generating copula on grid')
        kde_grid = np.zeros(x1.shape)
        for i in range(x1.shape[0]):
            for j  in range(u1.shape[1]):

                mask = (np.abs(X[0]-x1[i,j])/h[0] < 5) | (np.abs(X[1]-x2[i,j])/h[1] < 5)

                kde_grid[i,j] = nrm_fac* np.sum( f(x1[i,j], X[0][mask], h[0]) * f(x2[i,j], X[1][mask], h[1])  )

        print('generating copula on data')
        kde_data = np.zeros(len(self.U[0]))
        X_all = [norm.ppf(self.U[i]) for i in range(d)]
        for i in range(len(self.U[0])):
        
            mask = (np.abs(X[0]-X_all[0][i])/h[0] < 5) | (np.abs(X[1]-X_all[1][i])/h[1] < 5)

            kde_data[i] = nrm_fac* np.sum( f(X_all[0][i], X[0][mask], h[0]) * f(X_all[1][i], X[1][mask], h[1]))
        
        return np.array((u1,u2)), kde_grid, kde_data