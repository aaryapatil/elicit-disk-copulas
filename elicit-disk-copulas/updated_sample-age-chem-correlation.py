# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 16:29:04 2022

Author(s):
Sebastian Jaimungal
Aarya Patil
"""

from astropy.io import fits
from copula import copula
from conditional_correlation import conditional_correlation
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm

import matplotlib as mpl
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize']= 12
mpl.rcParams['ytick.labelsize']= 12

# Load astroNN data
select_stars = np.load("apogee_cross_matched_nn.npy")
# Load Leung et al. 2023 ages
select_stars_ages = np.load("ages_nn.npy")

data = [select_stars_ages, select_stars['FE_H']]

# Compute copula
model = copula(data)
# Copula space data with uniform margins
u1 = model.F[0](data[0])
u2 =  model.F[1](data[1])

plt.scatter(data[0], data[1], s=1, alpha=0.1, c=select_stars['galr'])
plt.show()

# Add gaussian margins
x1 = norm.ppf(u1)
x2 = norm.ppf(u2)
mask= ~(np.isinf(x1) | np.isinf(x2))
x1 = x1[mask]
x2 = x2[mask]

# Creat copula dataset
data = np.array([x1, x2, select_stars['galr'][mask],
                 select_stars['galz'][mask]])

# Test loss of neural network to fix iterations
cm = conditional_correlation(data)
cm.Estimate(n_iter=10_000, n_print=2_000,
            labels=['R', 'age', 'Fe/H'])
np.savetxt('loss', cm.loss)

# Split data for jackknifing
splits = 50
data = np.array(data)
data_split = np.split(data, splits, axis=1)

# Jackknifed correlation estimates
cm_list = []
for ind in np.arange(splits):
    data_jk = np.delete(data_split, ind, axis=0)
    data_jk = [data_jk[:, 0].flatten(), data_jk[:, 1].flatten(),
               data_jk[:, 2].flatten(), data_jk[:, 3].flatten()]
    cm = conditional_correlation(data_jk)
    # 3000 iterations chosen based on loss
    cm.Estimate(n_iter=3_000, n_print=1_000,
                labels=['R', 'age', 'Fe/H'])
    cm_list.append(cm)

# Compare jackknifed correlations on grid
N = 101
rho_all = np.zeros(shape=(9, splits, N))
z_values = np.linspace(0, 2, 9)
for ind, z in enumerate(z_values):
    gal = z*np.ones(shape=(N, 2))
    gal[:, 0] = np.linspace(0, 20, N)
    gal = torch.tensor(gal).to(cm.device)
    rho = []
    for cm in cm_list:
        rho.append(cm.rho(gal).squeeze().detach().cpu().numpy())
    rho_all[ind] = np.array(rho)

np.save('rho_all.npy', rho_all)

gal = np.ones(shape=(N, 2))
gal[:, 0] = np.linspace(0, 20, N)
z_values = np.linspace(-2, 2, 17)

fig, ax = plt.subplots(1, 1, figsize=(8, 5), sharex=True, sharey=True)
mean_rho = rho_all[0].mean(axis=0)
var_rho = (splits-1)*rho_all[0].var(axis=0)

ax.plot(gal[:, 0], mean_rho, color='black')
ax.fill_between(gal[:, 0], mean_rho_0 + np.sqrt(var_rho),
                mean_rho - np.sqrt(var_rho), alpha=0.25, color='grey')
ax.set_xlim([0, 20])
ax.set_ylim([-1, 1])
ax.grid()
ax.set_xticks(np.arange(0, 21, 2), fontsize=12)
ax.set_yticks(np.linspace(-0.8, 0.8, 5), fontsize=12)
ax.set_xlabel("Galactocentric Radius $R$ [kpc]", fontsize=18)
ax.set_ylabel(r"Correlation $\rho (\tau, M \mid R, z=0 \; \mathrm{kpc})$", fontsize=18)
plt.savefig('age-metallicity-z=0.pdf',format='pdf')

#%%
# Repeat above procedure for high-alpha stars
select_stars = np.load("high_alpha_apogee.npy")
select_stars_ages = np.load("high_alpha_ages.npy")

data = [select_stars_ages, select_stars['FE_H']]
model = copula(data)
u1 = model.F[0](data[0])
u2 =  model.F[1](data[1])

plt.scatter(u1, u2, s=1, alpha=0.1, c=select_stars['galr'])
plt.show()

x1 = norm.ppf(u1)
x2 = norm.ppf(u2)
mask= ~(np.isinf(x1) | np.isinf(x2))
x1 = x1[mask]
x2 = x2[mask]

plt.scatter(x1, x2, s=1, alpha=0.1, c=select_stars['galr'][mask])
plt.show()

data = [x1, x2, select_stars['galr'][mask], select_stars['galz'][mask]]
data = np.array(data)

data_split = np.split(data, splits, axis=1)

cm_list = []
for ind in np.arange(splits):
    data_jk = np.delete(data_split, ind, axis=0)
    data_jk = [data_jk[:, 0].flatten(), data_jk[:, 1].flatten(),
               data_jk[:, 2].flatten(), data_jk[:, 3].flatten()]
    cm = conditional_correlation(data_jk)
    cm.Estimate(n_iter=3_000, n_print=1_000, labels=['R', 'age', 'Fe/H'])
    cm_list.append(cm)

rho_high = np.zeros(shape=(9, splits, N))
z_values = np.linspace(0, 2, 9)
for ind, z in enumerate(z_values):
    gal = z*np.ones(shape=(N, 2))
    gal[:, 0] = np.linspace(0, 20, N)
    gal = torch.tensor(gal).to(cm.device)
    rho = []
    for cm in cm_list:
        rho.append(cm.rho(gal).squeeze().detach().cpu().numpy())
    rho_high[ind] = np.array(rho)

np.save('rho_high.npy', rho_high)

#%%
# Repeat above procedure for low-alpha stars
select_stars = np.load("low_alpha_apogee.npy")
select_stars_ages = np.load("low_alpha_ages.npy")

data = [select_stars_ages, select_stars['FE_H']]
model = copula(data)
u1 = model.F[0](data[0])
u2 =  model.F[1](data[1])

plt.scatter(u1, u2, s=1, alpha=0.1, c=select_stars['galr'])
plt.show()

x1 = norm.ppf(u1)
x2 = norm.ppf(u2)
mask= ~(np.isinf(x1) | np.isinf(x2))
x1 = x1[mask]
x2 = x2[mask]

plt.scatter(x1, x2, s=1, alpha=0.1, c=select_stars['galr'][mask])
plt.show()

data = [x1, x2, select_stars['galr'][mask], select_stars['galz'][mask]]
data = np.array(data)

data_split = np.split(data, splits, axis=1)

cm_list = []
for ind in np.arange(splits):
    data_jk = np.delete(data_split, ind, axis=0)
    data_jk = [data_jk[:, 0].flatten(), data_jk[:, 1].flatten(),
               data_jk[:, 2].flatten(), data_jk[:, 3].flatten()]
    cm = conditional_correlation(data_jk)
    cm.Estimate(n_iter=3_000, n_print=1_000, labels=['R', 'age', 'Fe/H'])
    cm_list.append(cm)

rho_low = np.zeros(shape=(9, splits, N))
for ind, z in enumerate(z_values):
    gal = z*np.ones(shape=(N, 2))
    gal[:, 0] = np.linspace(0, 20, N)
    gal = torch.tensor(gal).to(cm.device)
    rho = []
    for cm in cm_list:
        rho.append(cm.rho(gal).squeeze().detach().cpu().numpy())
    rho_low[ind] = np.array(rho)

np.save('rho_low.npy', rho_low)

#%%
fig, ax = plt.subplots(3, 5, figsize=(34, 15), sharey=True, sharex=True)

splits_all = 50
splits_high = 50
splits_low = 50

for ind, i in enumerate(range(0, 9, 2)):
    mean_rho_all = np.nanmean(rho_all[i], axis=0)
    var_rho_all = (splits_all - 1)*np.nanvar(rho_all[i], axis=0)
    
    mean_rho_high = np.nanmean(rho_high[i], axis=0)
    var_rho_high = (splits_high - 1)*np.nanvar(rho_high[i], axis=0)
    
    mean_rho_low = np.nanmean(rho_low[i], axis=0)
    var_rho_low = (splits_low - 1)*np.nanvar(rho_low[i], axis=0)

    ax[0][ind].plot(gal[:, 0], mean_rho_all, label='all',
                    color='black')
    ax[0][ind].fill_between(gal[:, 0], mean_rho_all + np.sqrt(var_rho_all),
                            mean_rho_all - np.sqrt(var_rho_all), alpha=0.15,
                            color='grey')
    
    ax[1][ind].plot(gal[:, 0], mean_rho_high, label=r'high-$\alpha$')
    ax[1][ind].fill_between(gal[:, 0], mean_rho_high + np.sqrt(var_rho_high),
                            mean_rho_high - np.sqrt(var_rho_high), alpha=0.15)
    
    ax[2][ind].plot(gal[:, 0], mean_rho_low, label=r'low-$\alpha$',
                    color='tab:orange')
    ax[2][ind].fill_between(gal[:, 0], mean_rho_low + np.sqrt(var_rho_low),
                            mean_rho_low - np.sqrt(var_rho_low), alpha=0.15,
                            color='tab:orange')
    
    ax[0][ind].set_ylim([-1, 1])
    ax[1][ind].set_ylim([-1, 1])
    ax[2][ind].set_ylim([-1, 1])
    
    ax[0][ind].grid()
    ax[1][ind].grid()
    ax[2][ind].grid()
    
    ax[0][ind].set_yticks(np.linspace(-0.8, 0.8, 5), fontsize=16)
    ax[1][ind].set_yticks(np.linspace(-0.8, 0.8, 5), fontsize=16)
    ax[2][ind].set_yticks(np.linspace(-0.8, 0.8, 5), fontsize=16)
    
    ax[0][ind].set_xticks(np.arange(0, 25, 5), fontsize=16)
    ax[1][ind].set_xticks(np.arange(0, 25, 5), fontsize=16)
    ax[2][ind].set_xticks(np.arange(0, 25, 5), fontsize=16)

ax[0][0].legend(loc='upper center', fontsize=18)
ax[1][0].legend(loc='upper center', fontsize=18)
ax[2][0].legend(loc='upper center', fontsize=18)
fig.text(0.43, 0.08, "Galactocentric radius $R$ [kpc]", fontsize=28)
ax[1][0].set_ylabel(r"Correlation $\rho(\tau, \mathrm{M} \mid R, z)$", fontsize=28)

ax[0][0].set_title(r'$|z|=0$ kpc', fontsize=20)
ax[0][1].set_title(r'$|z|=0.5$ kpc', fontsize=20)
ax[0][2].set_title(r'$|z|=1$ kpc', fontsize=20)
ax[0][3].set_title(r'$|z|=1.5$ kpc', fontsize=20)
ax[0][4].set_title(r'$|z|=2$ kpc', fontsize=20)
plt.savefig('age-metallicity-z=all.pdf',format='pdf')
