import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from astropy.io import fits
from apogee.tools import bitmask as bm
from apogee.tools.path import change_dr
import apogee.tools.read as apread

from copula import copula 

mpl.rcParams["axes.labelsize"] = 20
mpl.rcParams['xtick.labelsize']= 15
mpl.rcParams['ytick.labelsize']= 15

# APOGEE DR17
change_dr('17')
allStar = apread.allStar(raw=True)
# astroNN DR17
astroNN_allStar = fits.open("data/apogee_astroNN-DR17.fits")[1].data
# Leung et al. 2023
allStar_ages = pd.read_csv("data/nn_latent_age_dr17.csv.gz")

# Main survey targets
select_stars = allStar[allStar['EXTRATARG'] == 0]
select_stars_astroNN = astroNN_allStar[allStar['EXTRATARG'] == 0]
select_stars_ages = allStar_ages[allStar['EXTRATARG'] == 0]

# Remove all flagged stars as suggested in Leung et al. 2023
star_ind = (select_stars['ASPCAPFLAG'] == 0)
select_stars = select_stars[star_ind]
select_stars_astroNN = select_stars_astroNN[star_ind]
select_stars_ages = select_stars_ages[star_ind]

star_ind = (select_stars['STARFLAG'] == 0)
select_stars = select_stars[star_ind]
select_stars_astroNN = select_stars_astroNN[star_ind]
select_stars_ages = select_stars_ages[star_ind]

# Remove cluster members
cond_cluster = (select_stars['MEMBERFLAG']==0)
select_stars = select_stars[np.where(cond_cluster)]
select_stars_astroNN = select_stars_astroNN[np.where(cond_cluster)]
select_stars_ages = select_stars_ages[cond_cluster]

# Distance < 20 % errors
perc_dist_err = select_stars_astroNN['DIST_ERROR']/select_stars_astroNN['DIST']*100
select_stars_astroNN = select_stars_astroNN[np.where(perc_dist_err < 20)]
select_stars_ages = select_stars_ages[perc_dist_err < 20]

# log g cut to > 2.55 as suggested in Leung et al. 2023
cond_logg = (select_stars_astroNN['LOGG'] > 2.55) & (select_stars_astroNN['LOGG'] < 3.8)
select_stars_astroNN = select_stars_astroNN[cond_logg]
select_stars_ages = select_stars_ages[cond_logg]

# Temperature cuts
cond_teff_4000 = (select_stars_astroNN['TEFF'] > 4000) & (select_stars_astroNN['TEFF'] < 5500)

astroNN_4000 = select_stars_astroNN[cond_teff_4000]
ages_4000 = select_stars_ages[cond_teff_4000]

# Age cuts
cond = (astroNN_4000['age'] < 13)
astroNN_4000 = astroNN_4000[cond]
ages_4000 = ages_4000[cond]

# R, z cuts for the disk provide the updated sample
select_stars = astroNN_4000[astroNN_4000['galr'] < 20]
select_stars_ages = ages_4000[astroNN_4000['galr'] < 20]

select_stars_final = select_stars[np.abs(select_stars['galz']) < 2]
select_stars_ages_final = select_stars_ages[np.abs(select_stars['galz']) < 2]

# Remove stars without latent space ages
ages_final = select_stars_ages_final[~np.isnan(select_stars_ages_final['Age'])]
apogee_final = select_stars_final[~np.isnan(select_stars_ages_final['Age'])]

# Age error cuts to < 40% as suggested in Leung et al. 2023
err_ind = (ages_final['Age_Error']/ages_final['Age']*100 < 40)
ages_final = ages_final[err_ind]
apogee_final = apogee_final[err_ind]

# Save data
np.save('ages_nn', ages_final['Age'])
np.save('apogee_cross_matched_nn', apogee_final)

#-------------------------------------------------------------------------------
# Plot the updated sample ages (Leung et al. 2023) and compare with astroNN
fig, ax = plt.subplots(1, 2, figsize=(18, 5), sharey=True, sharex=True)

im = ax[0].scatter(apogee_final['Age'], apogee_final['FE_H'],
                   c=(apogee_final['MG_H'] - apogee_final['FE_H']), s=0.3,
                   cmap=plt.cm.plasma,
                   vmin=np.min(apogee_final['MG_H'] - apogee_final['FE_H']),
                   vmax=np.max(apogee_final['MG_H'] - apogee_final['FE_H']))
ax[0].set_xlim(0, 14)
ax[0].set_ylim(-0.9, 0.6)
ax[0].set_yticks(np.linspace(-0.9, 0.6, 6))
ax[0].set_ylabel(r'$[\mathrm{Fe/H}]$', fontsize=20)
ax[0].set_xlabel('astroNN Age (Gyr)', fontsize=20)
plt.subplots_adjust(hspace=0)

im = ax[1].scatter(ages_final['Age'], apogee_final['FE_H'],
                   c=(apogee_final['MG_H'] - apogee_final['FE_H']), s=0.3,
                   cmap=plt.cm.plasma,
                   vmin=np.min(apogee_final['MG_H'] - apogee_final['FE_H']),
                   vmax=np.max(apogee_final['MG_H'] - apogee_final['FE_H']))
ax[1].set_xlim(0, 14)
ax[1].set_ylim(-0.9, 0.6)
ax[1].set_yticks(np.linspace(-0.9, 0.6, 6))
ax[1].set_xlabel('encoder-decoder Age (Gyr)', fontsize=20)
plt.subplots_adjust(hspace=0, wspace=0.1)
# Add a colorbar
cbar = fig.colorbar(im, ax=ax[:], orientation='vertical')
cbar.ax.get_yaxis().labelpad = 25
cbar.ax.set_ylabel(r'$[\mathrm{Mg/Fe}]$', rotation=270, fontsize=18)
ax[0].grid()
ax[1].grid()
plt.savefig('leung_data.pdf', format='pdf', bbox_inches = 'tight')

#-------------------------------------------------------------------------------
# Obtain the copula space of age-metallicity
# Compute copula on a grid
data = [ages_final['Age'], apogee_final['FE_H']]
model = copula(data)
u_grid, kde_grid = model.Generate_Copula_KDE()

# Plot copula and show addition of Gaussian margins
fig, ax = plt.subplots(1, 3, figsize=(20, 8))

plt.subplots_adjust(hspace=0, wspace=0.4)
im = ax[0].hist2d(data[0], data[1], bins=100,
                  norm=mpl.colors.LogNorm(), cmap='magma')[3]
ax[0].set_xlabel(r'Age ($\tau$)')
ax[0].set_ylabel(r'$[\mathrm{Fe/H}] \; (M)$')
plt.colorbar(im, ax=ax[0], orientation='horizontal', label='Number Density')

im = ax[1].scatter(u_grid[0], u_grid[1], s=60, c=kde_grid, cmap='magma')
ax[1].set_xlabel(r'$\tau_U \sim U$')
ax[1].set_ylabel(r'$\mathrm{M}_U \sim U$')
ax[1].set_xlim([0, 1])
ax[1].set_ylim([0, 1])
plt.colorbar(im, ax=ax[1], orientation='horizontal', label='Copula Density')

# Gaussian margins
G = []
G.append(norm.ppf(model.U[0]))
G.append(norm.ppf(model.U[1]))

im = ax[2].hist2d(G[0], G[1], bins=100, range=[[-4, 4], [-4, 4]],
                  norm=mpl.colors.LogNorm(), cmap='magma')[3]
ax[2].set_xlabel(r'$\tau_\mathcal{N} \sim \mathcal{N}$')
ax[2].set_ylabel(r'$\mathrm{M}_\mathcal{N} \sim \mathcal{N}$')
plt.colorbar(im, ax=ax[2], orientation='horizontal', label='Number Density')

fig.text(0.345, 0.6, r'$\rightarrow$', fontsize=30)
fig.text(0.63, 0.6, r'$\rightarrow$', fontsize=30)

#-------------------------------------------------------------------------------
# Split updated sample into alpha sequences
# Load data space split
split_line_coeff = np.load('dat_space_split_fit')
split_line = np.poly1d(split_line_coeff)

# Divide the data using data space split
data = [apogee_final['MG_H'] - apogee_final['FE_H'], apogee_final['FE_H']]
above_inds = (data[0] > split_line(data[1])
below_inds = (data[0] <= split_line(data[1])

np.save("low_alpha_apogee.npy", apogee_final[below_inds])
np.save("low_alpha_ages.npy", ages_final['Age'][below_inds])

np.save("high_alpha_apogee.npy", apogee_final[above_inds])
np.save("high_alpha_ages.npy", ages_final['Age'][above_inds])
