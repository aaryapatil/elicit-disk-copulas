import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats

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

# Main survey targets
select_stars = allStar[allStar['EXTRATARG'] == 0]
select_stars_astroNN = astroNN_allStar[allStar['EXTRATARG'] == 0]

# Remove star bad
star_bad_ind = bm.bit_set(23, select_stars['ASPCAPFLAG'])
select_stars = select_stars[~star_bad_ind]
select_stars_astroNN = select_stars_astroNN[~star_bad_ind]

# Remove S/N < 50
sn_bad_ind = bm.bit_set(27, select_stars['ASPCAPFLAG'])
select_stars = select_stars[~sn_bad_ind]
select_stars_astroNN = select_stars_astroNN[~sn_bad_ind]

# Remove bad pixels
pix_bad_ind = bm.bit_set(0, select_stars['STARFLAG'])
select_stars = select_stars[~pix_bad_ind]
select_stars_astroNN = select_stars_astroNN[~pix_bad_ind]

# Remove very bright neighbours
neigh_bad_ind = bm.bit_set(3, select_stars['STARFLAG'])
select_stars = select_stars[~neigh_bad_ind]
select_stars_astroNN = select_stars_astroNN[~neigh_bad_ind]

# Remove cluster members
cluster_mem = np.where(select_stars['MEMBERFLAG']==0)
select_stars = select_stars[cluster_mem]
select_stars_astroNN = select_stars_astroNN[cluster_mem]

# Distance < 20 % errors
perc_dist_err = select_stars_astroNN['DIST_ERROR']/select_stars_astroNN['DIST']*100
select_stars_astroNN = select_stars_astroNN[np.where(perc_dist_err < 20)]
select_stars = select_stars[np.where(perc_dist_err < 20)]

# log g cuts
cond_logg = (select_stars_astroNN['LOGG'] > 1) & (select_stars_astroNN['LOGG'] < 3.8)
select_stars_astroNN = select_stars_astroNN[cond_logg]
select_stars = select_stars[cond_logg]

# Temperature cuts
cond_teff_4000 = (select_stars_astroNN['TEFF'] > 4000) & (select_stars_astroNN['TEFF'] < 5500)
astroNN_4000 = select_stars_astroNN[cond_teff_4000]
apokasc_4000 = select_stars[cond_teff_4000]

# Age cuts
cond = (astroNN_4000['age'] < 13)
astroNN_4000 = astroNN_4000[cond]
apokasc_4000 = apokasc_4000[cond]

# R, z cuts for the disk provide the original sample
select_stars = astroNN_4000[astroNN_4000['galr'] < 20]
select_stars = select_stars[np.abs(select_stars['galz']) < 2]

#------------------------------------------------------------------------------------
# Plot spatial distribution of original sample
fig, ax = plt.subplots(1, 1, figsize=(9, 4))

c = SkyCoord(ra=select_stars['RA']*u.degree, dec=select_stars['DEC']*u.degree,
             distance=select_stars['dist']*u.pc)
c_gal = c.galactocentric

ax.grid(True)
img = ax.scatter(select_stars['galr'], select_stars['galz'], c=select_stars['FE_H'],
                 s=0.075)
ax.set_ylim([-2, 2])
ax.set_xlabel(r'Galactocentric radius $R$')
ax.set_ylabel(r'Distance from midplane $z$')
# Add a colorbar
cbar = fig.colorbar(img, ax=ax)
cbar.ax.get_yaxis().labelpad = 8
cbar.ax.set_ylabel(r'$[\mathrm{Fe/H}]$', fontsize=18)
plt.savefig('stellar_sample_space.pdf', format='pdf', bbox_inches = 'tight')

#------------------------------------------------------------------------------------
# Plot [alpha/Fe] vs [Fe/H] and age-metallicity space of original sample
fig, ax = plt.subplots(1, 2, figsize=(18, 6))

im = ax[0].scatter(select_stars['FE_H'], select_stars['MG_H'] - select_stars['FE_H'],
                   c=select_stars['age'], s=0.03, cmap=plt.cm.coolwarm)
ax[0].set_xlim(-1.1, 0.6)
ax[0].set_ylim(-0.1, 0.45)
ax[0].set_xlabel(r'$[\mathrm{Fe/H}]$', fontsize=20)
ax[0].set_ylabel(r'$[\mathrm{Mg/Fe}]$', fontsize=20)
cbar = fig.colorbar(im, ax=ax[0])
cbar.ax.get_yaxis().labelpad = 25
cbar.ax.set_ylabel('Age (Gyr)', rotation=270, fontsize=18)

im = ax[1].scatter(select_stars['age'], select_stars['FE_H'],
                   c=select_stars['MG_H'] - select_stars['FE_H'], s=0.03, cmap=plt.cm.plasma)
ax[1].set_ylim(-1.1, 0.6)
ax[1].set_xlabel('Age (Gyr)', fontsize=20)
ax[1].set_ylabel(r'$[\mathrm{Fe/H}]$', fontsize=20)
# Add a colorbar
cbar = fig.colorbar(im, ax=ax[1])
cbar.ax.get_yaxis().labelpad = 25
cbar.ax.set_ylabel(r'$[\mathrm{Mg/Fe}]$', rotation=270, fontsize=18)
ax[0].grid()
ax[1].grid()
plt.savefig('stellar_sample_complete.pdf', format='pdf', bbox_inches = 'tight')

#------------------------------------------------------------------------------------
# Compute [alpha/Fe] vs [Fe/H] copula space of original sample
data = [select_stars['FE_H'], select_stars['MG_H'] - select_stars['FE_H']]
model = copula(data)

# Plot the data space of [alpha/Fe] vs [Fe/H]
fig, ax = plt.subplots(1, 2, figsize=(15, 8))

plt.subplots_adjust(hspace=0, wspace=0.4)
im = ax[0].hist2d(select_stars['FE_H'], select_stars['MG_H'] - select_stars['FE_H'],
                  bins=180, norm=mpl.colors.LogNorm(), cmap='magma')[3]

ax[0].set_xlim(-1.25, 0.75)
ax[0].set_ylim(-0.1, 0.5)
ax[0].set_xlabel(r'$[\mathrm{Fe/H}] \,\, \mathrm{or} \,\, x_1$')
ax[0].set_ylabel(r'[Mg$\mathrm{/Fe}] \,\, \mathrm{or} \,\, x_2$')
ax[0].set_xticks([-1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75])
ax[0].set_xticklabels(['', -1, '', -0.5, '', 0, '', 0.5, ''])
ax[0].set_yticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax[0].set_yticklabels(['', 0, '', 0.2, '', 0.4, ''])
plt.colorbar(im, ax=ax[0], orientation='horizontal', label='Number Density')

# create new axes on the right and on the top of the current axes
divider = make_axes_locatable(ax[0])
# below height and pad are in inches
ax_histx = divider.append_axes("top", 0.8, pad=0.2, sharex=ax[0])
ax_histy = divider.append_axes("right", 0.8, pad=0.2, sharey=ax[0])

# make some labels invisible
ax_histx.xaxis.set_tick_params(labelbottom=False)
ax_histy.yaxis.set_tick_params(labelleft=False)

# Plot margins of data
kdex = stats.gaussian_kde(select_stars['FE_H'])
kdey = stats.gaussian_kde(select_stars['MG_H'] - select_stars['FE_H'])
xx = np.linspace(-1.25, 0.7, 800)
xy = np.linspace(-0.15, 0.5, 800)
ax_histx.plot(xx, kdex(xx), color='black', lw=3, label=r'$f_1$')
ax_histy.plot(kdey(xy), xy, color='black', lw=3, label=r'$f_2$')
ax_histx.set_ylim(0, 2)
ax_histy.set_xlim(0, 7.5)
ax_histx.legend(fontsize=10)
ax_histy.legend(fontsize=10)

# Plot the copula space of [alpha/Fe] vs [Fe/H]
im = ax[1].hist2d(model.U[0], model.U[1], bins=180, norm=mpl.colors.LogNorm(), cmap='magma')[3]
xp = np.linspace(0, 1, 500)
ax[1].set_xlim(-0.075, 1.05)
ax[1].set_ylim(-0.075, 1.075)
ax[1].set_xticks([0, 0.25, 0.5, 0.75, 1])
ax[1].set_yticks([0, 0.25, 0.5, 0.75, 1])
ax[1].set_xticklabels([0, '', 0.5, '', 1])
ax[1].set_yticklabels([0, '', 0.5, '', 1])
ax[1].set_xlabel(r'$u_1$')
ax[1].set_ylabel(r'$u_2$')
plt.colorbar(im, ax=ax[1], orientation='horizontal', label='Number Density')
# create new axes on the right and on the top of the current axes
divider = make_axes_locatable(ax[1])
# below height and pad are in inches
ax_histx = divider.append_axes("top", 0.8, pad=0.2, sharex=ax[1])
ax_histy = divider.append_axes("right", 0.8, pad=0.2, sharey=ax[1])

# make some labels invisible
ax_histx.xaxis.set_tick_params(labelbottom=False)
ax_histy.yaxis.set_tick_params(labelleft=False)

# Plot margins of copula
dat_range = np.linspace(-0.075, 1.075, 1000)
ax_histx.plot(dat_range, stats.uniform.pdf(dat_range), color='black', lw=3)
ax_histy.plot(stats.uniform.pdf(dat_range), dat_range, color='black', lw=3)
ax_histx.set_ylim(0, 1.25)
ax_histy.set_xlim(0, 1.25)

fig.text(0.48, 0.55, r'$\rightarrow$', fontsize=30)
fig.text(0.47, 0.65, r'$F_{1}(x_1)$', fontsize=20)
fig.text(0.47, 0.6, r'$F_{2}(x_2)$', fontsize=20)

#------------------------------------------------------------------------------------
# Obtain alpha sequence split in copula space of original sample
# Compute copula on a grid
u_grid, kde_grid = model.Generate_Copula_KDE()

# Compute copula contours
cop_contour = plt.contour(u_grid[0], u_grid[1], kde_grid, levels=53)

# Compute flow line using contours
vertices = []
for ind in range(1, 53):
    paths = cop_contour.collections[ind].get_paths()
    for p in paths:
        v = p.vertices
        vertices.append(v)
        
split_vertices = np.array([vertices[4][14], vertices[7][15], vertices[8][0],
                           vertices[10][18], vertices[12][2], vertices[14][43],
                           vertices[17][47], vertices[18][2], vertices[20][52],
                           vertices[23][61], vertices[25][1], vertices[25][3],
                           vertices[27][67], vertices[28][5]])

# Fit a second-degree polynomial to copula split
fit_vert_2 = np.polyfit(split_vertices[:, 0], split_vertices[:, 1], deg=2)
fit_val_2 = np.poly1d(fit_vert_2)

# Plot the alpha sequence split in copula space
fig, ax = plt.subplots(2, 2, figsize=(25, 18), sharey=True)

im = ax[0][0].scatter(u_grid[0], u_grid[1], c=kde_grid, s=275)
xp = np.linspace(0, 1, 500)
ax[0][0].set_xlim(0, 1)
ax[0][0].set_ylim(0, 1)
ax[0][0].set_xticks([0, 0.25, 0.5, 0.75, 1])
ax[0][0].set_yticks([0, 0.25, 0.5, 0.75, 1])
ax[0][0].set_xticklabels([0, '', 0.5, '', 1])
ax[0][0].set_yticklabels([0, '', 0.5, '', 1])
ax[0][0].set_xlabel(r'$u_1$')
ax[0][0].set_ylabel(r'$u_2$')

im = ax[0][1].scatter(u_grid[0], u_grid[1], c=kde_grid, s=275)
cop_contour = ax[0][1].contour(u_grid[0], cop_grid_all[1], kde_grid,
                               levels=53, colors='white', alpha=0.75)
xp = np.linspace(0, 1, 500)
ax[0][1].set_xlim(0, 1)
ax[0][1].set_ylim(0, 1)
ax[0][1].set_xticks([0, 0.25, 0.5, 0.75, 1])
ax[0][1].set_yticks([0, 0.25, 0.5, 0.75, 1])
ax[0][1].set_xticklabels([0, '', 0.5, '', 1])
ax[0][1].set_yticklabels([0, '', 0.5, '', 1])
ax[0][1].set_xlabel(r'$u_1$')
plt.colorbar(im, ax=ax[:], orientation='vertical', label='Copula Density')

im = ax[1][1].scatter(u_grid[0], u_grid[1], c=kde_grid, s=275)
cop_contour = ax[1][1].contour(u_grid[0], u_grid[1], kde_grid, levels=53,
                               colors='white', alpha=0.75)
xp = np.linspace(0, 1, 500)
ax[1][1].set_xlim(0, 1)
ax[1][1].set_ylim(0, 1)
ax[1][1].set_xticks([0, 0.25, 0.5, 0.75, 1])
ax[1][1].set_yticks([0, 0.25, 0.5, 0.75, 1])
ax[1][1].set_xticklabels([0, '', 0.5, '', 1])
ax[1][1].set_yticklabels([0, '', 0.5, '', 1])
ax[1][1].set_xlabel(r'$u_1$')
ax[1][1].scatter(split_vertices[:, 0], split_vertices[:, 1], color='white', s=75)

xp = np.linspace(0, 1, 500)
im = ax[1][0].scatter(u_grid[0], u_grid[1], c=kde_grid, s=275)
ax[1][0].plot(xp, fit_val_2(xp), '--', color='white', lw=3,
              label='Copula Space Split')
ax[1][0].set_xlim(0, 1)
ax[1][0].set_ylim(0, 1)
ax[1][0].set_xticks([0, 0.25, 0.5, 0.75, 1])
ax[1][0].set_yticks([0, 0.25, 0.5, 0.75, 1])
ax[1][0].set_xticklabels([0, '', 0.5, '', 1])
ax[1][0].set_yticklabels([0, '', 0.5, '', 1])
ax[1][0].set_xlabel(r'$u_1$')
ax[1][0].set_ylabel(r'$u_2$')
ax[1][0].legend(fontsize=14)
plt.savefig('split_tracks.pdf', format='pdf', bbox_inches = 'tight')

#------------------------------------------------------------------------------------
# Obtain alpha sequence split in data space of original sample (using copula split)

X = np.zeros(shape=(select_stars.shape[0], 2))
X[:, 0] = select_stars['FE_H']
X[:, 1] = select_stars['MG_H'] - X[:, 0]

# Generate the inverse ECDF lambda function
IECDF = []
interp = []
for i in range(2):
    X_sorted = np.sort(X[:, i])
    U_sorted = model.F[i](X_sorted)
    interp.append(interp1d(U_sorted, X_sorted))
    IECDF.append(lambda u, i=i: interp[i](u))

# Use the inverse ECDF to obtain data space split
xp = np.linspace(0.001, 0.999, 500)
dat_fit_x = IECDF[0](xp)
dat_fit_y = IECDF[1](fit_val_2(xp))

# Fit a third-degree polynomial to data split
dat_fit = np.polyfit(dat_fit_x, dat_fit_y, deg=3)
dat_fit_line = np.poly1d(dat_fit)

np.save('dat_space_split_fit', dat_fit_line.c)

# Divide the data using data space split
above_inds = (model.U[1] > fit_val_2(model.U[0]))
below_inds = (model.U[1] <= fit_val_2(model.U[0]))

fig, ax = plt.subplots(1, 1, figsize=(9, 5))
im_1 = ax.hist2d(select_stars[above_inds]['FE_H'],
                 select_stars[above_inds]['MG_H'] - select_stars[above_inds]['FE_H'],
                 bins=180, norm=mpl.colors.LogNorm(), cmap='Blues')[3]

im_2 = ax.hist2d(select_stars[below_inds]['FE_H'],
                 select_stars[below_inds]['MG_H'] - select_stars[below_inds]['FE_H'],
                 bins=180, norm=mpl.colors.LogNorm(), cmap='Oranges')[3]

ax.plot(dat_fit_x, dat_fit_y, color='grey', lw=3,
        label='Data Space Split (from copula)')

ax.plot(x_dat, dat_fit_line(x_dat), '--', color='black', lw=2,
        label='Polynomial Fit to Data Space Split')
ax.set_xlim(-1.25, 0.75)
ax.set_ylim(-0.1, 0.525)
ax.set_xlabel(r'$[\mathrm{Fe/H}]$')
ax.set_ylabel(r'[$\mathrm{Mg/Fe}]$')
ax.legend(fontsize=11)
ax.set_xticks([-1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75])
ax.set_xticklabels(['', -1, '', -0.5, '', 0, '', 0.5, ''])
ax.set_yticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.set_yticklabels(['', 0, '', 0.2, '', 0.4, ''])

plt.colorbar(im_1, ax=ax, pad=0, label='Number Density')
plt.colorbar(im_2, ax=ax)
plt.savefig('split_tracks_data.pdf', format='pdf', bbox_inches = 'tight')

#------------------------------------------------------------------------------------
# Compare our alpha sequence split with those in literature
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
im_1 = ax.hist2d(select_stars[above_inds]['FE_H'],
                 select_stars[above_inds]['MG_H'] - select_stars[above_inds]['FE_H'],
                 bins=180, norm=mpl.colors.LogNorm(), cmap='Blues', alpha=0.25)[3]

im_2 = ax.hist2d(select_stars[below_inds_2]['FE_H'],
                 select_stars[below_inds]['MG_H'] - select_stars[below_inds]['FE_H'],
                 bins=180, norm=mpl.colors.LogNorm(), cmap='Oranges', alpha=0.25)[3]

ax.plot(x_dat, dat_fit_line(x_dat), '--', color='black', lw=2,
        label='Patil et al. 2023 (Mg)')
ax.plot(x_dat, dat_fit_line(x_dat), color='black', lw=1,
    label='Patil et al. 2023 (Mg)')

x_sep1 = np.linspace(-1, 0.2, 500)
x_sep2 = np.linspace(0.2, 0.5, 500)
ax.plot(x_sep1, (0.12/-0.6)*x_sep1 + 0.05 + 0.05, '-.', color='#2FD7D4', lw=2)
ax.plot(x_sep2, ((0.12/-0.6)*0.2 + 0.05 + 0.05)*np.ones(500), '-.', color='#2FD7D4',
        lw=2, label=r'Mackereth et al. 2019 (Mg)')

x_sep1 = np.linspace(-1, 0, 500)
x_sep2 = np.linspace(0, 0.5, 500)
ax.plot(x_sep1, 0.12 -0.13*x_sep1, '-.', color='#D81B60', lw=2, label='')
ax.plot(x_sep2, 0.12*np.ones(500), '-.', color='#D81B60', lw=2,
        label='Weinberg et al. 2019 (Mg)')

ax.plot(x_dat, -0.12*x_dat + 0.13, '-.', color='#004D40', lw=2,
        label=r'Frankel et al. 2020 ($\alpha$)')

ax.plot(x_dat, -0.08*x_dat + 0.14, '-.', color='#FFC107', lw=2,
        label=r'Gandhi & Ness 2019 ($\alpha$, LAMOST)')

ax.set_xlim(-1.25, 1)
ax.set_ylim(-0.1, 0.55)
ax.set_xlabel(r'$[\mathrm{Fe/H}]$')
ax.set_ylabel(r'[$\mathrm{Mg/Fe}$] or [$\alpha\mathrm{/Fe}]$')
ax.legend(fontsize=10)
ax.set_xticks([-1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75])
ax.set_xticklabels(['', -1, '', -0.5, '', 0, '', 0.5, ''])
ax.set_yticks([-0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])
ax.set_yticklabels(['', 0, '', 0.2, '', 0.4, ''])
plt.savefig('splits_compare.pdf', format='pdf', bbox_inches = 'tight')
