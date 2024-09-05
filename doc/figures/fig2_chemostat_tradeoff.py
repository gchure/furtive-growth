#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import tqdm 
import growth.model 
import growth.viz
cor, pal = growth.viz.matplotlib_style()
np.random.seed(3028)

# Define the shared parameters
params = {'lambda_necro_max': 0,
          'Km_necro': 0.01,
          'Km_nut': 0.01,
          'Y_necro': 0.01,
          'Y_nut': 0.1}

# Set the range of maximum growth rates
N_species = 100
lambda_max = np.random.normal(1, 0.1, N_species-1)
lambda_max = np.sort(np.concatenate([lambda_max, [1.3]]))

#%%
# Apply the linear tradeoff
dlambda = lambda_max.max() - lambda_max.min() 
slope = 0.09 / dlambda
gamma = (lambda_max-lambda_max.min()) * slope + 0.01 + np.random.normal(0, 0.002, N_species)


# Set a color palette for gamma
gamma_pal = sns.color_palette('crest_r', n_colors=N_species)

# Set the species array
bugs = [growth.model.Species(lambda_nut_max=lambda_max[i], gamma=gamma[i], **params) for i in range(N_species)]

# Set the ecosystem preferences
eco = growth.model.Ecosystem(bugs, init_total_biomass=100, init_total_necromass=0.0)

# Simulate the ecosystem
delta = 0.1
species, bulk = eco.grow(500, feed_freq=-1, feed_conc=1E4, delta=delta, dt=0.01,
                         biomass_thresh=0)

#%%
fig, ax = plt.subplots(2,1, figsize=(2, 2), sharex=True)
ax[0].plot(bulk['time'].values[::10] * delta, bulk['M_bio_tot'].values[::10], color=cor['blue'], lw=1.5)  
for g, d in species.groupby('species_idx'):
    if g == 100:
        color = cor['primary_red']
        lw=1.5
        zorder=1000
    else:
        color = gamma_pal[g-1]
        lw=0.2
        zorder=10
    ax[1].plot(d['time'].values[::10] * delta, d['mass_frequency'].values[::10], 
               color=color,
               lw=lw, zorder=zorder)
ax[1].set_yscale('log')
ax[1].set_xlabel('t / $\delta$', fontsize=6)
ax[0].set_ylabel('total biomass', fontsize=6)
ax[1].set_ylabel('frequency', fontsize=6)
ax[1].set_ylim([1E-4, 1E-1])
plt.savefig('./plots/fig2_chemostat_tradeoff_dynamics.pdf', bbox_inches='tight')

#%%
fig, ax = plt.subplots(1, 1, figsize=(2, 1))
ax.scatter(lambda_max, gamma, s=3, c=gamma_pal)
ax.plot(lambda_max[-1], gamma[-1], marker='o', ms=3, c=cor['primary_red'], zorder=1000)
ax.set_ylim([0.005, 0.11])
ax.set_yticks([0.01, 0.05, 0.1])
ax.set_xlim([0.7, 1.4])
ax.set_xlabel('$\lambda^*$ [hr$^{-1}$]', fontsize=6)
ax.set_ylabel('$\gamma$ [hr$^{-1}$]', fontsize=6)
plt.savefig('./plots/fig2_tradeoff.pdf', bbox_inches='tight')


#%%
# Scan over dilution rates
delta_range = np.linspace(0.05, 0.4, 300)
steadystate = pd.DataFrame()
steadystate_bulk = pd.DataFrame()
for i, delta in enumerate(tqdm.tqdm(delta_range)):
    species, bulk = eco.grow(10000, feed_freq=-1, feed_conc=1E4, delta=delta,
                         biomass_thresh=0, verbose=False)
    _bulk = bulk[bulk['time'] == bulk['time'].max()]
    _bulk = _bulk[['M_bio_tot', 'M_nut']]
    _bulk['delta'] = delta
    steadystate_bulk = pd.concat([steadystate_bulk, _bulk], sort=False)
    _species = species[species['time'] == species['time'].max()]
    _species = _species[['species_idx', 'mass_frequency', 'lambda_inst_nut', 'lambda_max_nut', 'gamma']]
    _species['cap_lambda_max'] = _species['lambda_max_nut'] - _species['gamma'] - delta
    _species['cap_lambda_eff'] = _species['lambda_inst_nut'] - _species['gamma'] - delta
    _species['mass_frequency'] = np.round(np.abs(_species['mass_frequency'].values), decimals=4)
    _species['delta'] = delta
    steadystate = pd.concat([steadystate, _species], sort=False)

#%%
fig, ax = plt.subplots(1,1, figsize=(4.5, 1))
for g, d in steadystate.groupby('species_idx'):
    if g == 100:
        lw=1
        color = cor['primary_red']
    else:
        lw=1
        color = gamma_pal[g-1]
    ax.plot(d['delta'], d['mass_frequency'],  ms=4, color=color, lw=lw)
ax.set_xlabel('$\delta$ [hr$^{-1}$]', fontsize=6)
ax.set_ylabel('end-state\nspecies frequency', fontsize=6)
plt.savefig('./plots/fig2_chemostat_tradeoff_endstate_frequency.pdf', bbox_inches='tight')

#%%
steadystate.to_csv('./fig2D_chemostat_tradeoff_endstate_frequency.csv')

#%%
steadystate