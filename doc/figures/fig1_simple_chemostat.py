#%%
import importlib
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import growth.viz
import growth.model
import seaborn as sns
import tqdm
importlib.reload(growth.model)
np.random.seed(3028)  # Set a random seed for reproducibility

cor, pal = growth.viz.matplotlib_style()

# Define thej shared parameters
params = {'lambda_necro_max': 0,
          'Km_necro': 0.01,
          'Km_nut': 0.01,
          'gamma': 0.1,
          'Y_necro': 0.01,
          'Y_nut': 0.1}

# Define the number of species and randomly draw lambda_max
N_species = 100
lambda_max = np.random.normal(1, 0.1, N_species-1)
lambda_max = np.concatenate([lambda_max, [1.3]])

# Set the species array
bugs = [growth.model.Species(lambda_nut_max=lambda_max[i], **params) for i in range(N_species)]

# Define the ecosystem
eco = growth.model.Ecosystem(bugs, init_total_biomass=100, init_total_necromass=0.0)

# Simulate the ecosystem
delta = 0.1
species, bulk = eco.grow(200, feed_freq=-1, feed_conc=1E4, delta=delta, dt=0.001,
                         biomass_thresh=0)

#%%
# Set up a summary figure. 
fig, ax = plt.subplots(2, 1, figsize=(2, 2), sharex=True)

ax[0].plot(bulk['time'].values[::10] * delta, bulk['M_bio_tot'].values[::10], color=cor['blue'], lw=1.5)
# Prepare data for stackplot
mass_frequencies = [d['mass_frequency'].values for g, d in species.groupby('species_idx')]
time_values = [d['time'].values for g, d in species.groupby('species_idx')]

for g, d in species.groupby('species_idx'):
    if g == 100:
        color = cor['primary_red']
        lw=1.5
        zorder=1000
    else:
        color = cor['primary_black']
        lw=0.5
        zorder=10
    ax[1].plot(d['time'].values[::10] * delta, d['mass_frequency'].values[::10], 
               color=color,
               lw=lw, zorder=zorder)

ax[1].set_yscale('log')
ax[0].set_ylabel('total biomass', fontsize=6)
ax[1].set_ylabel('frequency', fontsize=6)
ax[1].set_xlabel('$t/\delta$', fontsize=6)

ax[0].set_ylim([10, 1E3])
ax[1].set_ylim([1E-3, 1])
plt.savefig('./plots/fig1_chemostat_dynamics.pdf', bbox_inches='tight')
#%%
# Plot the distribution of 
fig, ax = plt.subplots(1,1, figsize=(2,1))
ax.hist(lambda_max[:-1], bins=20, color=cor['light_black'])
ax.bar(lambda_max[-1], 1, width=0.02,  color=cor['primary_red'])
ax.set_xlabel('$\lambda^*_s$', fontsize=6)
ax.set_ylabel('number of species', fontsize=6)
plt.savefig('./plots/fig1_chemostat_lambda_dist.pdf', bbox_inches='tight')


#%%
# Scan over dilution rates
delta_range = np.linspace(0.05, 0.8, 10)
steadystate = pd.DataFrame()
for i, delta in enumerate(tqdm.tqdm(delta_range)):
    species, bulk = eco.grow(2000, feed_freq=-1, feed_conc=1E4, delta=delta,
                         biomass_thresh=0, verbose=False)
    _species = species[species['time'] == species['time'].max()]
    _species = _species[['species_idx', 'mass_frequency']]
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
        lw=0.5
        color = cor['primary_black']
    ax.plot(d['delta'], d['mass_frequency'],  ms=4, color=color, lw=lw)
ax.set_xlabel('$\delta$ [hr$^{-1}$]', fontsize=6)
ax.set_ylabel('end-state speciesfrequency', fontsize=6)
plt.savefig('./plots/fig1_chemostat_endstate_frequency.pdf', bbox_inches='tight')
 
