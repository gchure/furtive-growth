#%%

import importlib
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import growth.viz
import growth.model
importlib.reload(growth.model)
cor, pal = growth.viz.matplotlib_style()

N_bugs = 299 

# Set bugs with normally distributed maximum growth rates
lambda_nut_max = np.abs(np.random.normal(1, 0.2, N_bugs))
lambda_nut_max = np.append(lambda_nut_max, 2)

# Define the common parameters
common_pars = {'lambda_necro_max': 0,
               'Km_nut': 0.1,
               'Km_necro': 0.1,
               'gamma': 0.01,
               'Y_nut': 0.1,
               'Y_necro':0.1}

# Set the list of bugs and populate the ecosystem 
bugs = [growth.model.Species(lambda_nut_max=lambda_nut_max[i], **common_pars) for i in range(len(lambda_nut_max))]
eco = growth.model.Ecosystem(bugs, init_total_biomass=0.05, init_total_necromass=0.0)
species, bulk = eco.grow(30, feed_conc=10, feed_freq=0.1, delta=0.05)

#%%
fig, ax = plt.subplots(2, 2, figsize=(4, 3))
ax = ax.ravel()
for a in ax[1:]:
    a.set_yscale('log')
    a.set_xlabel('time', fontsize=6)

# Plot the distribution of maximum growth rates and color indicate
ax[0].hist(lambda_nut_max, bins=20, color=cor['primary_black'], alpha=0.5, zorder=1000)
ylim = ax[0].get_ylim()[1]
ax[0].set_yticks([])
ax[0].fill_betweenx([0, ylim], 0.4, 1, color=cor['pale_red'], alpha=0.5, zorder=10)
ax[0].fill_betweenx([0, ylim], 1, 1.8, color=cor['pale_blue'], alpha=0.5, zorder=11)

# Plot the total biomass and the nutrient concentration
ax[1].plot(bulk['time'], bulk['M_bio_tot'], '-', color=cor['primary_black'], lw=1)
twax = ax[1].twinx()
twax.plot(bulk['time'], bulk['M_nut'], '-', color=cor['primary_green'], lw=1)
twax.set_yscale('log')
twax.set_ylabel('nutrient conc', fontsize=6, color=cor['primary_green'])
twax.set_ylim([1E-5, 1E1])
for g, d in species.groupby('species_idx'):
    if lambda_nut_max[g - 1] >= 1:
        c = cor['primary_blue']
        lw=0.1
    else:
        c = cor['primary_red']
        lw=0.1
    if lambda_nut_max[g - 1] >= 2:
        lw=1 
    ax[2].plot(d['time'], d['M_bio'], '-', color=c, lw=lw)
    ax[3].plot(d['time'], d['mass_frequency'], '-', color=c,  lw=lw)

# Add plot labels
ax[0].set_xlabel('max growth rate [inv. time]', fontsize=6)
ax[1].set_ylabel('total biomass', fontsize=6)
ax[2].set_ylabel('species biomass', fontsize=6)
ax[3].set_ylabel('species mass frequency', fontsize=6)
plt.tight_layout()
#%%
# Set two species with different growth rate
bug_1 = growth.model.Species(lambda_nut_max=1.0,
                           lambda_necro_max=0,
                           Km_nut=0.1, 
                           Km_necro=0.1,
                           gamma=0.1,  
                           Y_nut=0.1,
                           Y_necro=0.1)

bug_2 = growth.model.Species(lambda_nut_max=0.75,
                           lambda_necro_max=0,
                           Km_nut=0.1, 
                           Km_necro=0.1,
                           gamma=0.1,  
                           Y_nut=0.1,
                           Y_necro=0.1)
eco = growth.model.Ecosystem([bug_1, bug_2], 
                            init_total_biomass=0.1,
                            init_total_necromass=0.0)

species, bulk = eco.grow([0, 100], feed_conc=10, feed_freq=-1)