"""
This script makes a simple set of plots of two species competing for growth in a
chemostat.
"""
#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import growth.viz
from growth.model import Species, Ecosystem
cor, pal = growth.viz.matplotlib_style()

# Generate an array of growth rates.
n_species = 2
lambda_max = [1.5, 2]

# Set the common parameter dictionary for all species
pars = {'Km': 0.01,
        'gamma': 0.8,
        'Y': 0.1}

# Set the community using a list comprehension
community = [Species(lambda_max=lambda_max[0], label='slow', **pars),
             Species(lambda_max=lambda_max[1], label='fast', **pars)]

# Set the ecosystem
M0 = 1E2
ecosystem = Ecosystem(community, init_total_biomass=M0)

# Grow under a constant dilution scheme
dilution_rate = 0.001
n_dil = 10
total_time = n_dil / dilution_rate
# total_time = 1E2
c0 = 1E5
species_df, bulk_df = ecosystem.grow(total_time, feed_conc=c0,
                                     delta=dilution_rate)
bulk_df['n_dil'] = bulk_df['time']# * dilution_rate
species_df['n_dil'] = species_df['time']# * dilution_rate

#%%
# Set the figure with the example dynamics
fig, ax = plt.subplots(2, 1, figsize=(3, 3), sharex=True)
biomass_ax = ax[0]
freq_ax = ax[1]

# Plot the total biomass and nutrient concentration
biomass_ax.plot(bulk_df['n_dil'], bulk_df['M_tot'], '-', lw=1, color=cor['primary_black'], label='total')


# Plot species mass frequencies
spec_colors = [cor['primary_green'], cor['primary_blue']]
spec_linestyle = ['-', '-']
for i, (g, d) in enumerate(species_df.groupby('label')): 
    biomass_ax.plot(d['n_dil'], d['M'], lw=1, color=spec_colors[i], label=g)
    freq_ax.plot(d['n_dil'], d['frequency'], lw=1, 
                 color=spec_colors[i], label=g)
# Set axis scaling and limits
biomass_ax.set_yscale('log')
# biomass_ax.set_ylim(1, 1E3)
freq_ax.set_ylim([1E-3, 2])
freq_ax.set_yscale('log')
biomass_ax.legend()
# Add context
biomass_ax.set_ylabel('$M(t)$\nsystem biomass', fontsize=6)
freq_ax.set_ylabel('$M_s(t)/M(t)$\nspecies mass-frequency', fontsize=6)
