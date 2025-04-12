#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import growth.viz
import growth.model
cor, pal = growth.viz.matplotlib_style()
RNG = np.random.default_rng(3094)

# Generate an array of growth rates.
n_species = 49 
lambda_max = np.sort(RNG.normal(1.5, 0.1, size=n_species))
np.append(lambda_max, 2)

# Set the common parameter dictionary for all species
pars = {'Km': 0.01,
        'gamma': 0.1,
        'Y': 0.1}

# Set the community using a list comprehension
community = [growth.model.Species(lam, label=i, **pars) for i, lam in enumerate(lambda_max)]

# Set the ecosystem
ecosystem = growth.model.Ecosystem(community)

# Grow an example system.
dil_rate = 0.4
species, bulk = ecosystem.grow(lifetime=400, feed_conc=1E3, delta=dil_rate)
species['n_dil'] = species['time'] * dil_rate 
bulk['n_dil'] = bulk['time'] * dil_rate 

#%%
fig, ax = plt.subplots(2, 1, figsize=(4, 3), sharex=True)

# Plot species frequency dynamics
pal = sns.color_palette("crest", n_colors=n_species)

for i, (g, d) in enumerate(species.groupby('label')):
    ax[0].plot(d['n_dil'], d['frequency'], '-', color=pal[i])

# Plot nutrient and total biomass dynamics
twin = ax[1].twinx()
twin.set_yscale('log')
twin.plot(bulk['n_dil'], bulk['M_tot'], '-', color=cor['primary_black'], lw=1)
ax[1].plot(bulk['n_dil'], bulk['M_nut'], '-', color=cor['primary_red'], lw=1)

# Add labels:
ax[0].set_ylabel('species mass frequency', fontsize=6)
ax[1].set_ylabel('nutrient mass [a.u.]', color=cor['red'], fontsize=6)
twin.set_ylabel('total biomass [a.u.]', fontsize=6)
ax[1].set_xlabel(r'number of dilutions [$t\times \delta$]', fontsize=6)

# Add scaling and other context
twin.grid(False)
for a in ax:
    a.set_yscale('log')
ax[0].set_ylim([1E-3, 2])

