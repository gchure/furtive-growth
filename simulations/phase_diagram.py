#%%
import numpy as np
import pandas as pd
from growth.model import Species, Ecosystem
import growth.viz
from tqdm import tqdm
cor, pal = growth.viz.matplotlib_style()

# Set a two species environment
lambda_max = np.array([0.5, 1.0])

# Set a range of correlations for the death rate
corr_strength = np.linspace(0.01, 1, 50)
gamma_min = 0.05

# Set the dilution rate range
dil_rate_range = np.linspace(0.01, 0.45, 50)

# Run the simulation for 100 dilutions
total_time = 200 / dil_rate_range

# Set the feed concentration and the total initial biomass
M0 = 2
c0 = 1E5
labels = ['slow', 'fast']

#%%
# Store the resulting dataframes
dfs = []
fast_freq = np.zeros((len(corr_strength), len(dil_rate_range)))

# Iterate through simulations
for i, rho in enumerate(tqdm(corr_strength)):
    for j, delta in enumerate(dil_rate_range):
        # Determine the death rates.
        gamma = (lambda_max - lambda_max.min()) * rho + gamma_min

        # Set the species 
        bugs = [Species(lambda_max=lam, gamma=g, label=ell) for (lam, g, ell) in zip(lambda_max, gamma, labels)]

        # Set the ecosystem
        eco = Ecosystem(bugs, init_total_biomass=M0)

        # Grow
        species_df, bulk_df = eco.grow(total_time[j], feed_conc=c0, delta=delta,
        verbose=False)


        # Take the final time point and generate the dataframe
        spec = species_df[species_df['time'] == species_df['time'].max()].copy()
        spec['rho'] = rho
        spec['dil_rate'] = delta
        spec['M_nut_ss'] = bulk_df.iloc[-1]['M_nut']
        dfs.append(spec)

        fast = spec[spec['label']=='fast']
        fast_freq[i][j] = fast['frequency'].values[0] 
        # if np.sum(spec['M'] <= )
sim_df = pd.concat(dfs)

#%%
import matplotlib.pyplot as plt
plt.imshow(fast_freq)
plt.colorbar()