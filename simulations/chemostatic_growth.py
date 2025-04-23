#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm
from growth.model import Species, Ecosystem

# Set up the conditions
lambda_max = np.array([0.5, 1.0])
gamma_min = 0.05
rho_range = np.array([0.0, 1.0])
gamma_range = np.array([gamma_min + rho * (lambda_max - lambda_max.min()) for rho in rho_range])
M0 = 10 
c0 = 1E2
delta = 0.35
total_time = 100 / delta  

# Iterate through the gammas
labels = [False, True]
dfs = []
for i, gamma in enumerate(tqdm(gamma_range)):
    # Set the bugs and ecosystem
    bugs = [Species(lambda_max=lambda_max[0], gamma=gamma[0], label='slow'),
            Species(lambda_max=lambda_max[1], gamma=gamma[1], label='fast')]
    eco = Ecosystem(bugs, init_total_biomass=M0)

    # Grow for the time period, add context, and store
    species_df, _ = eco.grow(total_time, feed_conc=c0, delta=delta, verbose=False)
    species_df['tradeoff'] = labels[i]
    species_df['dil_rate'] = delta
    species_df.drop(columns=['extinct'], inplace=True)
    dfs.append(species_df)

df = pd.concat(dfs) 
df.to_csv('./output/chemostat_tradeoff_comparison.csv', index=False)


