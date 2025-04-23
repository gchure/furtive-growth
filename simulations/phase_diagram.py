"""
This script runs a number of simulations under different correlation strengths.
"""
#%%
import numpy as np
import pandas as pd
from growth.model import Species, Ecosystem
from tqdm import tqdm
import multiprocessing as mp
from joblib import Parallel, delayed
import gc
resolution = 100
# Set a two species environment
lambda_max = np.array([0.5, 1.0])

# Set a range of correlations for the death rate
corr_strength = np.linspace(0, 1, resolution)
gamma_min = 0.05

# Set the dilution rate range
dil_rate_range = np.linspace(0.01, 0.9, resolution)

# Run the simulation for 100 dilutions
total_time = 800 / dil_rate_range

# Set the feed concentration and the total initial biomass
M0 = 10 
c0 = 1E2
min_thresh = 2.0 # Minimum total biomass threshold for a living, non-washout scenario
labels = ['slow', 'fast']

# Set param combinations
param_combinations = []
for i, rho in enumerate(corr_strength):
    for j, delta in enumerate(dil_rate_range):
        param_combinations.append((rho, delta, total_time[j])) 

# Define a function for running the simulation that can be parallelized if desired.
def run_sim(pars: list) -> pd.DataFrame:
    """
    Runs a simulation and returns the endpoint species abundances.  
    """
    # Unpack parameters
    rho, delta, time = pars

    # Define the death rate given the correlation
    gamma = (lambda_max - lambda_max.min()) * rho + gamma_min

    # Define the species
    iterator = zip(lambda_max, gamma, labels)
    bugs = [Species(lambda_max=lam, gamma=gam, label=ell) for (lam, gam, ell) in iterator]

    # Set the ecosystem and grow
    eco = Ecosystem(bugs, init_total_biomass=M0) 
    species_df, bulk_df = eco.grow(time, feed_conc=c0, delta=delta, verbose=False)
    spec = species_df[species_df['time']==species_df['time'].max()].copy()
    spec['rho'] = rho
    spec['dil_rate'] = delta
    spec['c_nt_ss'] = bulk_df.iloc[-1]['M_nut']
    spec['frequency'] = [f if np.sum(spec['M']) >= min_thresh else float('inf') for f in spec['frequency'].values]
    return spec[['label', 'lambda_max', 'gamma', 'frequency', 
                 'rho', 'dil_rate', 'c_nt_ss', 'label']]

#%%
#Execute the simulation parallelized across cores
def main():
    jobs = mp.cpu_count() - 1
    batch_size = 50
    results = [] 
    for i in tqdm(range(0, len(param_combinations), batch_size)):
        batch = param_combinations[i:i+batch_size]
        batch_results = Parallel(n_jobs=jobs, verbose=0, backend='multiprocessing', 
                             max_nbytes='100M')(
            delayed(run_sim)(pars) for pars in batch
        )
        results.extend(batch_results)
        gc.collect()
    return pd.concat(results) 


df = main()
df.to_csv('./output/phase_diagram_end_states.csv', index=False)
print('Done!')