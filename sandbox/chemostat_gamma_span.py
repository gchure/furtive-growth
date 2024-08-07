#%%
import importlib
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import growth.viz
import growth.model
import tqdm
importlib.reload(growth.model)
cor, pal = growth.viz.matplotlib_style()

gamma = np.array([0, 0.1])
lambda_max = np.array([1, 1.25])
common_pars = {'lambda_necro_max': 0,
                'Km_nut': 0.1,
               'Km_necro': 0.1,
               'Y_nut': 0.01,
               'Y_necro':0.01}

#%%
delta_range = np.linspace(0.01, 0.4, 100)
gamma_range = np.linspace(0, 0.1, 100)
freqs = pd.DataFrame([])
traj = pd.DataFrame([])
# Choose which to save 
idx = [[0, 0], [50, 40], [60, 80]]
# Set the storage matrix
species_freq = np.zeros((len(delta_range), len(gamma_range)))
for i, delta in enumerate(tqdm.tqdm(delta_range)):
    for j, gamma in enumerate(tqdm.tqdm(gamma_range)):
        _gamma = [gamma, 0.1]

        # Set the species
        bugs = [growth.model.Species(lambda_nut_max=lambda_max[i], gamma=_gamma[i], 
                        **common_pars) for i in range(2)]
        eco = growth.model.Ecosystem(bugs, init_total_biomass=1, init_total_necromass=0.0)
                                    

        # Growth the system to the end state 
        species_df, bulk_df = eco.grow(10000, feed_conc=1E4, feed_freq=-1, delta=delta,
                freq_thresh=0, solver_kwargs={'method': 'LSODA'}, 
                term_event={'type':'extinction', 'thresh':1E-3},
                verbose=False)
        species_df['delta'] = delta

        end_state = species_df[species_df['time']==species_df['time'].max()]
        if [i,j] in idx:
            freqs = pd.concat([freqs, end_state])
            traj = pd.concat([traj, species_df])
        species_freq[i,j] = end_state[end_state['species_idx']==2]['mass_frequency'].values[0]

#%%
plt.matshow(species_freq.T, cmap='RdGy_r', origin='lower')
plt.grid(False)
plt.colorbar()

#%%
for g, d in traj.groupby('delta'):
    _d = d[d['species_idx']==1]
    plt.semilogy(_d['time'], _d['mass_frequency'], '-')