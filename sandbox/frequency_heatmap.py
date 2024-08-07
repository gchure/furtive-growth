#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import growth.viz 
import growth.model
import tqdm
cor, pal = growth.viz.matplotlib_style()

gamma_range = np.linspace(0.01, 0.099, 50)
lambda_max = np.array([1, 2]) 

# Define the common species parameters
common_pars = {'lambda_necro_max': 0,
               'Km_nut': 0.1,
               'Km_necro': 0.1,
               'Y_nut': 0.01,
               'Y_necro':0.01}
total_time = 1000 
omegas = np.logspace(np.log10(2/total_time),  -1, 50)
specs = np.zeros((len(gamma_range), len(omegas)))
#%%
freqs = pd.DataFrame([])
for i, g in enumerate(tqdm.tqdm(gamma_range)):
#Set the species as a bug array
    gamma = [g, 0.1]
    bugs = [growth.model.Species(lambda_nut_max=lambda_max[i], gamma=gamma[i], **common_pars) for i in range(2)]

    # Define the ecosystem
    eco = growth.model.Ecosystem(bugs, init_total_biomass=1, init_total_necromass=0.0)
    for j, w in enumerate(tqdm.tqdm(omegas)):
        species, bulk = eco.grow(total_time, feed_conc=1E6, feed_freq=w, 
                         delta=0, freq_thresh=0, solver_kwargs={'method': 'LSODA'},
                         verbose=False, term_event={'type':'extinction', 'thresh':1E-3})
        if eco.terminated:
            end_state = species[species['time']==species['time'].max()].copy()
        else:
            end_state = species[species['integration_window']>=(species['integration_window'].max() - 5)].copy()
            end_state = end_state.groupby('species_idx').mean().reset_index()
            end_state.drop(columns=['M_bio','M_necro', 'lambda_inst_nut', 'lambda_inst_necro',
                            'lambda_max_necro', 'Km_nut', 'Km_necro',
                            'Y_nut', 'Y_necro'], inplace=True)
            end_state['omega'] = w    # break
        spec2 = end_state[end_state['species_idx']==2]['mass_frequency'].values[0]
        specs[i,j] = spec2
        freqs = pd.concat([freqs, end_state])

#%%
spec1 = freqs[freqs['species_idx']==1]

#%%
plt.matshow(np.log10(specs), cmap='RdGy')
np.log10(specs)