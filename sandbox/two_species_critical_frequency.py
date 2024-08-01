#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import growth.viz 
import growth.model
import tqdm
cor, pal = growth.viz.matplotlib_style()

# Set up two species parameters 
gamma = np.array([0.1, 0.1])
corr_strength = 5
lambda_max = np.array([1, 2]) 

# Define the common species parameters
common_pars = {'lambda_necro_max': 0,
               'Km_nut': 0.1,
               'Km_necro': 0.1,
               'Y_nut': 0.1,
               'Y_necro':0.1}

# Set the species as a bug array
bugs = [growth.model.Species(lambda_nut_max=lambda_max[i], gamma=gamma[i], **common_pars) for i in range(2)]

# Define the ecosystem
eco = growth.model.Ecosystem(bugs, init_total_biomass=0.05, init_total_necromass=0.0)

# Set a range of frequencies
omegas = np.logspace(-3, 0, 100)
total_time = 1000 

freqs = pd.DataFrame()
for i, w in enumerate(tqdm.tqdm(omegas)):

    species, bulk = eco.grow(total_time, feed_conc=1E3, feed_freq=0.02, dt=0.001,
                         delta=0, freq_thresh=1E-4, solver_kwargs={'method': 'LSODA'},
                         term_event={'type':'extinction', 'thresh':1E-4},
                         verbose=False)
    end_state = species[species['time']==species['time'].max()].copy()
    end_state.drop(columns=['M_bio','M_necro', 'lambda_inst_nut', 'lambda_inst_necro',
                            'lambda_max_necro', 'Km_nut', 'Km_necro',
                            'Y_nut', 'Y_necro'], inplace=True)
    end_state['omega'] = w
    break
    freqs = pd.concat([freqs, end_state])

#%% 
# Add colors to the species
species['color'] = cor['primary_black']
species.loc[species['species_idx']==2, 'color'] = cor['primary_red']

fig, ax = plt.subplots(1, 2, figsize=(4, 2))
for a in ax:
    a.set_yscale('log')
    a.set_xlabel('time', fontsize=6)
for g, d in species.groupby('color'):
    ax[0].plot(d['time'], d['M_bio'], '-', color=g, lw=1)
    ax[1].plot(d['time'], d['mass_frequency'], '-', color=g, lw=1)
ax[0].set_ylim([0.001, 1E5])
ax[1].set_ylim([1E-4, 2])

#%%
for g, d in freqs.groupby('species_idx'):
    if g == 1:
        c = cor['primary_black']
    else:
        c = cor['primary_red']
    plt.plot(d['omega'], d['mass_frequency'], lw=1, color=c)

#%%
fig, ax = plt.subplots(1, 2, figsize=(4, 2))
for a in ax:
    # a.set_yscale('log')
    a.set_xlabel('time', fontsize=6)
for g, d in species.groupby('color'):
    ax[0].plot(d['time'], d['lambda_inst_nut'], '-', lw=1)
    # ax[1].plot(d['time'], d['mass_frequency'], '-', color=g, lw=1)





