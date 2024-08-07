#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import growth.viz 
import growth.model
import tqdm
cor, pal = growth.viz.matplotlib_style()

# Set up two species parameters 
gamma = np.array([0.05, 0.1])
lambda_max = np.array([1, 2]) 
cap_lam = lambda_max - gamma
ratio = cap_lam[1] / cap_lam[0]

#%%
# Define the common species parameters
common_pars = {'lambda_necro_max': 0,
               'Km_nut': 0.1,
               'Km_necro': 0.1,
               'Y_nut': 0.01,
               'Y_necro':0.01}

# Set the species as a bug array
bugs = [growth.model.Species(lambda_nut_max=lambda_max[i], gamma=gamma[i], **common_pars) for i in range(2)]

# Define the ecosystem
eco = growth.model.Ecosystem(bugs, init_total_biomass=1, init_total_necromass=0.0)
#%%
# Set a range of frequencies
total_time = 2000 
omegas = np.logspace(np.log10(2/total_time),  -1, 300)
freqs = pd.DataFrame()
for i, w in enumerate(tqdm.tqdm(omegas)):
    species, bulk = eco.grow(total_time, feed_conc=1E6, feed_freq=w, 
                         delta=0, freq_thresh=0, solver_kwargs={'method': 'LSODA'},
                         verbose=False, term_event={'type':'extinction', 'thresh':1E-3})
    if eco.terminated:
        end_state = species[species['time']==species['time'].max()].copy()
    else:
        end_state = species[species['integration_window']>=(species['integration_window'].max() - 10)].copy()
        end_state = end_state.groupby('species_idx').mean().reset_index()
    end_state.drop(columns=['M_bio','M_necro', 'lambda_inst_nut', 'lambda_inst_necro',
                            'lambda_max_necro', 'Km_nut', 'Km_necro',
                            'Y_nut', 'Y_necro'], inplace=True)
    end_state['omega'] = w    # break
    freqs = pd.concat([freqs, end_state])

#%% 
species = species[species['time']>=0.75*total_time]
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
# ax[0].set_ylim([0.001, 1E5])
ax[1].set_ylim([1E-4, 2])

#%%
fig, ax = plt.subplots(1, 1, figsize=(4, 2))
ax.set_xscale('log')
label = {1 : {'c': cor['primary_black'],
              'label': f'$\gamma$ = {gamma[0]}; $\lambda$ = {lambda_max[0]}'},
         2 : {'c':  cor['primary_red'],
              'label': f'$\gamma$ = {gamma[1]}; $\lambda$ = {lambda_max[1]}'}
}
for g, d in freqs.groupby('species_idx'):
    ax.plot(d['omega'], d['mass_frequency'],lw=1, color=label[g]['c'],
            label=label[g]['label'])
ax.set_xlabel('nutrient loading frequency [inv. time]', fontsize=6)
ax.set_ylabel('species frequency', fontsize=6)
ax.legend()
