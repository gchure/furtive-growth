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
idx = [[0, 0], [90, 7], [60, 80]]
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
        species_df, bulk_df = eco.grow(50000, feed_conc=1E4, feed_freq=-1, delta=delta,
                freq_thresh=0, solver_kwargs={'method': 'LSODA'},dt=0.1, 
                term_event={'type':'extinction', 'thresh':1E-3},
                verbose=False)
        species_df['delta'] = delta

        end_state = species_df[species_df['time']==species_df['time'].max()]
        if (end_state['mass_frequency'].values[0] >= 0.4) & (end_state['mass_frequency'].values[0] <= 0.6):
            print(i,j)
        if [i,j] in idx:
            freqs = pd.concat([freqs, end_state])
            traj = pd.concat([traj, species_df])
        species_freq[i,j] = end_state[end_state['species_idx']==2]['mass_frequency'].values[0]

#%%
plt.figure(figsize=(2,2))
plt.imshow(species_freq.T, cmap='RdGy_r', origin='lower')
plt.grid(False)
ticks = [0, 24, 49, 74, 99]
xtick_labels = [f'{delta_range[i]:.2f}' for i in ticks]
ytick_labels = [f'{gamma_range[i]:.2f}' for i in ticks]
plt.xticks(ticks, xtick_labels)
plt.yticks(ticks, ytick_labels)
plt.xlabel('dilution rate $\delta$ [inv. time]', fontsize=6)
plt.ylabel('death rate $\gamma_2$ [inv. time]', fontsize=6)
plt.savefig('./death_v_dilution.pdf', bbox_inches='tight')
# ytick_labels = [f'{delta_range[i]:.2f}' for i in xticks]
# plt.colorbar()

#%%
fig, ax = plt.subplots(1, 3, figsize=(6,1.5))

for a in ax:
    a.set_xlabel('time', fontsize=6)
    a.set_ylabel('mass frequency', fontsize=6)
    a.set_ylim([1E-3, 1])
    a.set_xscale('log')
for i, (g, d) in enumerate(traj.groupby('delta')):
    _d = d[d['species_idx']==1] 
    ax[i].semilogy(_d['time'], _d['mass_frequency'], '-', 
                    color=cor['primary_black'], lw=1)
    _d = d[d['species_idx']==2] 
    ax[i].semilogy(_d['time'], _d['mass_frequency'], '-', 
                    color=cor['primary_red'], lw=1)
plt.tight_layout()
plt.savefig('./death_v_dilution_traj.pdf', bbox_inches='tight')


#%%

#%%
gamma = np.array([0, 0.1])
lambda_max = np.array([1, 2])
common_pars = {'lambda_necro_max': 0,
                'Km_nut': 0.1,
               'Km_necro': 0.1,
               'Y_nut': 0.01,
               'Y_necro':0.01}

#%%
omega_range = np.logspace(-3, -1, 50)
gamma_range = np.linspace(0, 0.1, 50)
# gamma_range = [0, 0.01, 0.05, 0.08, 0.09]
freqs = pd.DataFrame([])
traj = pd.DataFrame([])
lambda_max = np.array([1, 2])
# Choose which to save 
idx = [[0, 0], [49,49]]
# Set the storage matrix
species_freq = np.zeros((len(omega_range), len(gamma_range)))
for i, omega in enumerate(tqdm.tqdm(omega_range)):
    for j, gamma in enumerate(tqdm.tqdm(gamma_range)):
        _gamma = [gamma, 0.1]

        # Set the species
        bugs = [growth.model.Species(lambda_nut_max=lambda_max[i], gamma=_gamma[i], 
                        **common_pars) for i in range(2)]
        eco = growth.model.Ecosystem(bugs, init_total_biomass=1, init_total_necromass=0.0)
                                    

        # Growth the system to the end state 
        species_df, bulk_df = eco.grow(3000, feed_conc=1E4, feed_freq=omega, 
                                        delta=0, freq_thresh=0, 
                                        solver_kwargs={'method': 'LSODA'},dt=0.1, 
                                        term_event={'type':'extinction', 'thresh':1E-5},
                                        verbose=False)
        species_df['omega'] = omega

        if eco.terminated:
            end_state = species_df[species_df['time']==species_df['time'].max()].copy()
        else:
            end_state = species_df[species_df['integration_window']>=(species_df['integration_window'].max() - 1)].copy()
            end_state = end_state.groupby('species_idx').median().reset_index()
            end_state = species_df[species_df['time']==species_df['time'].max()]
        end_state['omega'] = omega
        if (end_state['mass_frequency'].values[0] >= 0.4) & (end_state['mass_frequency'].values[0] <= 0.6):
            print(i,j)
        # if [i,j] in idx:
            # freqs = pd.concat([freqs, end_state])
            # traj = pd.concat([traj, species_df])
        species_freq[i,j] = end_state[end_state['species_idx']==2]['mass_frequency'].values[0]

#%%
# plt.figure(figsize=(2,2))
plt.imshow(species_freq.T, cmap='RdGy_r', origin='lower', vmin=1E-3, vmax=1)
plt.grid(False)
# ticks = [0, 24, 49, 74, 99]
# xtick_labels = [f'{delta_range[i]:.2f}' for i in ticks]
# ytick_labels = [f'{gamma_range[i]:.2f}' for i in ticks]
# plt.xticks(ticks, xtick_labels)
# plt.yticks(ticks, ytick_labels)
plt.xlabel('feed frequency $\omega$ [inv. time]', fontsize=6)
plt.ylabel('death rate $\gamma_2$ [inv. time]', fontsize=6)
# ytick_labels = [f'{delta_range[i]:.2f}' for i in xticks]
# plt.colorbar()

#%%
fig, ax = plt.subplots(1, 3, figsize=(6,1.5))

for a in ax:
    a.set_xlabel('time', fontsize=6)
    a.set_ylabel('mass frequency', fontsize=6)
    a.set_ylim([1E-3, 1])
    # a.set_xscale('log')
for i, (g, d) in enumerate(traj.groupby('omega')):
    _d = d[d['species_idx']==1] 
    ax[i].semilogy(_d['time'], _d['mass_frequency'], '-', 
                    color=cor['primary_black'], lw=1)
    _d = d[d['species_idx']==2] 
    ax[i].semilogy(_d['time'], _d['mass_frequency'], '-', 
                    color=cor['primary_red'], lw=1)
plt.tight_layout()