#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import growth.viz
from matplotlib.colors import ListedColormap
cor, pal = growth.viz.matplotlib_style()

# Load the example chemostat data
dynamics = pd.read_csv('../../../simulations/output/chemostat_tradeoff_comparison.csv')
dynamics['n_dil'] = dynamics['time'] * dynamics['dil_rate']

fig, ax = plt.subplots(2, 2, figsize=(3, 2))

colors = {'slow': cor['primary_blue'], 'fast': cor['primary_green']}
for g, d in dynamics.groupby(['tradeoff', 'label']):
    if g[0]:
        biomass_ax = ax[0, 1]
        freq_ax = ax[1, 1]
    else:
        biomass_ax = ax[0, 0]
        freq_ax = ax[1, 0]
    biomass_ax.plot(d['n_dil'], d['M'], lw=1, color=colors[g[1]])
    freq_ax.plot(d['n_dil'], d['frequency'], lw=1, color=colors[g[1]])

for i in range(2): 
    ax[0, i].set_ylim([-0.5, 12])
    ax[0, i].set_ylabel('biomass [a.u.]', fontsize=6)
    ax[1, i].set_ylabel('mass frequency', fontsize=6)

for a in ax.ravel(): 
    a.set_xlabel('number of dilutions', fontsize=6)
plt.savefig('./plots/chemostat_tradeoff_comparison.pdf', bbox_inches='tight')

#%% Plot the tradeoff relationship (hardcoded for now)
fig, ax = plt.subplots(1,2, figsize=(3, 1.5))
ax[0].plot([0.5, 1.0], [0.05, 0.05], '-', lw=0.5, color=cor['primary_black'])
ax[0].plot(0.5, 0.05, 'o', color=cor['primary_blue'], ms=5)
ax[0].plot(1.0, 0.05, 'o', color=cor['primary_green'], ms=5)

lam = np.array([0.5, 1.0])
ax[1].plot(lam, 0.05 + (lam - lam.min()), '-', lw=0.5, color=cor['primary_black'])
ax[1].plot(0.5, 0.05, 'o', color=cor['primary_blue'], ms=5)
ax[1].plot(1.0, 0.05 + 0.5 * lam[1], 'o', color=cor['primary_green'], ms=5)

ax[0].set_ylim([0, 0.1])
ax[1].set_ylim([0, 1.0])
for a in ax:
    a.set_ylabel('$\gamma$ [T$^{-1}$]\ndeath rate', fontsize=6)
    a.set_xlabel('$\lambda_{max}$ [T$^{-1}$]\nmax growth rate', fontsize=6)

plt.tight_layout()
plt.savefig('./plots/tradeoff_explainer.pdf')
#%%
phase_cmap = ListedColormap([cor['primary_blue'], cor['primary_green'], 
                                cor['primary_black']])

# Load the phase diagram data
data = pd.read_csv('../../../simulations/output/phase_diagram_end_states.csv')
fast = data[data['label']=='fast']

# Set up the matrix for display
rho, delta = fast['rho'].unique(), fast['dil_rate'].unique()
mat = np.zeros((len(delta), len(rho)))
rho_map = {r:i for i, r in enumerate(rho)}
delta_map = {d:i for i, d in enumerate(delta)}

# Instantiate the matrix
mat = np.zeros((len(rho), len(delta)))
for g, _ in fast.groupby(['rho', 'dil_rate', 'frequency']):
    i, j = rho_map[g[0]], delta_map[g[1]]
    print(g[2] == np.inf)
    if (g[2] > 0.5) and (g[2] != np.inf):
        mat[i,j] = 1
    elif g[2] == np.inf:
        mat[i, j] = 2
    else:
        mat[i, j] = 0

fig, ax = plt.subplots(1,1, figsize=(4,4))
ax.imshow(mat, cmap=phase_cmap)
ax.grid(False)
ax.set_xlim([0, 100])
ax.set_xticks([0, 50, 100])
ax.set_yticks([0, 50, 100])
ax.set_xticklabels([f'{delta[0]}', f'{(delta[-1] - delta[0])/2:0.2f}', f'{delta[-1]}'])
ax.set_yticklabels([f'{rho[-1]}', f'{(rho[-1] - rho[0])/2:0.2f}', f'{rho[0]}'])
ax.set_ylabel(r'$\rho$' + '\ngrowth-death tradeoff strength', fontsize=6)
ax.set_xlabel(r'$\delta$' + '\nsystem dilution rate [T$^{-1}$]', fontsize=6)
plt.savefig('./plots/phase_diagram.png', bbox_inches='tight')

#%% 
zindex = {'slow':1, 'fast': 2}
fig, ax = plt.subplots(3, 1, figsize=(2.5, 4), sharex=True)
rhos = [rho[0], rho[int(len(rho)/2)], rho[-5]]
for i, r in enumerate(rhos):
    df = data[data['rho']==r]
    for g, d in df.groupby('label'):
        d = d.copy()
        d.loc[d['frequency']==np.inf, 'frequency'] = 0 
        ax[i].plot(d['dil_rate'], d['frequency'], '-', color=colors[g], lw=1, 
                   zorder=zindex[g])
for a in ax:
    a.set_ylabel('end-state mass frequency', fontsize=6)
ax[-1].set_xlabel('dilution rate [T$^{-1}$]', fontsize=6)
plt.savefig('./plots/phase_switches.pdf', bbox_inches='tight')