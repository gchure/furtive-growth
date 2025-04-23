#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import growth.viz
from matplotlib.colors import ListedColormap
cor, pal = growth.viz.matplotlib_style()
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
ax.imshow(mat, origin='lower', cmap=phase_cmap)
ax.grid(False)
ax.set_xlim([0, 100])
ax.set_xticks([0, 50, 100])
ax.set_yticks([0, 50, 100])
ax.set_xticklabels([f'{delta[0]}', f'{(delta[-1] - delta[0])/2:0.2f}', f'{delta[-1]}'])
ax.set_yticklabels([f'{rho[0]}', f'{(rho[-1] - rho[0])/2:0.2f}', f'{rho[-1]}'])
ax.set_ylabel(r'$\rho$' + '\ngrowth-death tradeoff strength', fontsize=6)
ax.set_xlabel(r'$\delta$' + '\nsystem dilution rate [T$^{-1}$]', fontsize=6)
plt.savefig('./plots/phase_diagram.pdf', bbox_inches='tight')

#%% 
# Set two examples of chemostatic growth (one with tradeoff, one without)