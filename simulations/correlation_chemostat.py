#%%
import numpy as np 
import pandas as pd 
import growth.model
import tqdm

# Set the variable growth rates 
lambda_max = np.array([0.8, 0.9, 1.1, 1.3])

# Set the correlated death rates
slope = 0.25
gamma = (lambda_max-lambda_max.min()) * slope + 0.05

# Set the collection of species
bugs = [growth.model.Species(lambda_nut_max=lambda_max[i], 
                             gamma = gamma[i],
                             Km_nut = 0.01,
                             Y_nut = 0.1) for i in range(len(lambda_max))]

# Set the initial dilution rate for the example trajectory and the feedstock concentration.
delta = 0.1
c_0 = 1E3
M_init = 1E2
eco = growth.model.Ecosystem(bugs, init_total_biomass=M_init)

# Run an example simulation with a moderate dilution rate (0.5).
species, bulk = eco.grow(1000, feed_freq=-1, feed_conc=c_0, delta=delta,
                dt=1E-3, term_event={'type': 'fixation', 'freq_thresh': 0.999})
species['dilution_rate'] = delta
bulk['dilution_rate'] = delta
species.to_csv('output/correlation_chemostat_example_species.csv', index=False)
bulk.to_csv('output/correlation_death_chemostat_example_bulk.csv', index=False)

#%%
# Sweep over dilution rates. 
# Sweep over dilution rates and compute the end-state frequency.
endstate = pd.DataFrame([])
delta_range = np.linspace(0.01, 0.9, 100)
for i, delta in enumerate(tqdm.tqdm(delta_range)):
    eco = growth.model.Ecosystem(bugs, init_total_biomass=M_init)
    species, bulk = eco.grow(1E6, feed_freq=-1, feed_conc=c_0,
                             delta=delta,
                             term_event={'type':'fixation',
                                         'freq_thresh': 0.999},
                            verbose=False)
    # Get the end state frequency
    _endstate = species[species['time']==species['time'].max()]
    _endstate = _endstate[['mass_frequency', 'lambda_max_nut', 'gamma', 'species_idx']]
    _endstate['dilution_rate'] = delta
    endstate = pd.concat([endstate, _endstate])
endstate.to_csv('output/correlation_chemostat_endstate.csv', index=False)

#%%
import matplotlib.pyplot as plt
for g, d in endstate.groupby(['species_idx']):
    plt.plot(d['dilution_rate'], d['mass_frequency'], label=f"Species {g}")
plt.xlim([0, 0.2])