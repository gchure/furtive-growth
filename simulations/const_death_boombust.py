#%%
import numpy as np 
import pandas as pd 
import growth.model
import tqdm

# Set the variable growth rates 
lambda_max = np.linspace(0.8, 1.3, 5)

# Set the collection of species
bugs = [growth.model.Species(lambda_nut_max=lambda_max[i], 
                             gamma = 0.05,
                             Km_nut = 0.01,
                             Y_nut = 0.1) for i in range(len(lambda_max))]

# Set the inits
c_0 = 1E5
M_init = 1E2
feed_freq = 0.25

eco = growth.model.Ecosystem(bugs, init_total_biomass=M_init)

# Run an example simulation with a moderate feeding frequency
species, bulk = eco.grow(1000, feed_freq=feed_freq, feed_conc=c_0, delta=0,
                dt=1E-2, term_event={'type': 'fixation', 'freq_thresh': 0.999})
species['feed_freq'] = feed_freq
bulk['feed_freq'] = feed_freq
species.to_csv('output/const_death_boombust_example_species.csv', index=False)
bulk.to_csv('output/const_death_boombust_example_bulk.csv', index=False)

#%%
# Sweep over feed frequencies and compute the end-state frequency.
endstate = pd.DataFrame([])
feed_freq_range = np.linspace(0.05, 1, 100)
for i, omega in enumerate(tqdm.tqdm(feed_freq_range)):
    M_init = 1E2
    eco = growth.model.Ecosystem(bugs, init_total_biomass=M_init)
    species, bulk = eco.grow(1E4, feed_freq=omega, feed_conc=1E5,
                             delta=0, dt=1E-3, 
                             term_event={'type':'fixation',
                                         'freq_thresh': 0.999},
                            verbose=False)
    # Get the end state frequency
    _endstate = species[species['time']==species['time'].max()]
    _endstate = _endstate[['mass_frequency', 'lambda_max_nut', 'gamma', 'species_idx']]
    _endstate['feed_frequency'] = omega
    endstate = pd.concat([endstate, _endstate])
endstate.to_csv('output/const_death_boombust_endstate.csv', index=False)

#%%
for g, d in endstate.groupby(['species_idx']):
    plt.plot(d['feed_frequency'], d['mass_frequency'], label=f"Species {g}")
plt.legend()
plt.xlabel('Feed frequency')
plt.ylabel('Mass frequency')
plt.xlim([0, 1])
plt.ylim([1E-4, 1])
plt.show()
