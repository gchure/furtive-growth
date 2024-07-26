#%%

import importlib
import numpy as np 
import pandas as pd 
import growth.viz
import growth.model
importlib.reload(growth.model)
cor, pal = growth.viz.matplotlib_style()

# Set an initial species
bug = growth.model.Species(lambda_nut_max=1.0,
                           lambda_necro_max=0.1,
                           Km_nut=0.1, 
                           Km_necro=0.1,
                           gamma=0.1,  
                           Y_nut=0.1,
                           Y_necro=0.1)

eco = growth.model.Ecosystem([bug], 
                            init_total_biomass=1.0,
                            init_total_necromass=0.0)
eco.grow([0, 10])