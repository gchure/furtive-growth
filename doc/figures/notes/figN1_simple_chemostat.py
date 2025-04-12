"""
This script makes a simple set of plots of two species competing for growth in a
chemostat.
"""
#%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import growth.viz
from growth.model import Species, Ecosystem
cor, pal = growth.viz.matplotlib_style()

# Generate an array of growth rates.
n_species = 2
lambda_max = [1.5, 2]

# Set the common parameter dictionary for all species
pars = {'Km': 0.01,
        'gamma': 0.1,
        'Y': 0.1}

# Set the community using a list comprehension
community = [Species(lambda_max=lambda_max[0], label='slow', **pars),
             Species(lambda_max=lambda_max[0], label='fast', **pars)]

# Set the ecosystem
ecosystem = growth.model.Ecosystem(community)
