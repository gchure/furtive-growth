#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import growth.viz
import growth.model
cor, pal = growth.viz.matplotlib_style()


# Set species parameters
gamma = np.array([0.08, 0.1])
lambda_max = np.array([1, 1.5])

