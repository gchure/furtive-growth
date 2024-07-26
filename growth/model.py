import numpy as np 
from scipy.integrate import solve_ivp
from pandas.core.frame import DataFrame
import pandas as pd
from tqdm import tqdm
from typing import Union

class Species():
    """Base class for a simple self replicator following Monod growth kinetics"""
    def __init__(self,
                 lambda_nut_max : float,
                 lambda_necro_max : float,
                 Km_nut : float,
                 Km_necro : float,
                 gamma : float,
                 Y_nut : float,
                 Y_necro : float 
                 ):
        """ 
        Initializes a self-replicating species with Monod growth kinetics.

        Parameters
        ----------
        lambda_nut_max : float
            The maximum growth rate of the species on nutrients.
        lambda_nut_max : float
            The maximum growth rate of the species on necromass. 
        Km_nut : float
            The Monod constant for growth on nutrients. 
        Km_necro : float
            The Monod constant for growth on necromass.
        gamma : float
            The death rate of the species.
        Y_nut : float
            The yield coefficient for growth on nutrients. Has dimensions of 
            biomass produced per unit of substrate consumed.            
        Y_necro : float
            The yield coefficient for growth on necromass. Has dimensions of 
            biomass produced per unit of substrate consumed.            
 
        """
        # Set the parameters
        self.lambda_nut_max = lambda_nut_max
        self.lambda_necro_max = lambda_necro_max
        self.Km_nut = Km_nut
        self.Km_necro = Km_necro
        self.Y_nut = Y_nut
        self.Y_necro = Y_necro
        self.gamma = gamma

    def compute_properties(self,
                           nut_conc : float,
                           M_necro_tot : float):
        """
        Computes the growth rates on nutrient and necromass

        Parameters
        ----------
        M_bio : float
            The biomass of the species.
        M_necro : float
            The necromass of the species.
        nut_conc : float
            The environmental nutrient concentration.
        M_nectro_tot: float 
            The total consumable necromass in the environment.
        delta : float
            The outflow dilution rate of the system.
        
        """
        self.nut_growth_rate = self.lambda_nut_max * nut_conc / (self.Km_nut + nut_conc)
        self.necro_growth_rate = self.lambda_necro_max * M_necro_tot / (self.Km_necro + M_necro_tot)
               

    def compute_derivatives(self,
                            M_bio : float,
                            M_necro : float,
                            nut_conc : float,
                            M_necro_tot : float,
                            delta : float,
                            ) ->  np.ndarray:
        """
        Computes the time derivatives of biomass and nutrient consumption
        per species.

        Parameters
        ----------
        M_bio : float
            The biomass of the species.
        M_necro : float
            The necromass of the species.
        nut_conc : float
            The environmental nutrient concentration.
        M_nectro_tot: float 
            The total consumable necromass in the environment.
        delta : float
            The outflow dilution rate of the system.
        
        Returns
        -------
        np.ndarray
            An array containing the time derivatives of species biomass, species 
            necromass, and nutrient consumption per species.

        """
        self.compute_properties(nut_conc, M_necro_tot)

        dM_bio_dt = self.nut_growth_rate * M_bio + self.necro_growth_rate * M_bio \
                - self.gamma * M_bio - delta * M_bio
        dM_necro_dt = self.gamma * M_bio - self.necro_growth_rate * M_bio - delta * M_necro
        dnut_dt = -self.nut_growth_rate * M_bio / self.Y_nut
        return np.array([dM_bio_dt, dM_necro_dt, dnut_dt])  


class Ecosystem():
    def __init__(self,
                 species : list[Species],
                 species_freqs : Union[None, list[float]] = None,
                 init_total_biomass : float = 1,
                 init_total_necromass : float = 0):
        """
        Initializes an ecosystem with a list of species. 

        Parameters
        ----------
        species : list
            A list of Species objects representing the species in the ecosystem.
       species_freqs : list[float] or None
            The frequency of each species in the ecosystem. If None, all species
            are assumed to have equal frequency.
        """
        self.species = species
        self.num_species = len(species)
        if species_freqs is None:
            self.species_freqs = np.ones(self.num_species)/self.num_species
        else:
            self.species_freqs = species_freqs

        # Set the initial masses 
        self._init = np.vstack((
            np.ones(self.num_species) * init_total_biomass * self.species_freqs,
            np.ones(self.num_species) * init_total_necromass * self.species_freqs,
            )).reshape((-1,),order='F')
        self._init = np.append(self._init, [init_total_necromass, 0])

    def _dynamical_system(self,
                          t : float,
                          pars : np.ndarray[float]):
        """
        Computes the mass derivatives for all species in the ecosystem 
        as well as the total nutrient dynamics. 

        Parameters
        ----------
        pars : np.ndarray[float]
            The bio- and necromasses of the species in the ecosystem, followed by the 
            total environmental necromass and total nutrient concentration.
        
        Returns
        -------
        derivs : np.ndarray[float]
            An array containing the time derivatives of the species biomass, 
            species necromass, total necromass, and  nutrient consumption for 
            all species in the ecosystem.
        """

        # Unpack the masses
        biomass = pars[:-2][::2]
        necromass = pars[:-2][1::2]
        
        # Compute the total nutrient sources
        nut_tot = pars[-1]
        necro_tot = pars[-2]

        # Iterate through each species and compute the derivatives
        derivs = np.zeros_like(pars)
        for i, s in enumerate(self.species):
            _derivs = s.compute_derivatives(biomass[i], necromass[i], nut_tot, necro_tot, self.delta)
            derivs[i*2:i*2+2] = _derivs[:-1]

            # Keep track of total nutrient and total necromass consumption
            derivs[-1] += _derivs[-1]
            derivs[-2] += _derivs[1]

        # Account for the inflow feedstock
        if self.feed_freq == -1:
            derivs[-1] += self.feed_conc * self.delta

        # Account for dilutive loss         
        derivs[-2:] -= self.delta * pars[-2:]
        return derivs

    def _unpack_soln(self,
                     time_shift : float = 0) -> list[DataFrame]:
        """
        Unpacks the solution from the solver into two DataFrames -- one for
        the environment and one for the species. 

        Parameters
        ----------
        time_shift : float
            For serial integrations, the time shift to apply to the time.

        Returns
        -------
        [species_df, bulk_df] : list[DataFrame]
            A list containing the time and mass trajectories of all species 
            and the total nutrient concentration in the ecosystem. 
        """
        result = self._last_soln.y
        time_dim = self._last_soln.t
        biomasses = result[:-2][::2]
        necromasses = result[:-2][1::2]

        # Determine total mass to compute frequencies
        tot_mass = np.sum(biomasses, axis=0)

        # Instantiate the species DataFrame
        dfs = [] 
        for i in range(self.num_species): 
            _df = pd.DataFrame(np.array([biomasses[i], necromasses[i]]).T,
                               columns=['M_bio', 'M_necro'])
            # Populate with species information 
            _df['species_idx'] = i + 1
            _df['mass_frequency'] = biomasses[i] / tot_mass
            _df['lambda_inst_nut'] = self.species[i].nut_growth_rate
            _df['lambda_max_nut'] = self.species[i].lambda_nut_max
            _df['lambda_inst_necro'] = self.species[i].necro_growth_rate
            _df['lambda_max_necro'] = self.species[i].lambda_necro_max
            _df['Km_nut'] = self.species[i].Km_nut
            _df['Km_necro'] = self.species[i].Km_necro
            _df['gamma'] = self.species[i].gamma
            _df['Y_nut'] = self.species[i].Y_nut
            _df['Y_necro'] = self.species[i].Y_necro

            # Add time dimension
            _df['time'] = time_dim + time_shift
            dfs.append(_df)
        species_df = pd.concat(dfs, sort=False)
         
        # Convert the bulk results into a DataFrame
        bulk_df = pd.DataFrame(result[-2:].T, columns=['M_necro_tot', 'M_nut'])
        bulk_df['M_bio_tot'] = tot_mass
        bulk_df['time'] = time_dim + time_shift
        return [species_df, bulk_df]


    def grow(self, 
             lifetime : float,
             feed_conc: float = 1,
             feed_freq: float = -1,
             delta : float = 0.1,
             dt : float = 0.1,
             solver_kwargs : dict = {'method': 'LSODA'},
             verbose : bool = True,
             ) -> list[DataFrame]:
        """
        Numerically integrates the ecosystem over the provided time span, 
        returning the time and mass trajectories of all species at a spacing 
        of dt.

        Parameters
        ----------
        lifetime : list[float]
            The length of time to integrate the ecosystem.
        feed_conc : float
            The concentration of the nutrient in the feedstock.
        feed_freq: float
            The time frequency at which feedstock is added as an impulse to the 
            ecosystem. If set to -1, feedstock concentration is flowed in at a
            rate delta.  
        delta : float
            The outflow dilution rate of the ecosystem.
        dt : float
            The time spacing at which to return the mass trajectories.
        solver_kwargs : dict
            Additional keyword arguments to pass to the solver.
        verbose : bool
            Whether to display a progress bar during the integration. Default is
            True

        Returns
        -------
        [species_df, bulk_df] : list[DataFrame]
            A list containing the time and mass trajectories of all species 
            and the total nutrient concentration in the ecosystem. 
        
        """
        # Determine the time range
        self.delta = delta
        self.feed_conc = feed_conc
        self.feed_freq = feed_freq

        # Add ithe initial nutrient concentration
        self._init[-1] = self.feed_conc

        # If feedstock is added as an impulse, set the the time ranges
        if self.feed_freq != -1:
            interval = self.feed_freq**-1
            num_integrations = int(np.floor(lifetime/interval))
            spans = [[i*interval, interval*(i + 1) -dt] for i in range(num_integrations)]
            if lifetime%interval != 0:
                spans.append([num_integrations*interval, lifetime])
        else:
            spans = [[0, lifetime]]
        # Set the iterator based on verbosity
        if verbose:
            if len(spans) == 1:
                print("Growing ecosystem...")
                iterator = spans
            else:     
                iterator = tqdm(spans, desc="Growing ecosystem...")
        else:
            iterator = spans

        # Iterate through each time span and perform the integration 
        out = [[], []]
        for i, _span in enumerate(iterator):
            # Set the initial conditions
            if i == 0:
                p0 = self._init
            else:
                p0 = self._last_soln.y[:,-1] 

            # Set the nutrient concentration as an impulse 
            p0[-1] = self.feed_conc

            # Define the actual time window and solve the system
            interval = _span[1] - _span[0] 

            t_eval = np.arange(0,  interval, dt)
            self._last_soln = solve_ivp(self._dynamical_system,[0, interval], p0, t_eval=t_eval,
                                        **solver_kwargs)
            # Unpack the results into dataframes, keep track of the integration window,
            # and return
            dfs = self._unpack_soln(time_shift=_span[0])
            for j, d in enumerate(dfs):
                d['integration_window'] = i+1
                out[j].append(d)
        if verbose:
            print("done!")
        if len(spans) == 1:
            return [out[0][0], out[1][0]]
        else:
            return [pd.concat(out[0], sort=False), pd.concat(out[1], sort=False)]


