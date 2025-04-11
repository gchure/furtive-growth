import numpy as np 
from scipy.integrate import solve_ivp
from pandas.core.frame import DataFrame
from .callbacks import extinction_event, fixation_event
import pandas as pd
from tqdm import tqdm
from typing import Union

class Species():
    """
    Base class for a simple self replicator following Monod growth kinetics.
    """
    def __init__(self,
                 lambda_max : float = 1.0,
                 Km         : float = 0.01,
                 gamma      : float = 0.1,
                 Y          : float = 0.1,
                 ) -> None:
        """ 
        Initializes a self-replicating species with Monod growth kinetics.

        Parameters
        ----------
        lambda_max : float
            The maximum growth rate of the species on nutrients.
        Km : float
            The Monod constant for growth on nutrients. 
        gamma : float
            The death rate of the species.
        Y : float
            The yield coefficient for growth on nutrients. Has dimensions of 
            biomass produced per unit of substrate consumed.            
        """
        # Set the parameters
        self.lambda_max = lambda_max
        self.Km = Km
        self.Y = Y
        self.gamma = gamma

        # Set parameters to keep track of the individual species
        self.extinct = False 
        self.fixed = False

    def compute_growth_rate(self,
                           c_nt  : float,
                           ) -> None:
        """
        Computes the growth rates on nutrient and necromass

        Parameters
        ----------
        c_nt : float
            The environmental nutrient concentration.
        """

        # Ensure that the nutrient concentration does not become unphysical
        c_nt = max(c_nt, 0)
        self.growth_rate = self.lambda_max * c_nt / (self.Km + c_nt)
               
    def compute_derivatives(self,
                            M     : float,   
                            c_nt  : float, 
                            delta : float
                            ) ->  np.ndarray:
        """
        Computes the time derivatives of biomass and nutrient consumption
        per species.

        Parameters
        ----------
        M : float
            The biomass of the species.
        c_nt: float
            The environmental nutrient concentration.
        delta : float
            The outflow dilution rate of the system.
        
        Returns
        -------
        np.ndarray
            An array containing the time derivatives of species biomass and
            nutrient consumption per species.

        """
        self.compute_growth_rate(c_nt)

        # Compute and return the individual derivatives
        dM_dt = M * (self.growth_rate - self.gamma - delta)
        dc_nt_dt = -self.growth_rate * M / self.Y
        return np.array([dM_dt, dc_nt_dt])  

class Ecosystem():
    def __init__(self,
                 species              : list[Species],
                 species_freqs        : Union[None, list[float]] = None,
                 init_total_biomass   : float = 1,
                 ) -> None:
        """
        Initializes an ecosystem with a list of species. 

        Parameters
        ----------
        species : list
            A list of Species objects representing the species in the ecosystem.
        species_freqs : list[float] or None
            The frequency of each species in the ecosystem. If None, all species
            are assumed to have equal frequency.
        init_total_biomass  : float
            The initial sum-total biomass of all species in the system.
        """
        self.species = species
        self.num_species = len(species)

        # If initial frequencies not supplied,assume they are equally abundant.
        if species_freqs is None:
            self.species_freqs = np.ones(self.num_species)/self.num_species
        else:  # Otherwise, use the user-provided frequencies
            self.species_freqs = species_freqs

        # Set the initial masses 
        self._init = np.ones(self.num_species) * init_total_biomass * species_freqs
        self._init = np.append(self._init, [0])

    def _dynamical_system(self,
                          t : float,
                          pars : np.ndarray[float],
                          args : dict) -> np.ndarray[float]:
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
        biomass = pars[:-1] 

        # Compute the total nutrient sources
        pars[-1] = 0 if pars[-1] < args['nut_thresh'] else pars[-1]  
        c_nt = pars[-1]

        # Iterate through each species and compute the derivatives
        derivs = np.zeros_like(pars)
        for i, s in enumerate(self.species):
            # Evaluate derivs.
            _derivs = s.compute_derivatives(biomass[i], c_nt, self.delta)

            # Update biomass 
            derivs[i] = _derivs[0]
            
            # Update nutrient concentration
            derivs[-1] += _derivs[-1]

        # Account for the inflow feedstock
        if self.feed_freq == -1:
            derivs[-1] += self.feed_conc * self.delta

        # Account for dilutive loss         
        derivs[-1] -= self.delta * pars[-1]
        return derivs

    def _unpack_soln(self,
                     time_shift : float = 0
                     ) -> list[DataFrame]:
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
        biomasses = result[:-1]
        result[-1] = max(result[-1], 0)
        # Determine total mass to compute frequencies
        tot_mass = np.sum(biomasses, axis=0)

        # Instantiate the species DataFrame
        dfs = [] 
        for i in range(self.num_species): 
            s = self.species[i]
            s.compute_growth_rate(result[-1]) 
            _df = pd.DataFrame(
                {'M': biomasses[i],
                 'frequency': biomasses[i] / tot_mass,
                 'growth_rate': self.species[i].growth_rate,
                 'growth_rate_max': self.species[i].lambda_max,
                 'gamma': self.species[i].gamma,
                 'growth_rate_eff': self.species[i].growth_rate - self.species[i].gamma,
                 'Km': self.species[i].Km,
                 'Y': self.species[i].Y,
                 'time': time_dim + time_shift
                },
                index=[0]
                )
            dfs.append(_df)
        species_df = pd.concat(dfs, sort=False)
         
        # Convert the bulk results into a DataFrame
        bulk_df = pd.DataFrame(result[-1].T, columns=['M_nut'])
        bulk_df['M_tot'] = tot_mass
        bulk_df['time'] = time_dim + time_shift
        return [species_df, bulk_df]


    def grow(self, 
             lifetime       : float,
             feed_conc      : float = 1,
             feed_freq      : float = -1,
             delta          : float = 0.1,
             dt             : float = 0.1,
             verbose        : bool = True,
             biomass_thresh : float = 1.0,
             nut_thresh     : float = 0, 
             term_event     : dict = {'type': None},
             solver_kwargs  : dict = {'method': 'LSODA'},
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
        verbose : bool
            Whether to display a progress bar during the integration. Default is
            True
        biomass_thresh : float
            The threshold below which the biomass is set to zero.
        nut_thresh : float
            The threshold below which the nutrient concentration is set to zero.
        term_event : dict
            A dictionary of termination events to pass to the solver. Only 
            acceptable responses are "extinction" and "fixation". Must also 
            provide the frequency threshold for the event. 
        solver_kwargs : dict
            Additional keyword arguments to pass to the solver.
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
        self.terminated = False

        # Add the initial nutrient concentration
        self._init[-1] = self.feed_conc

        # If feedstock is added as an impulse, set the the time ranges
        if self.feed_freq >= 0:
            if self.feed_freq > 0:
                interval = self.feed_freq**-1
                if np.floor(interval) > lifetime:
                    raise ValueError("The feed frequency is longer than the integration time")
                if interval < max([dt, 0.01]):
                    raise ValueError("The feed frequency is shorter than the integration time step")
            elif self.feed_freq == 0:
                interval = lifetime + 0.01
            num_integrations = int(np.floor(lifetime/interval))

            # FIXME: This is slow, but fine if you're not doing enormously long
            # integrations. 
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

        # Specify arguments to be fed to the integrator
        args = {'nut_thresh':nut_thresh,
                'biomass_thresh': biomass_thresh}

        # Determine if callbacks should be applied
        events = []
        if term_event['type'] == 'extinction':

            if verbose:
                print("Watching for extinction events...")

            # Set extinction callback attributes.
            extinction_event.terminal = True
            extinction_event.direction = 1

            if 'freq_thresh' in term_event:
                args['freq_thresh'] = term_event['freq_thresh']
            if 'biomass_thresh' in term_event:
                args['biomass_thresh'] = term_event['biomass_thresh']
            if ('freq_thresh' not in args) and ('biomass_thresh' not in args):
                raise ValueError("Must provide either 'freq_thresh' and 'biomass_thresh' for extinction callback")
            events.append(extinction_event)

        elif term_event['type'] == 'fixation':
            if verbose: 
                print("Watching for a fixation event...")
            args['freq_thresh'] = term_event['freq_thresh']

            # Set fixation callback attributes
            fixation_event.terminal = True
            fixation_event.direction = 1
            events.append(fixation_event)

        args['species'] = self.species


        # TODO: Allow a way to bypass returning the trajectories and 
        # just return the steady-state. May be a different method.
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
            if dt > 0:
                t_eval = np.arange(0,  interval, dt)
                self._last_soln = solve_ivp(self._dynamical_system,[0, interval], p0,
                                        t_eval=t_eval, args=(args,),events=events, 
                                        **solver_kwargs)
            else:
                self._last_soln = solve_ivp(self._dynamical_system,[0, interval], p0,
                                        args=(args,),events=events, 
                                        **solver_kwargs)

            # Unpack the results into dataframes, keep track of the integration window,
            # and return
            dfs = self._unpack_soln(time_shift=_span[0])
            for j, d in enumerate(dfs):
                d['integration_window'] = i+1
                out[j].append(d)
            if self._last_soln.status == 1:
                self.terminated = True
                if verbose:
                    print(f'An {term_event["type"]} event occurred at t = {_span[0] + self._last_soln.t[-1]:0.1f}')
                break

        if verbose:
            print("done!")
        if len(spans) == 1:
            return [out[0][0], out[1][0]]
        else:
            return [pd.concat(out[0], sort=False), pd.concat(out[1], sort=False)]


