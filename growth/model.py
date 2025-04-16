import numpy as np 
from scipy.integrate import solve_ivp
from pandas.core.frame import DataFrame
from .callbacks import extinction_event, fixation_event
import pandas as pd
from tqdm import tqdm
from typing import Union
from dataclasses import dataclass, field

@dataclass
class Species():
    """
    Base class for a simple self replicator following Monod growth kinetics.
    
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
    label : int
        The species label.
        
    Attributes
    ----------
    growth_rate : float
        Current growth rate based on nutrient availability.
    effective_growth_rate : float
        Growth rate minus death rate.
    extinct : bool
        Flag indicating if the species is extinct.
    fixed : bool
        Flag indicating if the species is fixed in the population.
    """

    # User-specified parameters
    lambda_max: float = 1.0
    Km: float = 0.01
    gamma: float = 0.1
    Y: float = 0.1
    label : Union[int, str] = 0

    # Internal state variables.
    growth_rate: float = field(default=0.0, init=False)
    effective_growth_rate: float = field(default=0.0, init=False)
    extinct: bool = field(default=False, init=False)
    extinction_time: bool = field(default=float('inf'), init=False)
    fixed: bool = field(default=False, init=False) 

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.lambda_max < 0:
            raise ValueError("lambda_max must be non-negative.")
        if self.Km <= 0:
            raise ValueError("Km must be positive.")
        if self.gamma < 0:
            raise ValueError("gamma must be non-negative.")
        if self.Y <= 0:
            raise ValueError("Y must be positive.")
    def _compute_growth_rate(self,
                            c_nt: Union[float, list[float]]) -> list[float]:
        """
        Computes and returns the instantaneous and effective growth rate at 
        a given nutrient concentration.

        Parameters 
        -----------
        c_nt : float
            The environmental nutrient concentration. 
        
        Returns
        -------
        [growth_rate, eff_growth_rate] : [float, float]
            The instantaneous and effective growth rate. 
        """
        c_nt = np.maximum(c_nt, 0)
        growth_rate = self.lambda_max * c_nt / (self.Km + c_nt)
        eff_growth_rate = growth_rate - self.gamma
        return [growth_rate, eff_growth_rate]
    def update_growth_rate(self, 
               c_nt: float) -> None:
        """
        Update the species' state based on nutrient concentration.
        
        Parameters
        ----------
        c_nt : float
            The environmental nutrient concentration.
        """
        c_nt = np.maximum(c_nt, 0)  # Ensure physical concentration
        self.growth_rate, self.effective_growth_rate = self._compute_growth_rate(c_nt)
        
    def compute_derivatives(self, 
                            M    : float, 
                            c_nt : float, 
                            delta: float) -> np.ndarray:
        """
        Computes the time derivatives of biomass and nutrient consumption.
        
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
        if self.extinct:
            self.update_growth_rate(0)
            return np.zeros(2)

        self.update_growth_rate(c_nt) 
        dM_dt = M * (self.effective_growth_rate - delta)
        dc_nt_dt = -self.growth_rate * M / self.Y

        return np.array([dM_dt, dc_nt_dt])

@dataclass
class Ecosystem:
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
        
    Attributes
    ----------
    species : list
        The list of Species objects in the ecosystem.
    species_freqs : list[float]
        The frequency of each species in the ecosystem.
    init_total_biomass : float
        The initial sum-total biomass of all species in the system.
    num_species : int
        The number of species in the ecosystem.
    _init : np.ndarray
        The initial biomasses of each species followed by a placeholder for nutrient.
    """
    species: list[Species]
    species_freqs: Union[None, list[float]] = None
    init_total_biomass: float = 1
    
    # Fields that are computed from the inputs
    num_species: int = field(init=False)
    _init: np.ndarray = field(init=False)
    
    def __post_init__(self):
        """
        Initializes computed fields after the dataclass initialization.
        """
        self.num_species = len(self.species)

        # Ensure that each species has a unique label
        labels = set() 
        for s in self.species:
            if s.label in labels:
                raise ValueError(f'Species label {s.label} is not unique!')
                
        # If initial frequencies not supplied, assume they are equally abundant
        if self.species_freqs is None:
            self.species_freqs = np.ones(self.num_species)/self.num_species
            
        # Set the initial masses and add placeholder for nutrient
        self._init = np.ones(self.num_species) * self.init_total_biomass * self.species_freqs
        self._init = np.append(self._init, [0])
       
    def _dynamical_system(self, 
                          t   : float, 
                          pars: np.ndarray, 
                          args: dict) -> np.ndarray:
        """
        Computes the mass derivatives for all species in the ecosystem 
        as well as the total nutrient dynamics. 

        Parameters
        ----------
        t : float
            Current time (unused but required by solve_ivp).
        pars : np.ndarray
            The biomasses of the species followed by nutrient concentration.
        args : dict
            Dictionary of additional parameters.
        
        Returns
        -------
        derivs : np.ndarray
            An array containing the time derivatives.
        """
        # Extract biomasses and nutrient concentration
        biomass = pars[:-1]
        c_nt = max(pars[-1], 0) if pars[-1] >= args['nut_thresh'] else 0
        biomass_thresh = args['biomass_thresh'] 

        # Initialize derivatives array
        derivs = np.zeros_like(pars)
         
        # Compute derivatives for each species
        for i, s in enumerate(self.species):
            if s.extinct:
                biomass[i] = 0  # Force biomass to zero for extinct species
                derivs[i] = 0
                continue
            
            # Check for new extinction events
            if biomass[i] < biomass_thresh:
                s.extinct = True
                s.extinction_time = t
                biomass[i] = 0
                derivs[i] = 0 
                continue

            _derivs = s.compute_derivatives(biomass[i], c_nt, self.delta)
            derivs[i] = _derivs[0]        # Biomass derivative
            derivs[-1] += _derivs[-1]     # Nutrient consumption
            
        # Add nutrient inflow (continuous)
        if self.feed_freq == -1:
            derivs[-1] += self.feed_conc * self.delta
            
        # Account for dilutive loss
        derivs[-1] -= self.delta * c_nt
        
        return derivs

    def _unpack_soln(self, 
                     time_shift: float = 0) -> list[DataFrame]:
        """
        Unpacks the solution from the solver into DataFrames.

        Parameters
        ----------
        time_shift : float
            For serial integrations, the time shift to apply to the time.

        Returns
        -------
        [species_df, bulk_df] : list[DataFrame]
            DataFrames containing species and environment data.
        """
        # Extract solution data
        result = self._last_soln.y
        time_dim = self._last_soln.t
        biomasses = result[:self.num_species]
        c_nt = np.maximum(result[-1], 0) # Ensure physical nutrient concentration
        tot_mass = np.sum(biomasses, axis=0) 

        # Create species dataframes
        dfs = []
        for i, s in enumerate(self.species):
            # Update species state based on final nutrient concentration
            growth_rate, eff_growth_rate = s._compute_growth_rate(c_nt)
            
            # Create dataframe with all species properties
            _df = pd.DataFrame({
                'M': biomasses[i],
                'frequency': biomasses[i] / tot_mass,
                'time': time_dim + time_shift,
                'lambda_max': s.lambda_max,
                'gamma': s.gamma,
                'growth_rate': growth_rate,
                'effective_growth_rate': eff_growth_rate,
                'extinct': s.extinct,
                'label': s.label,
                'Km': s.Km,
                'Y': s.Y
            })

            if s.extinct:
                _df.loc[_df['time'] <= s.extinction_time, 'extinct'] = False
            dfs.append(_df)
        
        # Combine all species data
        species_df = pd.concat(dfs, sort=False)
        
        # Create environment dataframe
        bulk_df = pd.DataFrame({
            'M_nut': c_nt,
            'M_tot': tot_mass,
            'time': time_dim + time_shift
        })
        
        return [species_df, bulk_df]

    def grow(self, 
            lifetime      : float,
            feed_conc     : float = 1,
            feed_freq     : float = -1,
            delta         : float = 0.1,
            dt            : float = 0.01,
            verbose       : bool = True,
            biomass_thresh: float = 0,
            nut_thresh    : float = 0, 
            term_event    : dict = {'type': None},
            solver_kwargs : dict = {'method': 'LSODA'}) -> list[DataFrame]:
        """
        Numerically integrates the ecosystem over the provided time span.

        Parameters
        ----------
        lifetime : float
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
            The timestep at which to integrate.
        verbose : bool
            Whether to display a progress bar during the integration.
        biomass_thresh : float
            The threshold below which the biomass for each species is set to zero.
        nut_thresh : float
            The threshold below which the nutrient concentration is set to zero.
        term_event : dict
            A dictionary of termination events to pass to the solver.
        solver_kwargs : dict
            Additional keyword arguments to pass to the solver.
            
        Returns
        -------
        [species_df, bulk_df] : list[DataFrame]
            A list containing the time and mass trajectories.
        """
        # Reset all species as not extinct
        for s in self.species:
            s.extinct = False

        # Store simulation parameters as instance variables
        self.delta = delta
        self.feed_conc = feed_conc
        self.feed_freq = feed_freq
        self.terminated = False
        self._init[-1] = feed_conc  # Set initial nutrient concentration
        
        # Determine time spans based on feeding strategy
        spans = self._set_time_spans(lifetime, dt)
        
        # Setup iterator with optional progress bar
        iterator = self._set_iterator(spans, verbose)
        
        # Setup arguments for the integrator
        args, events = self._set_events(biomass_thresh, nut_thresh, term_event, verbose)
        
        # Run the integration
        return self._integrate(iterator, dt, args, events, solver_kwargs, verbose)
    
    def _set_time_spans(self, 
                        lifetime: float, 
                        dt      : float) -> list:
        """Set up time spans for integration based on feeding frequency."""
        if self.feed_freq >= 0:
            if self.feed_freq > 0:
                interval = self.feed_freq**-1
                if np.floor(interval) > lifetime:
                    raise ValueError("Feed frequency is longer than integration time")
                if interval < max([dt, 0.01]):
                    raise ValueError("Feed frequency is shorter than integration step")
            else:  # feed_freq == 0
                interval = lifetime + 0.01
                
            # Calculate spans for pulsed feeding
            num_integrations = int(np.floor(lifetime/interval))
            spans = [[i*interval, interval*(i+1)-dt] for i in range(num_integrations)]
            
            # Add final span if needed
            if lifetime % interval != 0:
                spans.append([num_integrations*interval, lifetime])
        else:
            # Continuous feeding - single integration span
            spans = [[0, lifetime]]
            
        return spans
    
    def _set_iterator(self, 
                      spans: list, 
                      verbose: bool) -> list:
        """Set up iterator with optional progress bar."""
        if verbose:
            if len(spans) == 1:
                print("Growing ecosystem...")
                return spans
            else:
                return tqdm(spans, desc="Growing ecosystem...")
        else:
            return spans
    
    def _set_events(self, 
                    biomass_thresh: float, 
                    nut_thresh    : float, 
                    term_event    : dict, 
                    verbose       : bool) -> tuple:
        """Set up termination events and arguments."""
        args = {
            'nut_thresh': nut_thresh,
            'biomass_thresh': biomass_thresh,
            'species': self.species
        }
        
        events = []
        if term_event['type'] == 'extinction':
            if verbose:
                print("Watching for extinction events...")
                
            # Configure extinction event
            extinction_event.terminal = True
            extinction_event.direction = 1
            
            # Set thresholds
            if 'freq_thresh' in term_event:
                args['freq_thresh'] = term_event['freq_thresh']
            if 'biomass_thresh' in term_event:
                args['biomass_thresh'] = term_event['biomass_thresh']
            if ('freq_thresh' not in args) and ('biomass_thresh' not in args):
                raise ValueError("Must provide threshold for extinction callback")
                
            events.append(extinction_event)
            
        elif term_event['type'] == 'fixation':
            if verbose:
                print("Watching for a fixation event...")
                
            # Set fixation threshold
            args['freq_thresh'] = term_event['freq_thresh']
            
            # Configure fixation event
            fixation_event.terminal = True
            fixation_event.direction = 1
            events.append(fixation_event)
            
        return args, events
    
    def _integrate(self, 
                    iterator, 
                    dt           : float, 
                    args         : dict, 
                    events       : list, 
                    solver_kwargs: dict, 
                    verbose      : bool) -> list:
        """Run the integration over all time spans."""
        out = [[], []] 

        for i, _span in enumerate(iterator):
            # Set initial conditions
            p0 = self._init if i == 0 else self._last_soln.y[:,-1]
            
            # Set nutrient concentration as an impulse
            if self.feed_freq >= 0:
                p0[-1] = self.feed_conc
                
            # Define time window
            interval = _span[1] - _span[0]
            
            # Solve the system
            if dt > 0:
                t_eval = np.arange(0, interval, dt)
                self._last_soln = solve_ivp(
                    self._dynamical_system, [0, interval], p0,
                    t_eval=t_eval, args=(args,), events=events, 
                    **solver_kwargs
                )
            else:
                self._last_soln = solve_ivp(
                    self._dynamical_system, [0, interval], p0,
                    args=(args,), events=events, **solver_kwargs
                )
                
            # Process results
            dfs = self._unpack_soln(time_shift=_span[0])
            for j, d in enumerate(dfs):
                d['integration_window'] = i+1
                out[j].append(d)
                
            # Check if terminated by event
            if self._last_soln.status == 1:
                self.terminated = True
                if verbose:
                    event_time = _span[0] + self._last_soln.t[-1]
                    print(f'An {args["type"]} event occurred at t = {event_time:0.1f}')
                break
                
        if verbose:
            print("done!")
            
        # Return results
        if len(out[0]) == 1:
            return [out[0][0], out[1][0]]
        else:
            return [pd.concat(out[0], sort=False), pd.concat(out[1], sort=False)]
