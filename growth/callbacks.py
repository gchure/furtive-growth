import numpy as np 
from numpy import ndarray

def extinction_event(t     : float, 
                     params: ndarray,
                     args  : dict) -> int:
    """
    Callback for determining if the ecosystem has *any* extinction event. 
    
    Parameters
    ----------
    t : float 
        The current timestep. Only provided by the ODE solver.
    params : ndarray 
        A numpy array of ecosystem parameters as [biomasses, nutrient].
    args : dict 
        A dictionary with the following keys:
            'freq_thresh' : float (optional)
                The frequency threshold below which a species is considered extinct.
            'biomass_thresh' : float (optional)
                The biomass threshold below which a species is considered extinct.
            'species': list[Species]
                A list of species objects.
        
    Returns
    -------
    int 
        0 if there is an extinction event, -1 if there is no extinction event.
    """
    # Unpack the parameters and compute the frequency
    biomasses = params[:-1]  # All but last element (nutrient)
    total_biomass = np.sum(biomasses)
    freqs = biomasses / total_biomass if total_biomass > 0 else np.zeros_like(biomasses)
    
    # Initialize extinction flags
    freq_ext = np.zeros_like(biomasses, dtype=bool)
    biomass_ext = np.zeros_like(biomasses, dtype=bool)
    
    # Check frequency threshold if provided
    if 'freq_thresh' in args:
        freq_ext = freqs <= args['freq_thresh']
    
    # Check biomass threshold if provided
    if 'biomass_thresh' in args:
        biomass_ext = biomasses <= args['biomass_thresh']
    
    # Check if any species is extinct
    if np.any(freq_ext) or np.any(biomass_ext):
        # Mark extinct species
        for i, (f_ext, b_ext) in enumerate(zip(freq_ext, biomass_ext)):
            if f_ext or b_ext:
                args['species'][i].extinct = True
        return 0
    else:
        return -1

def fixation_event(t    : float, 
                  params: ndarray,
                  args  : dict) -> int:
    """
    Callback for determining if the ecosystem has met a fixation event threshold. 
    
    Parameters
    ----------
    t : float 
        The current timestep. Only provided by the ODE solver.
    params : ndarray 
        A numpy array of ecosystem parameters as [biomasses, nutrient].
    args : dict 
        A dictionary with the following keys:
            'freq_thresh' : float
                The frequency threshold above which a species is considered fixed.
            'species': list[Species]
                A list of species objects.
        
    Returns
    -------
    int 
        0 if there is a fixation event, 1 if there is no fixation event.
    """
    # Unpack the parameters and compute the frequency
    biomasses = params[:-1]  # All but last element (nutrient)
    total_biomass = np.sum(biomasses)
    freqs = biomasses / total_biomass if total_biomass > 0 else np.zeros_like(biomasses)
    
    # Check if any species has reached fixation threshold
    if np.any(freqs >= args['freq_thresh']):
        # Mark fixed species
        for i, f in enumerate(freqs):
            if f >= args['freq_thresh']:
                args['species'][i].fixed = True
        return 0
    else:
        return 1