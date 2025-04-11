import numpy as np 
from numpy import ndarray
def extinction_event(t : float, 
                     params : ndarray,
                     args : dict) -> int:
    """
    Callback for determining if the ecosystem has *any* extinction event. 

    Parameters
    ----------
    t : float 
        The current timestep. Only provided by the ODE solver.
    params : ndarray 
        A numpy array of ecosystem parameters as [[M_bio, M_necro] * N_s,
        M_necro_tot, M_nut].
    args : dict 
        A dictionary with the following keys:
            'extinction_threshold' : float 
                The mass frquency threshold below which a species is considered 
                extinct and the integration ends.
            'species': list[Species]
                A list of species objects.
        
    Returns
    -------
    int 
        0 if there is an extinction event, -1 if there is non extinction event.
        This is required as a zero crossing event for the root finding algorithm 
        in the ODE solver.
    """
    # Unpack the parameters and compute the frequency.
    biomasses = params[:-2][::2] 
    freqs = biomasses / np.sum(biomasses)
    if 'freq_thresh' in args:
        freq_ext = freqs <= args['freq_thresh']
    if 'biomass_thresh' in args:
        biomass_ext = biomasses <= args['biomass_thresh']
    if (freq_ext).any() or (biomass_ext).any():
        for i, _ in enumerate(freq_ext):
            args['species'][i].extinct = True
        return 0
    else:
        return -1


def fixation_event(t : float, 
                     params : ndarray,
                     args : dict) -> int:
    """
    Callback for determining if the ecosystem has met a fixation event threshold. 

    Parameters
    ----------
    t : float 
        The current timestep. Only provided by the ODE solver.
    params : ndarray 
        A numpy array of ecosystem parameters as [[M_bio, M_necro] * N_s,
        M_necro_tot, M_nut].
    args : dict 
        A dictionary with the following keys:
            'fixation_threshold' : float 
                The mass frquency threshold above which a species is considered 
                extinct and the integration ends.
            'species': list[Species]
                A list of species objects.
        
    Returns
    -------
    int 
        0 if there is a fixation event, -1 if there is non extinction event.
        This is required as a zero crossing event for the root finding algorithm 
        in the ODE solver.
    """

    # Unpack the parameters and compute the frequency.
    biomasses = params[:-2][::2] 
    freqs = biomasses / np.sum(biomasses)
    if (freqs >= args['freq_thresh']).any():
        for i, f in enumerate(freqs):
            if f >= args['freq_thresh']:
                args['species'][i].fixed = True
            return 0
    else:
        return 1