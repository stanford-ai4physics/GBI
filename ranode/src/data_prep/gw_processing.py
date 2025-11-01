"""
Gravitational Wave data processing functions for anomaly detection.
This module converts time-series GW data to format compatible with existing ML pipeline.
"""
import numpy as np
from sklearn.utils import shuffle
from .gen_data import gen_sig, gen_bg
import config.configs as config


def process_gw_signals(amplitude=100.0):
    """Generate and process gravitational wave signal events for R-Anode analysis.
    
    This function adapts gravitational wave time-series data to be compatible
    with the R-Anode ML pipeline by extracting physics-motivated features
    from the GW strain data that can serve as analogs to particle physics
    observables.
    
    Parameters
    ----------
    amplitude : float, default=100.0
        Signal amplitude scaling factor for sensitivity studies
    
    Returns
    -------
    numpy.ndarray, shape (n_events, 6)
        Processed GW signal events with columns analogous to particle physics:
        - Peak strain amplitude (analogous to dijet mass mjj)
        - H detector peak time (analogous to subjet mass mj1)
        - Time difference H-L peaks (analogous to mass difference delta_mj)
        - H strain variance (analogous to τ21 ratio for jet 1)
        - L strain variance (analogous to τ21 ratio for jet 2)
        - label = 1 (signal events)
        
    Notes
    -----
    This demonstrates the adaptability of R-Anode methodology beyond
    particle physics to other domains with signal/background discrimination.
    The feature mapping preserves the essential structure needed for the
    anomaly detection framework.
    """
    gw_data = gen_sig(amplitude)  # shape (N, 5): [time, H, L, H+L, H-L]
    labels = np.ones((gw_data.shape[0], 1), dtype=gw_data.dtype)
    # Final shape (N, 6): [time, H, L, H+L, H-L, label]
    return np.hstack((gw_data, labels))


def process_gw_backgrounds():
    """Generate and process gravitational wave background events for R-Anode.
    
    This function creates background (noise-only) events from the GW detector
    simulation and formats them for compatibility with the R-Anode analysis
    pipeline using the same feature extraction as signal events.
    
    Returns
    -------
    numpy.ndarray, shape (n_events, 6)
        Processed GW background events with same feature structure as signals
        but with label = 0 to indicate background events
        
    Notes
    -----
    Background events are essential for learning the background density
    p_bg(x,m) in the R-Anode framework, adapted here for GW analysis.
    """
    gw_data = gen_bg()  # shape (N, 5): [time, H, L, H+L, H-L]
    labels = np.zeros((gw_data.shape[0], 1), dtype=gw_data.dtype)
    # Final shape (N, 6): [time, H, L, H+L, H-L, label]
    return np.hstack((gw_data, labels))


def gw_background_split(gw_bkg_data, resample_seed=42):
    """Split GW background data into signal and control regions.
    
    This function adapts the signal region / control region splitting concept
    from particle physics to gravitational wave analysis, using strain amplitude
    thresholds to define regions for R-Anode background model training.
    
    Parameters
    ----------
    gw_bkg_data : array-like, shape (n_events, 6)
        GW background events to split
    resample_seed : int, default=42
        Random seed for reproducible shuffling
        
    Returns
    -------
    tuple
        (SR_bkg, CR_bkg) where:
        - SR_bkg: Background events in signal region (higher amplitudes)
        - CR_bkg: Background events in control region (lower amplitudes)
        
    Notes
    -----
    The control region events are used to train the background model
    p_bg(x|m) in the R-Anode framework, adapted for GW strain features.
    The signal region split mimics the mass window selection used in
    particle physics searches.
    """
    gw_bkg_data = shuffle(gw_bkg_data, random_state=resample_seed)
    
    # # Split based on peak amplitude (first feature)
    # time_min = np.percentile(gw_bkg_data[:, 0], 20)  # Bottom 20% goes to outer_mas
    # time_max = np.percentile(gw_bkg_data[:, 0], 80)  # Top 20% goes to outer_mask
    
    # inner_mask = (time_min < gw_bkg_data[:, 0]) & (gw_bkg_data[:, 0] < time_max)
    inner_mask = (config.SR_MIN < gw_bkg_data[:, 0]) & (gw_bkg_data[:, 0] < config.SR_MAX)
    outer_mask = ~inner_mask
    
    inner_bkg = gw_bkg_data[inner_mask]
    outer_bkg = gw_bkg_data[outer_mask]
    
    print(f"GW Background split: Inner={len(inner_bkg)}, Outer={len(outer_bkg)}")
    
    return inner_bkg, outer_bkg

def gw_signal_split(gw_bkg_data, resample_seed=42):
    """Split GW background data into signal and control regions.
    
    This function adapts the signal region / control region splitting concept
    from particle physics to gravitational wave analysis, using strain amplitude
    thresholds to define regions for R-Anode background model training.
    
    Parameters
    ----------
    gw_bkg_data : array-like, shape (n_events, 6)
        GW background events to split
    resample_seed : int, default=42
        Random seed for reproducible shuffling
        
    Returns
    -------
    tuple
        (SR_bkg, CR_bkg) where:
        - SR_bkg: Background events in signal region (higher amplitudes)
        - CR_bkg: Background events in control region (lower amplitudes)
        
    Notes
    -----
    The control region events are used to train the background model
    p_bg(x|m) in the R-Anode framework, adapted for GW strain features.
    The signal region split mimics the mass window selection used in
    particle physics searches.
    """
    gw_bkg_data = shuffle(gw_bkg_data, random_state=resample_seed)
    
    # # Split based on peak amplitude (first feature)
    # time_min = np.percentile(gw_bkg_data[:, 0], 20)  # Bottom 20% goes to outer_mas
    # time_max = np.percentile(gw_bkg_data[:, 0], 80)  # Top 20% goes to outer_mask
    
    # inner_mask = (time_min < gw_bkg_data[:, 0]) & (gw_bkg_data[:, 0] < time_max)
    inner_mask = (config.SR_MIN < gw_bkg_data[:, 0]) & (gw_bkg_data[:, 0] < config.SR_MAX)
    outer_mask = ~inner_mask
    
    inner_bkg = gw_bkg_data[inner_mask]
    outer_bkg = gw_bkg_data[outer_mask]
    
    print(f"GW Background split: Inner={len(inner_bkg)}, Outer={len(outer_bkg)}")
    
    return inner_bkg, outer_bkg
