from typing import Optional, Dict

import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt

from quickstats.plots import General2DPlot
from quickstats.utils.common_utils import combine_dict

METRIC_LABEL_MAP = {
    'log_loss' : 'Logloss',
    'AUC'      : 'AUC'
}

FEATURE_LABEL_MAP = {
    'high_level': 'High Level',
    'low_level': 'Low Level'
}

def plot_metric_landscape(df, metric:str='log_loss', m1:Optional[float]=None, m2:Optional[float]=None,
                          title:Optional[str]=None, cmap:str='viridis', highlight_color:str='hh:darkpink',
                          highlight_marker:str='*', markersize:float=20, fontsize:float=20, title_pad:float=30,
                          min_mass:float=0, max_mass:float=600, normalize:bool=False, vmin:float=None,
                          vmax:float=None, xlabel:str='$m_1^{WS}$ [GeV]', ylabel:str='$m_2^{WS}$ [GeV]', zlabel:str=None,
                          plot_styles:Optional[Dict]=None, text_config:Optional[Dict]=None):
    styles = {
        'pcolormesh': {
            'cmap': cmap,
            'rasterized': True
        },
        'title': {
            'fontsize': fontsize,
            'loc': 'center',
            'pad': title_pad
        },
        'xlabel': {
            'fontsize': fontsize
        },
        'ylabel': {
            'fontsize': fontsize
        }
    }
    styles = combine_dict(styles, plot_styles)
    highlight_styles = {
        'linewidth' : 0,
        'marker' : highlight_marker,
        'markersize' : markersize,
        'color' : highlight_color,
        'markeredgecolor' : 'black'
    }
    plotter = General2DPlot(df, styles=styles)
    if normalize:
        transform = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)) 
    else:
        transform = None
    if (vmin is not None) and (vmax is not None):
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = None
    metric_label = METRIC_LABEL_MAP.get(metric, metric)
    ax = plotter.draw('m1', 'm2', metric, xmin=min_mass, xmax=max_mass,
                      ymin=min_mass, ymax=max_mass, norm=norm,
                      xlabel=xlabel, ylabel=ylabel, zlabel=metric_label,
                      title=title, transform=transform)
    if text_config is not None:
        plotter.draw_text(ax, **text_config)
    if (m1 is not None) and (m2 is not None):
        ax.plot(float(m1), float(m2), **highlight_styles)
        ax.plot(float(m2), float(m1), **highlight_styles)
    return ax