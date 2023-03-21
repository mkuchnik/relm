import itertools

import matplotlib.pyplot as plt
import seaborn as sns


def set_plot_settings(color_palette="Set1", n_colors=10, font_scale=None):
    """
    Python settings for uniform plots
    """
    rc_params = {
        "backend": "pdf",
        "xtick.bottom": True,
        "ytick.left": True,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        #"text.usetex": True,
        #"font.family": "serif"
    }
    plt.rcParams.update(rc_params)
    sns.set_context("talk", font_scale=font_scale)
    #sns.set(font_scale=1.1)
    sns.despine(left=True)
    sns_style = {
        "xtick.major.size": 8,
        "ytick.major.size": 8,
        "axes.grid": True,
    }
    sns.set_style("ticks", sns_style)
    cmap = sns.color_palette(color_palette, n_colors)
    sns.set_palette(cmap)
    return cmap

def get_markers(n_markers):
    valid_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    marker_palette = itertools.cycle(valid_markers)
    markers = [next(marker_palette) for i in range(n_markers)]
    return markers

