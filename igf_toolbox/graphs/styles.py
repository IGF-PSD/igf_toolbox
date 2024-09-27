# Importation des modules
# Modules de base
import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Modules graphiques
from cycler import cycler
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.font_manager import fontManager


# Fonction de définition de la chrte igf
def set_igf_style() -> None:
    """Sets the default style for all graphs according to igf chart"""

    # Installation de la police Cambria
    fontManager.addfont(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "Cambria.ttf")
    )
    # fontManager.addfont("Cambria.ttf")

    # Figure size
    plt.rcParams["figure.figsize"] = (15, 7)

    # Line plot styles
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["lines.markersize"] = 8

    # Axis labels and ticks
    plt.rcParams["font.family"] = "Cambria"
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16

    # Legend
    plt.rcParams["legend.fontsize"] = 16
    plt.rcParams["legend.title_fontsize"] = 16
    plt.rcParams["legend.framealpha"] = 0

    plt.rcParams["legend.loc"] = "upper center"
    # add bboc8to_anchor=(0.5, -0.15) and ncol=... params when calling plt.legend(bboc_to)

    # Remove top and right spines
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.left"] = True
    plt.rcParams["axes.spines.bottom"] = True

    # Set custom colormap
    plt.rcParams["axes.prop_cycle"] = cycler(
        "color", ["#096c45", "#737c24", "#d69a00", "#e17d18", "#9f0025", "#ae535c"]
    )


# Fonction définissant un ScalarMappable à partir de la ColorMap de l'IGF en la mettant à l'échelle du min et du max
def get_scalar_mappable(data: Union[pd.DataFrame, np.ndarray]) -> ScalarMappable:
    """
    This function creates a ScalarMappable object from the IGF ColorMap,
    scaling it according to the minimum and maximum values in the data.

    Parameters:
    data (numpy.ndarray): A numpy array containing the data to be mapped.

    Returns:
    scalar_mappable (matplotlib.cm.ScalarMappable): A ScalarMappable object
    that can be used to map data values to colors using the IGF ColorMap.

    The function first defines the IGF ColorMap using the LinearSegmentedColormap
    class from matplotlib. It then determines the minimum and maximum values
    in the data. These values are used to normalize the ColorMap using the
    Normalize class from matplotlib. Finally, a ScalarMappable object is
    created using the normalized ColorMap and returned by the function.
    """
    # Définition de la ColorMap de l'IGF
    cmap_igf = LinearSegmentedColormap.from_list(
        "charte", ["#096c45", "#737c24", "#d69a00", "#e17d18", "#9f0025"], N=256
    )

    # Définition des valeurs minimales et maximales des données
    value_min = data.min()
    value_max = data.max()
    # Normalisation des valeurs de la ColorMap
    norm = Normalize(vmin=value_min, vmax=value_max)
    # Création du ScalarMappable
    scalar_mappable = ScalarMappable(norm=norm, cmap=cmap_igf)

    return scalar_mappable
