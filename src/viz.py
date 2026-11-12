"""Estilo común de gráficos: paleta categórica fija (validada para daltonismo) por clase."""
import matplotlib.pyplot as plt
import seaborn as sns

from .config import CLASSES

# Orden fijo: el color sigue a la clase, nunca a su rango
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CLASS_COLORS = dict(zip(CLASSES, PALETTE))
SEQ_CMAP = "Blues"      # magnitud: un solo tono
DIV_CMAP = "RdBu_r"     # polaridad: dos tonos + neutro


def set_style():
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#8a8984", "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.labelcolor": "#52514e",
        "xtick.color": "#52514e", "ytick.color": "#52514e", "legend.frameon": False,
    })
