"""Shared figure styling for every plot the project produces.

Importing this module selects a non-interactive matplotlib backend unless the
``AEROSONAR_SHOW_PLOTS`` environment variable is set. It must therefore be imported
before ``matplotlib.pyplot``: the backend cannot be changed afterwards, and a blocking
``plt.show()`` will stall a headless run.
"""
import os
from pathlib import Path

import matplotlib

if not os.environ.get("AEROSONAR_SHOW_PLOTS"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

STYLE = "seaborn-v0_8-muted"

#: Maps a semantic role to a colour so the same quantity is drawn the same way
#: in every figure.
COLORS = {
    "accuracy": "#5dcc5d",
    "f1": "#f14c4c",
    "precision": "#41a3db",
    "recall": "#ffa632",
    "train": "#41a3db",
    "val": "#fc6b03",
    "test": "#f14c4c",
    "reference": "#8a8a8a",
}

CONFUSION_CMAP = LinearSegmentedColormap.from_list("ambience", ["#f0f0f0", "#fc6b03"])


def new_figure(figsize=(7, 5)):
    """Create a figure with the project style applied.

    Use in place of ``plt.figure()`` so every plot picks up the shared style.

    Args:
        figsize: Figure size in inches, as ``(width, height)``.

    Returns:
        matplotlib.figure.Figure: The new figure.
    """
    plt.style.use(STYLE)
    return plt.figure(figsize=figsize)


def finish(path, show=None):
    """Lay out, save and close the current figure.

    Args:
        path: Destination file. Parent directories are created as needed.
        show: Whether to display the figure interactively. Defaults to the
            ``AEROSONAR_SHOW_PLOTS`` environment variable.

    Returns:
        str: The path the figure was written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    if show if show is not None else bool(os.environ.get("AEROSONAR_SHOW_PLOTS")):
        plt.show()
    plt.close()
    print(f"  Figure written to {path}")
    return str(path)


def plot_confusion_matrix(TP, FP, TN, FN, title, path):
    """Render an annotated 2x2 confusion matrix heatmap.

    Rows are the actual class and columns the predicted class, both ordered
    negative then positive.

    Args:
        TP: True positive count.
        FP: False positive count.
        TN: True negative count.
        FN: False negative count.
        title: Figure title.
        path: Destination file.

    Returns:
        str: The path the figure was written to.
    """
    data = np.array([[TN, FP], [FN, TP]])
    labels = np.array([[f'TN\n{TN}', f'FP\n{FP}'],
                       [f'FN\n{FN}', f'TP\n{TP}']])
    plt.style.use(STYLE)
    plt.figure(figsize=(6, 5))
    sns.heatmap(data, annot=labels, fmt="", cmap=CONFUSION_CMAP, cbar=True,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                annot_kws={"size": 14, "weight": "bold"})
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('Actual Label', fontsize=12)
    return finish(path)
