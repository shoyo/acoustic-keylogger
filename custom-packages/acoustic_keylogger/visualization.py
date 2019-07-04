"""
Visualization scripts to assist with data analysis.

Credit for the foundation of the scripts goes to the book 'Hands on ML with
Scikit Learn and Tensorflow' and its corresponding Jupyter notebooks.

Modified by Shoyo Inokuchi (July 2019).
"""


import os

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def plot_keystrokes(X, y_int, y_str, min_distance=0.05, images=None, figsize=(13, 10)):
    """Plot keystrokes data X in 2D plane with graphical labels.
    
    Mostly copied from plot_digits() function from:
    https://github.com/ageron/handson-ml/blob/master/08_dimensionality_reduction.ipynb
    """
    X_norm = MinMaxScaler().fit_transform(X)
    neighbors = np.array([[10., 10.]])
    plt.figure(figsize=figsize)
    cmap = plt.cm.get_cmap('jet')
    labels = np.unique(y_int)
    for label in labels:
        plt.scatter(X_norm[y_int == label, 0], X_norm[y_int == label, 1], c=[cmap(label / 28)])
    plt.axis('off')
    ax = plt.gcf().gca()
    for index, image_coord in enumerate(X_norm):
        closest_distance = np.linalg.norm(np.array(neighbors) - image_coord, axis=1).min()
        if closest_distance > min_distance:
            neighbors = np.r_[neighbors, [image_coord]]
            if images is None:
                if images is None:
                    plt.text(image_coord[0], image_coord[1], y_str[index],
                             color=cmap(y_int[index] / 9), fontdict={"weight": "bold", "size": 16})
                else:
                    image = images[index]
                    imagebox = AnnotationBbox(OffsetImage(image, cmap="binary"), image_coord)
                    ax.add_artist(imagebox)


def save_fig(fig_id, tight_layout=True):
    path = os.path.join('../datasets/lab/figs/' + fig_id + '.png')
    print(f'Saving figure to {path}')
    plt.tight_layout() if tight_layout
    plt.savefig(path, format='png', dpi=300)

