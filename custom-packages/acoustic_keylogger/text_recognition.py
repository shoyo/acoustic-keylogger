"""
Written by Shoyo Inokuchi (June 2019)

Scripts for the acoustic keylogger surrounding cluster labeling.
Repository is located at: https://github.com/shoyo-inokuchi/acoustic-keylogger
"""
from .audio_processing import *
from .unsupervised import extract_features

import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

import time


def signal_to_digit_sequence(signal, sr=44100, output=True):
    """Converts an audio signal to a sequence of digits corresponding to the
    clusters that each keystroke was categorized into.
    
    Example:
        Original Signal  
            => 'this is an example.'
        Returned Sequence
            => [0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 4, 7, 8, 5, 9, 10, 11, 12, 13]

    Explanation:
        The original signal contains 14 unique keystrokes ('t', 'h', 'i', etc),
        each of which are presumably grouped into separate clusters. Each 
        cluster is given a numeric label, and the returned sequence is a
        one-to-one mapping of the original keystrokes to their respective
        cluster labels.

        Note that the returned sequence above is meant to merely illustrate the
        pattern in which the original signal is mapped. In practice, the first
        unique letter does not necessarily map to '0', the second unique letter
        to '1' etc.
    """
    # Keystroke Detection
    t0 = time.time()
    if output: print('> Detecting keystrokes', end='')
    input_vec = detect_keystrokes(signal, sample_rate=sr)
    if output: print(f' => took {time.time() - t0}s')

    # Feature Extraction
    t0 = time.time()
    if output: print('> Extracting features', end='')
    feature_vec = np.array([extract_features(key) for key in input_vec])
    if output: print(f' => took {time.time() - t0}s')

    # Scaling Features
    t0 = time.time()
    if output: print('> Scaling features', end='')
    scaled_vec = MinMaxScaler().fit_transform(feature_vec)
    if output: print(f' => took {time.time() - t0}s')

    # Dimensionality Reduction with t-SNE
    t0 = time.time()
    if output: print(f'> Performing dimensionality reduction', end='')
    embedded_vec = TSNE().fit_transform(scaled_vec)
    if output: print(f' => took {time.time() - t0}s')

    # Clustering t-SNE output


    # Add label to return vector

    ret = np.empty_like(keys_raw)

        

