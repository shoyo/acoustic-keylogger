"""
Written by Shoyo Inokuchi (November 2019)

Repository is located at https://github.com/shoyo/acoustic-keylogger-research.
"""

from .audio_processing import *


def zip_keys(signal_path, labels_path):
    """Return data, integer label, and string label arrays.

    Arguments:
        signal_path (str) -- path to 16-bit WAV file
        labels_path (str) -- path to text file containing labels
    """
    signal  = wav_read(signal_path)
    X = detect_keystrokes(signal)

    keymap = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7,
              'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14,
              'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21,
              'w': 22, 'x': 23, 'y': 24, 'z': 25, ' ': 26, '.': 27, ',': 28}

    #TODO:optimize looping, handle multiline text
    y_int, y_str = [], []

    with open(labels_path) as f:
        line = f.readline()
        for char in line:
            if char == '\n':
                continue
            y_int.append(keymap[char])
            y_str.append(char)

    y_int, y_str = np.array(y_int), np.array(y_str)

    if len(X) != len(y_int):
        raise Exception(f'Length mismatch: detected keys in signal ({len(X)}) ',
                        f'and number of labels ({len(y_int)})')

    return X, y_int, y_str
