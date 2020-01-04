from hmmlearn import hmm
import numpy as np

import os


def read_encode():
    path = os.environ['ENCODE_PATH']
    with open(path, 'r') as f:
        encode = f.readlines()
    return np.array([int(v) for v in encode[0].split()])


def create_transmat(corpus, keys='abcdefghijklmnopqrstuvwxyz .,'):
    """Return a transition matrix based on the sequence of letters in corpus.

    Assume that each word in the corpus is delimited by a space.

    Args:
    corpus -- an iterable containing words as elements
    keys   -- all letters that are accounted for in the transition matrix
    """
    mat = np.zeros((len(keys), len(keys)), dtype=np.int8)
    key_id_map = id_map(keys)

    # TODO

    return mat


# Tests

def test_create_transmat():
        corpora = [
        ['This', 'is', 'a', 'sentence'],
        ['contains', '``', 'unrecognized', "''", 'characters'],
        [''],
    ]
    
    keys = 'abcdefghijklmnopqrstuvwxyz .,'
    key_map = id_map(keys)
    reverse_map = dict(enumerate(keys))
    base_mat = np.zeros((len(keys), len(keys)), dtype=int)
    
    transmats = [
        base_mat.copy(),
        base_mat.copy(),
        base_mat.copy(),
    ]

    # Create correct transition matrix for corpora[0]
    transmats[0][key_map['t']][key_map['h']] = 1
    transmats[0][key_map['h']][key_map['i']] = 1
    transmats[0][key_map['i']][key_map['s']] = 2
    transmats[0][key_map['s']][key_map[' ']] = 2
    transmats[0][key_map[' ']][key_map['i']] = 1
    transmats[0][key_map[' ']][key_map['a']] = 1
    transmats[0][key_map['a']][key_map[' ']] = 1
    transmats[0][key_map[' ']][key_map['s']] = 1
    transmats[0][key_map['s']][key_map['e']] = 1
    transmats[0][key_map['e']][key_map['n']] = 2
    transmats[0][key_map['n']][key_map['t']] = 1
    transmats[0][key_map['t']][key_map['e']] = 1
    transmats[0][key_map['n']][key_map['c']] = 1
    transmats[0][key_map['c']][key_map['e']] = 1
    
    # Create correct transition matrix for corpora[1]
    transmats[1][key_map['c']][key_map['o']] = 2
    transmats[1][key_map['o']][key_map['n']] = 1
    transmats[1][key_map['n']][key_map['t']] = 1
    transmats[1][key_map['t']][key_map['a']] = 1
    transmats[1][key_map['a']][key_map['i']] = 1
    transmats[1][key_map['n']][key_map['s']] = 1
    transmats[1][key_map['s']][key_map[' ']] = 1
    transmats[1][key_map[' ']][key_map['u']] = 1
    transmats[1][key_map['u']][key_map['n']] = 1
    transmats[1][key_map['n']][key_map['r']] = 1
    transmats[1][key_map['r']][key_map['e']] = 1
    transmats[1][key_map['e']][key_map['c']] = 1
    transmats[1][key_map['o']][key_map['g']] = 1
    transmats[1][key_map['g']][key_map['n']] = 1
    transmats[1][key_map['n']][key_map['i']] = 1
    transmats[1][key_map['i']][key_map['z']] = 1
    transmats[1][key_map['z']][key_map['e']] = 1
    transmats[1][key_map['e']][key_map['d']] = 1
    transmats[1][key_map['d']][key_map[' ']] = 1
    transmats[1][key_map[' ']][key_map['c']] = 1
    transmats[1][key_map['c']][key_map['h']] = 1
    transmats[1][key_map['h']][key_map['a']] = 1
    transmats[1][key_map['a']][key_map['r']] = 1
    transmats[1][key_map['r']][key_map['a']] = 1
    transmats[1][key_map['a']][key_map['c']] = 1
    transmats[1][key_map['c']][key_map['t']] = 1
    transmats[1][key_map['t']][key_map['e']] = 1
    transmats[1][key_map['e']][key_map['r']] = 1
    transmats[1][key_map['r']][key_map['s']] = 1
    
    for corpus, expected in zip(corpora, transmats):
        print('--- Running test ---')
        actual, _ = create_transmat(corpus, keys=keys)
        diff = actual == expected
        if not diff.all():
            for i in range(diff.shape[0]):
                for j in range(diff.shape[1]):
                    if not diff[i][j]:
                        print(f"Failed: '{reverse_map[i]}' -> '{reverse_map[j]}'\n",
                              f"Expected {expected[i][j]} but got {actual[i][j]}")
        else:
            print('Passed')
        print()


# Utilities

def id_map(keys):
    """Return a dict mapping each char in string to a unique integer.
    
    >>> id_map('abcd')
    {'a': 0,
     'b': 1,
     'c': 2,
     'd': 3}
    """
    return {value: key for key, value in dict(enumerate(keys)).items()}


def reverse_id_map(keys):
    """Return a dict mapping a unique integer to each char in string.

    >>> reverse_id_map('abcd')
    {0: 'a',
     1: 'b',
     2: 'c',
     3: 'd'}
    """
    return dict(enumerate(keys))


def pprint_transmat(transmat, keys):
    """Pretty-print the transition matrix."""
    h, w = transmat.shape
    print('  ', end='')
    for key in keys:
        print(f'{key} ', end='')
    print()
    for i in range(h):
        print(f'{keys[i]} ', end='')
        for j in range(w):
            print(f'{transmat[i][j]} ', end='')
        print()


def main():
    encode = read_encode()

    # TODO
    model = hmm.GaussianHMM(n_components=2)


if __name__ == '__main__':
    main()
