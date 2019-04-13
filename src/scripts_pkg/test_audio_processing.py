"""
Written by Shoyo Inokuchi (April 2019)

Tests for "audio_processing.py" in scripts_pkg
"""
import pytest
import numpy as np
import numpy.random as rand

from scripts_pkg.audio_processing import *


def test_wav_read():
    """Assert ouput is a 1D NumPy array of integers."""
    dummy_file = 'datasets/samples/dummy-recording.wav'
    result = wav_read(dummy_file)
    assert type(result) == np.ndarray
    assert result.ndim == 1
    assert type(result[0]) == np.int16


class TestSilenceThreshold:
    def test_not_enough_silence(self):
        """Raise exception when a sound contains no initial silence."""
        filename = 'datasets/samples/no-initial-silence.wav'
        input = wav_read(filename)
        with pytest.raises(Exception):
            silence_threshold(input)

    def test_threshold_1(self):
        initial_silence = [0 for _ in range(44100 * 5)]
        random_noise = [rand.randint(-1000, 1001) for _ in range(100)]
        input = np.array(initial_silence + random_noise)
        assert silence_threshold(input, factor=1) == 0
        

    def test_threshold_2(self):
        initial_silence = [rand.randint(-20, 21) for _ in range(44100*5 - 1)]
        initial_silence.append(25)
        random_noise = [rand.randint(-1000, 1001) for _ in range(100)]
        input = np.array(initial_silence + random_noise)
        assert silence_threshold(input, factor=1) == 25


def test_remove_random_noise():
    """Assert noise is removed in output and input is not mutated."""
    original = [2, 12, 4, -23, -4, 2, 0, 34]
    expected = [0, 12, 0, -23, 0, 0, 0, 34]
    threshold = 5
    input = np.array(original)
    output = remove_random_noise(input, threshold)
    for i in range(len(output)):
        assert output[i] == expected[i]
        assert input[i] == original[i]


class TestExtractKeystrokes:
    pass
    
