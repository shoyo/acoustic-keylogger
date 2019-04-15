"""
Written by Shoyo Inokuchi (April 2019)

Tests for "audio_processing.py" in "database_management" package.
"""
import pytest
import numpy as np
import numpy.random as rand

from dataman.audio_processing import *

BASE_DIR = './' # Assumes pytest is called from project base dir


def test_wav_read():
    """Assert ouput is a 1D NumPy array of integers."""
    dummy_file = 'datasets/samples/dummy-recording.wav'
    result = wav_read(dummy_file, base_dir=BASE_DIR)
    assert type(result) == np.ndarray
    assert result.ndim == 1
    assert type(result[0]) == np.int16


class TestSilenceThreshold:
    def test_not_enough_silence(self):
        """Raise exception when a sound contains no initial silence."""
        input = wav_read('datasets/samples/no-initial-silence.wav',
                         base_dir=BASE_DIR)
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
    def test_slowly_typed_phrases(self):
        phrases = {
            'hello_there',
            'jungle_cruise_',
            'this_is_not_a_password',
        }
        for phrase in phrases:
            filename = 'datasets/extraction-tests/' + phrase + '.wav'
            input = wav_read(filename, base_dir=BASE_DIR)
            output = extract_keystrokes(input)
            assert len(output) == len(phrase)
            
    def test_more_rapidly_typed_phrases(self):
        phrases = {
            'of_course_i_still_love_you_',
            'we_move_fast',
            'how_many_keystrokes_was_that_',
        }
        for phrase in phrases:
            filename = 'datasets/extraction-tests/' + phrase + '.wav'
            input = wav_read(filename, base_dir=BASE_DIR)
            output = extract_keystrokes(input)
            assert len(output) != len(phrase)
            
    def test_x(self):
        pass
    
    
class TestCollectKeystrokeData:
    base_dir = 'datsets/collection-tests/'
    keys = ['a', 'b', 'c', 'd', 'e', 'f']
    
    def test_standard_collection(self):
        collection = collect_keystroke_data(base_dir=self.base_dir,
                                            keys=self.keys)
        expected_len = {'a': 5, 'b': 8, 'c': 10, 'd': 3, 'e': 8, 'f': 2}
        for letter in expected_lens:
            assert len(collection[letter]) == expected_len[letter]
    
    def test_sound_digests_are_unique(self):
        collection = collect_keystroke_data(base_dir=self.base_dir,
                                            keys=self.keys)
        used_digests = set()
        for data in collection:
            assert data['sound_digest'] not in used_digests
            used_digests.add(data['sound_digest'])
            
    def test_ignore(self):
        ignore = {
            'a-5x.wav': {0, 1, 3, 4},
            'b-4x.wav': {0, 1, 2},
            'd-3x.wav': {2},
            'e-8x.wav': {6, 7},
        }
        no_ignore = collect_keystroke_data(base_dir=self.base_dir,
                                           keys=self.keys,
                                           ignore=ignore)
        with_ignore = collect_keystroke_data(base_dir=self.base_dir,
                                             keys=self.keys,
                                             ignore=ignore)
        num_ignore = sum([len([val for val in ignore[key]]) for key in ignore])
        assert len(no_ignore) - len(with_ignore) == num_ignore
        
        
# class TestDatabaseOperations():
#     url = os.environ['DATABASE_URL'] # Consider designating a separate test db
#     base_dir = 'datasets/database-tests/'
#     collection = collect_keystroke_data(base_dir)
#     
#     def test_connect_to_real_db(self):
#         engine = connect_to_database(url)
#         assert type(engine) == 
#     
#     def test
#     
#     
#     
# def test_scale_keystroke_data():
#     pass