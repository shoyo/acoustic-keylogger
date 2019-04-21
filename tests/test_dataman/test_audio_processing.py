"""
Written by Shoyo Inokuchi (April 2019)

Tests for "audio_processing.py" in "database_management" package.
"""
import pytest
import numpy as np
import numpy.random as rand
import sqlalchemy
import sqlalchemy.orm as orm
import psycopg2

from dataman.audio_processing import *


def test_wav_read():
    """Assert ouput is a 1D NumPy array of integers."""
    result = wav_read('datasets/samples/dummy-recording.wav')
    assert type(result) == np.ndarray
    assert result.ndim == 1
    assert type(result[0]) == np.int16


class TestSilenceThreshold:
    def test_not_enough_silence(self):
        """Raise exception when a sound contains no initial silence."""
        input = wav_read('datasets/samples/no-initial-silence.wav')
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
            filepath = 'datasets/extraction-tests/' + phrase + '.wav'
            input = wav_read(filepath)
            output = extract_keystrokes(input)
            assert len(output) == len(phrase)

    def test_more_rapidly_typed_phrases(self):
        phrases = {
            'of_course_i_still_love_you_',
            'we_move_fast',
            'how_many_keystrokes_was_that_',
        }
        for phrase in phrases:
            filepath = 'datasets/extraction-tests/' + phrase + '.wav'
            input = wav_read(filepath)
            output = extract_keystrokes(input)
            assert len(output) != len(phrase)

    def test_no_overlap(self):
        pass


class TestCollectKeystrokeData:
    fpb = 'datasets/collection-tests/'
    keys = ['a', 'b', 'c', 'd', 'e', 'f']

    def test_standard_collection(self):
        c = collect_keystroke_data(filepath_base=self.fpb,
                                   keys=self.keys)
        expected_len = {'a': 5, 'b': 8, 'c': 10, 'd': 3, 'e': 8, 'f': 2}
        for letter in expected_len:
            actual_len = len([d for d in c if d['key_type'] == letter])
            assert actual_len == expected_len[letter]

    def test_sound_digests_are_unique(self):
        collection = collect_keystroke_data(filepath_base=self.fpb,
                                            keys=self.keys)
        used_digests = set()
        for data in collection:
            assert data['sound_digest'] not in used_digests
            used_digests.add(data['sound_digest'])

    def test_ignore(self):
        ignore = {
            'a-5x.wav': {0, 1, 3, 4},
            'b-5x.wav': {0, 1, 2},
            'd-3x.wav': {2},
            'e-8x.wav': {6, 7},
        }
        no_ignore = collect_keystroke_data(filepath_base=self.fpb,
                                           keys=self.keys)
        with_ignore = collect_keystroke_data(filepath_base=self.fpb,
                                             keys=self.keys,
                                             ignore=ignore)
        num_ignore = sum([len([v for v in ignore[key]]) for key in ignore])
        assert len(no_ignore) - len(with_ignore) == num_ignore


class TestDatabaseOperations:
    def test_connect_to_database(self):
        engine = connect_to_database()
        assert type(engine) == sqlalchemy.engine.base.Engine

    def test_create_keystroke_table(self):
        engine = connect_to_database()
        Session = orm.sessionmaker(bind=engine)
        session = Session()
        with pytest.raises(psycopg2.errors.UndefinedTable):
            session.query(KeystrokeTest)
        create_keystroke_table()
        query = session.query(KeystrokeTest)
        assert type(query) == sqlalchemy.orm.query.Query

        
