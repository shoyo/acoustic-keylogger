"""
Written by Shoyo Inokuchi (April 2019)

Tests for "audio_processing.py" in "acoustic_keylogger" package.
"""
import os
import pytest
import numpy as np
import numpy.random as rand
import sqlalchemy
import sqlalchemy.orm as orm
import psycopg2

from acoustic_keylogger.audio_processing import *


def test_wav_read():
    """Assert ouput is a 1D NumPy array of integers."""
    result = wav_read('datasets/samples/dummy-recording.wav')
    assert type(result) == np.ndarray
    assert result.ndim == 1
    assert type(result[0]) == np.int16


class TestSilenceThreshold:
    @pytest.mark.skip(reason='Not enough silence no longer raises error')
    def test_not_enough_silence(self):
        """Raise exception when a sound contains no initial silence."""
        signal = wav_read('datasets/samples/no-initial-silence.wav')
        with pytest.raises(Exception):
            silence_threshold(signal)

    def test_threshold_1(self):
        initial_silence = [0 for _ in range(44100 * 5)]
        random_noise = [rand.randint(-1000, 1001) for _ in range(100)]
        signal = np.array(initial_silence + random_noise)
        assert silence_threshold(signal, factor=1) == 0


    def test_threshold_2(self):
        initial_silence = [rand.randint(-20, 21) for _ in range(44100*5 - 1)]
        initial_silence.append(25)
        random_noise = [rand.randint(-1000, 1001) for _ in range(100)]
        signal = np.array(initial_silence + random_noise)
        assert silence_threshold(signal, factor=1) == 25


def test_remove_random_noise():
    """Assert noise is removed in output and signal is not mutated."""
    original = [2, 12, 4, -23, -4, 2, 0, 34]
    expected = [0, 12, 0, -23, 0, 0, 0, 34]
    threshold = 5
    signal = np.array(original)
    output = remove_random_noise(signal, threshold)
    for i in range(len(output)):
        assert output[i] == expected[i]
        assert signal[i] == original[i]


class TestDetectKeystrokes:
    def test_slowly_typed_phrases(self):
        phrases = {
            'hello_there',
            'jungle_cruise_',
            'this_is_not_a_password',
        }
        for phrase in phrases:
            filepath = 'datasets/detection-tests/' + phrase + '.wav'
            signal = wav_read(filepath)
            output = detect_keystrokes(signal)
            assert len(output) == len(phrase)

    @pytest.mark.skip(reason="Current detection algorithm can't handle rapid typing")
    def test_more_rapidly_typed_phrases(self):
        phrases = {
            'of_course_i_still_love_you_',
            'we_move_fast',
            'how_many_keystrokes_was_that_',
        }
        for phrase in phrases:
            filepath = 'datasets/detection-tests/' + phrase + '.wav'
            signal = wav_read(filepath)
            output = detect_keystrokes(signal)
            assert len(output) == len(phrase)

    # ---------------------------------------------------------------- #

    @pytest.mark.skip(reason='Revised keystroke detection algorithm not yet implemented')
    def test_slowly_typed_phrases2(self):
        """Run test again but with detect_keystrokes_improved()."""
        phrases = {
            'hello_there',
            'jungle_cruise_',
            'this_is_not_a_password',
        }
        for phrase in phrases:
            filepath = 'datasets/detection-tests/' + phrase + '.wav'
            signal = wav_read(filepath)
            output = detect_keystrokes_improved(signal)
            assert len(output) == len(phrase)

    @pytest.mark.skip(reason='Revised keystroke detection algorithm not yet implemented')
    def test_more_rapidly_typed_phrases2(self):
        """Run test again but with detect_keystrokes_improved()."""
        phrases = {
            'of_course_i_still_love_you_',
            'we_move_fast',
            'how_many_keystrokes_was_that_',
        }
        for phrase in phrases:
            filepath = 'datasets/detection-tests/' + phrase + '.wav'
            signal = wav_read(filepath)
            output = detect_keystrokes_improved(signal)
            assert len(output) != len(phrase)


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
    url = os.environ.get('TEST_DATABASE_URL')

    @pytest.mark.skip()
    def test_connect_to_database(self):
        engine = connect_to_database(self.url)
        assert type(engine) == sqlalchemy.engine.base.Engine

    @pytest.mark.skip()
    def test_create_keystroke_table(self):
        engine = connect_to_database(self.url)
        Session = orm.sessionmaker(bind=engine)

        # Assert that table does not exist by making a query
        session = Session()
        with pytest.raises((sqlalchemy.exc.ProgrammingError,
                            sqlalchemy.exc.InternalError)):
            session.query(KeystrokeTest).all()
        session.close()

        create_keystroke_table(self.url)

        # Assert that table exists by making a query
        session = Session()
        query = session.query(KeystrokeTest).all()
        session.close()
        assert query == []

    @pytest.mark.skip()
    def test_drop_keystroke_table(self):
        engine = connect_to_database(self.url)
        Session = orm.sessionmaker(bind=engine)

        # Create and drop table
        create_keystroke_table(self.url)
        drop_keystroke_test_table(self.url)

        # Assert that table is dropped by making a query
        session = Session()
        with pytest.raises((sqlalchemy.exc.ProgrammingError,
                            sqlalchemy.exc.InternalError)):
            session.query(KeystrokeTest).all()
        session.close()

    @pytest.mark.skip()
    def test_store_keystroke_data_and_retrieval(self):
        # Initialize database and data to be stored
        fpb = 'datasets/collection-tests/'
        keys = ['a', 'b', 'c', 'd', 'e', 'f']
        c = collect_keystroke_data(filepath_base=fpb, keys=keys)
        create_keystroke_table(self.url)

        # Store data in database
        store_keystroke_test_data(c, url=self.url)

        # Assert that storage was succesful by making a query
        engine = connect_to_database(self.url)
        Session = orm.sessionmaker(bind=engine)
        session = Session()
        query = session.query(KeystrokeTest).all()
        session.close()
        assert type(query) == list
        assert len(query) == 36

    @pytest.mark.skip()
    def test_load_keystroke_data(self):
        pass        
        
