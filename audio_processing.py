"""
Written by Shoyo Inokuchi (March 2019)

Audio processing scripts for acoustic keylogger project. Repository is located at
https://github.com/shoyo-inokuchi/acoustic-keylogger-research.

Mirrors code on jupyter notebook in repository.
"""

import os
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.io import wavfile as wav
import tensorflow as tf
import sqlalchemy as db

# File input (single WAV file -> sound file data)

def wav_read(filename):
    """Return 1D NumPy array of wave-formatted audio data denoted by filename.
    
    Input should be a string containing the path to a wave-formatted audio file.
    File should be uncompressed 16-bit."""
    sample_rate, data_2d = wav.read(filename)
    data_1d = [val for val, _ in data_2d]
    return np.array(data_1d)


# Sound preprocessing before keystroke extraction

def silence_threshold(sound_data, n):
    """Return the silence threshold of the sound data.
    The sound data should begin with n-seconds of silence.
    """
    sampling_rate = 44100
    num_samples   = sampling_rate * n
    silence       = sound_data[:num_samples]
    tolerance     = 40
    factor        = 11  # factor multiplied to threshold
    if np.std(silence) > tolerance:
        raise Exception(f'Sound data must begin with at least {n}s of silence.')
    else:
        return max(np.amax(silence), abs(np.amin(silence))) * factor

    
def remove_random_noise(sound_data):
    """Remove random noise from sound data by replacing all values
    under the silence threshold to zero.
    """
    threshold = silence_threshold(sound_data, 5)
    sound_data_copy = sound_data[:]
    for i in range(len(sound_data_copy)):
        if abs(sound_data_copy[i]) < threshold:
            sound_data_copy[i] = 0
    return sound_data_copy


# Keystroke extraction (single WAV file -> keystroke data for file)

def extract_keystrokes(sound_data):
    """Return array of arrays denoting each keystroke detected in the sound_data.
    
    Each keystroke consists of a push peak (touch peak and hit peak) and a release peak.
    Returned keystrokes are coerced to be the same length by appending trailing zeros.
    
    :type sound_file  -- NumPy array denoting input sound clip
    :type sample_rate -- integer denoting sample rate (samples per second)
    :rtype            -- NumPy array of NumPy arrays
    """
    threshold          = silence_threshold(sound_data, 5)
    keystroke_duration = 0.3   # seconds (initial guess)
    sample_rate        = 44100 # Hz
    sample_length      = int(sample_rate * keystroke_duration)
    
    keystrokes = []
    i = 0
    while i < len(sound_data):
        if abs(sound_data[i]) > threshold:
            sample_start, sample_end = i, i + sample_length
            if sample_end <= len(sound_data) and abs(sound_data[sample_end]) > threshold:
                j = sample_end
                while sound_data[j] > threshold:
                    j -= 1
                sample_end = j
            keystroke = sound_data[sample_start:sample_end]
            trailing_zeros = np.array([0 for _ in range(sample_length - (sample_end - sample_start))])
            keystroke = np.concatenate((keystroke, trailing_zeros))
            keystrokes.append(keystroke)
            i = sample_end - 1
        i += 1
    return np.array(keystrokes)


# Data collection (multiple WAV files -> ALL keystroke data)

def collect_keystroke_data(output=False):
    """Read WAV files and return collected data.
    
    input format  -- WAV files in subdirectories of base_dir 
    output format -- list of dicts where each dict denotes a single collected
                     keystroke. Formatted like:
                         list(dict(keys: key type, sound digest, sound data))
    """
    base_dir = 'datasets/keystrokes/'
    alphabet = [letter for letter in 'abcdefghijklmnopqrstuvwxyz']
    other_keys = ['space', 'period', 'enter']
    keys = alphabet + other_keys
    
    collection = []
    for key in keys:
        wav_dir = base_dir + key + '/'
        if output: print(f'> Reading files from {wav_dir} for key "{key}"')
        for file in os.listdir(wav_dir):
            if output: print(f'  > Extracting keystrokes from "{file}"', end='')
            wav_data = wav_read(wav_dir + file)
            keystrokes = extract_keystrokes(wav_data)
            for keystroke in keystrokes:
                data = {
                    'key_type': key,
                    'sound_digest': hash(keystroke[:30].tobytes()),
                    'sound_data': keystroke,
                }
                collection.append(data)
            if output: print(f' => Found {len(keystrokes)} keystrokes')
    if output: print('> Done')
        
    return collection


# Data storage (ALL keystroke data -> store in database)

def connect_to_database():
    """Connect to database and return engine, connection, metadata."""
    engine = db.create_engine(os.environ['DATABASE_URL'])
    connection = engine.connect()
    metadata = db.MetaData()
    return engine, connection, metadata


def create_keystroke_table():
    """Create keystroke table in database and return create table."""
    keystrokes_tbl = db.Table('keystrokes', metadata,
                              db.Column('id', db.BigInteger, primary_key=True),
                              db.Column('key_type', db.String(32), nullable=False),
                              db.Column('sound_digest', db.BigInteger, nullable=False, unique=True),
                              db.Column('sound_data', db.ARRAY(db.Integer)))
    metadata.create_all(engine)
    return keystrokes_tbl


def store_keystroke_data(collected_data, engine, metadata):
    """Store collected data in database and return result proxy.
    
    input format  -- output of collect_keystroke_data()
    output format -- pandas DataFrame used to store data in database
    """
    keystrokes_tbl = metadata.tables['keystrokes']
    query = db.insert(keystrokes_tbl)
    result_proxy = engine.execute(query, collected_data)
    return result_proxy


# Data retrieval

def load_keystroke_data():
    """Format data to pass to Keras model.fit(). Return a tuple in the
    form of: (training data "x", labels "y").
    
    For details, view documentation at: https://keras.io/models/model/#fit
    """
    engine = db.create_engine("postgresql+psycopg2://postgres@acoustic-keylogger-research_db_1:5432")
    engine.connect()
    metadata = db.MetaData()



# Data preprocessing (before training)

def scale_keystroke_data(data):
    """Scale each keystroke data to a value between 0 and 1.
    Return a copy of data and don't modify data itself.
    """
    data_copy = deepcopy(data)
    for label in data_copy:
        for i in range(len(data_copy[label])):
            data_copy[label][i] /= max(data_copy[label][i])
    return data_copy

