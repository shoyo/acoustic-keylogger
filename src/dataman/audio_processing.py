"""
Written by Shoyo Inokuchi (March 2019)

Audio processing scripts for acoustic keylogger project. Repository is located at
https://github.com/shoyo-inokuchi/acoustic-keylogger-research.
"""

import os
from copy import deepcopy

import numpy as np
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
import sqlalchemy as db
import sqlalchemy.orm as orm
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects import postgresql


# File input (single WAV file -> sound file data)

def wav_read(filename, base_dir=None):
    """Return 1D NumPy array of wave-formatted audio data denoted by filename.
    
    Input should be a string containing the path to a wave-formatted audio file.
    File should be uncompressed 16-bit."""
    base_dir = base_dir or '/env/'
    sample_rate, data_2d = wav.read(base_dir + filename)
    data_1d = [val for val, _ in data_2d]
    return np.array(data_1d)


# Sound preprocessing before keystroke extraction

def silence_threshold(sound_data, n=5, factor=11):
    """Return the silence threshold of the sound data.
    The sound data should begin with n-seconds of silence.
    """
    sampling_rate = 44100
    num_samples   = sampling_rate * n
    silence       = sound_data[:num_samples]
    tolerance     = 40
    if np.std(silence) > tolerance:
        raise Exception(f'Sound data must begin with at least {n}s of silence.')
    else:
        return max(np.amax(silence), abs(np.amin(silence))) * factor

    
def remove_random_noise(sound_data, threshold=None):
    """Return a copy of sound_data where random noise is replaced with 0s.

    The original sound_data is not mutated.
    """
    threshold = threshold or silence_threshold(sound_data)
    sound_data_copy = deepcopy(sound_data)
    for i in range(len(sound_data_copy)):
        if abs(sound_data_copy[i]) < threshold:
            sound_data_copy[i] = 0
    return sound_data_copy


# Keystroke extraction (single sound data -> all keystroke data in data)

def extract_keystrokes(sound_data, sample_rate=44100):
    """Return slices of sound_data that denote each keystroke present.
    
    Returned keystrokes are coerced to be the same length by appending trailing
    zeros.

    Current algorithm:
    - Calculate the "silence threshold" of sound_data.
    - Traverse sound_data until silence threshold is exceeded.
    - Once threshold is exceeded, mark that index as "a".
    - Identify the index 0.3s ahead of "a", and mark that index as "b".
    - If "b" happens to either:
          1) denote a value that value exceeds the silence threshold (aka:
             likely impeded on the next keystroke waveform)
          2) exceed the length of sound_data
      then backtrack "b" until either:
          1) it denotes a value lower than the threshold
          2) "b" is 1 greater than "a"
    - Slice sound_data from index "a" to "b", and append that slice to the list
      to be returned. If "b" was backtracked, then pad the slice with trailing
      zeros to make it 0.3s long.

    :type sound_file  -- NumPy array denoting input sound clip
    :type sample_rate -- integer denoting sample rate (samples per second)
    :rtype            -- NumPy array of NumPy arrays
    """
    threshold          = silence_threshold(sound_data, 5)
    keystroke_duration = 0.23   # seconds 
    len_sample         = int(sample_rate * keystroke_duration)
    
    keystrokes = []
    i = 0
    while i < len(sound_data):
        if abs(sound_data[i]) > threshold:
            a, b = i, i + len_sample
            if b > len(sound_data):
                b = len(sound_data)
            while abs(sound_data[b]) > threshold and b > a:
                b -= 1
            keystroke = sound_data[a:b]
            trailing_zeros = np.array([0 for _ in range(len_sample - (b - a))])
            keystroke = np.concatenate((keystroke, trailing_zeros))
            keystrokes.append(keystroke)
            i = b - 1
        i += 1
    return np.array(keystrokes)


# Display extracted keystrokes (WAV file -> all keystroke graphs)

def visualize_keystrokes(filename, base_dir=None):
    """Display each keystroke contained in WAV file specified by filename."""
    wav_data = wav_read(filename, base_dir)
    keystrokes = extract_keystrokes(wav_data)
    n = len(keystrokes)
    print(f'Number of keystrokes detected in "{filename}": {n}')
    print('Drawing keystrokes...')
    num_cols = 3
    num_rows = n/num_cols + 1
    plt.figure(figsize=(num_cols * 6, num_rows * .75))
    for i in range(n):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.title(f'Index: {i}')
        plt.plot(keystrokes[i])
    plt.show()
    

# Data collection (multiple WAV files -> ALL keystroke data)

def collect_keystroke_data(output=False, ignore=None):
    """Read WAV files and return collected data.

    Arguments:
    output -- True to display status messages of keystroke extraction.
    ignore -- dict of filenames mapped to indices of keystrokes to ignore.
    
    input format  -- WAV files in subdirectories of "base_dir"
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

Base = declarative_base()


class Keystroke(Base):
    """Schema for Keystroke model."""
    __tablename__ = 'keystrokes'

    id = db.Column(db.BigInteger, primary_key=True)
    key_type = db.Column(db.String(32), nullable=False)
    sound_digest = db.Column(db.BigInteger, nullable=False, unique=True)
    sound_data = db.Column(postgresql.ARRAY(db.Integer))

    def __repr__(self):
        return f'<Keystroke(key={self.key_type}, digest={self.sound_digest})>'


def connect_to_database():
    """Connect to database and return engine, connection, metadata."""
    engine = db.create_engine(os.environ['DATABASE_URL'])
    connection = engine.connect()
    return engine


def create_keystroke_table():
    """Create keystroke table in database."""
    engine = connect_to_database()
    Base.metadata.create_all(engine)


def drop_keystroke_table():
    """Drop keystroke table in database."""
    engine = connect_to_database()
    Keystroke.__table__.drop(engine)


def store_keystroke_data(collected_data):
    """Store collected data in database and return result proxy.
    
    input format  -- output of collect_keystroke_data()
    """
    engine = connect_to_database()
    Session = orm.sessionmaker(bind=engine)
    session = Session()
    try:
        for data in collected_data:
            entry = Keystroke(key_type=data['key_type'],
                              sound_digest=data['sound_digest'],
                              sound_data=data['sound_data'])
            session.add(entry)
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


# Data retrieval

def load_keystroke_data():
    """Retrieve data from database, do relevant formatting, and return it.

    Return as a tuple of tuples of the form: (x, y)
    where x denotes training data and y denotes labels.

    This data will be passed to tf.keras.model.fit().
    For details, view documentation at: https://keras.io/models/model/#fit
    """
    engine = connect_to_database()
    Session = orm.sessionmaker(bind=engine)
    session = Session()
    keystrokes = session.query(Keystroke).all()
    session.close()

    np.random.shuffle(keystrokes)

    keytype_id = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6,
        'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14,
        'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22,
        'x': 23, 'y': 24, 'z': 25, 'space': 26, 'period': 27, 'enter': 28,
    }

    data, labels = [], []
    for row in keystrokes:
        data.append(row.sound_data)
        labels.append(keytype_id[row.key_type])

    return np.array(data), np.array(labels)


def load_keystroke_data_for_binary_classifier(classify={'space'}):
    """Load data from database for a binary classifier."""
    engine = connect_to_database()
    Session = orm.sessionmaker(bind=engine)
    session = Session()
    keystrokes = session.query(Keystroke).all()
    session.close()

    np.random.shuffle(keystrokes)

    data, labels = [], []
    for row in keystrokes:
        if row.key_type in classify:
            labels.append(1)
        else:
            labels.append(0)
        data.append(row.sound_data)

    return np.array(data), np.array(labels)


# Data preprocessing (before training)

def scale_keystroke_data(data):
    """Scale each value in data to an appropriate value between 0 and 1.
    
    input format -- NumPy array of Numpy arrays
    """
    data_copy = deepcopy(data.astype(float))
    for i in range(len(data_copy)):
        max_val, min_val = max(data_copy[i]), min(data_copy[i])
        data_copy[i] = (data_copy[i] - min_val) / (max_val - min_val)
    return data_copy

