#  Keyboard Acoustic Emanations Attack (Research) 

## Objectives
  * Create a proof-of-concept for a keyboard acoustic emanations attack, and evaluate
    its __accuracy__, __effectiveness__, and __accessibility__.
  * Keyboard acoustic emanations attack: Given just the sound of a victim typing on a keyboard,
    an attacker analyzes and extracts what was typed.
  * Theoretically requires no supervised training, and instead uses unsupervised methods to
    approximate typed keys according to various methods (view paper below).

## Motivation
Many research papers were published in the mid-2000s concerning the topic of keyboard acoustic
emanations attacks. Some research, such as [*Keyboard Acoustic Emanations Revisited* by L. Zhuang,
F. Zhou, J. D. Tygar in 2005](https://www.cs.cornell.edu/~shmat/courses/cs6431/zhuang.pdf), demonstrated
extremely accurate results (96% chars recovered from 10 minute sound recording) even without labeled 
training data. Technology surrounding machine learning has advanced considerably since the time such
research was published. This project therefore aims to identify how __accurate__, __effective__, and
__accessible__ such a security attack is in the current machine learning landscape. (If an undergraduate
student researcher like myself can create a relatively effective prototype, this would likely be a
sign for a considerable security concern.)

## Setting up
### Option 1 - Docker
This project uses a Python 3.6 development environment and a PostgreSQL database
to manage various audio data. This option conveniently spins up these environments with Docker Compose.  
* Install Docker. (https://www.docker.com/products/docker-desktop)  
* Build images with `$ docker-compose build`. This is only required the first time or whenever Docker settings are 
  changed.
        
This step will install all dependencies for env (such as Jupyter, Tensorflow, NumPy etc.)
and mount your local file system with the file system within the "env" Docker container.
        
* Spin up the database and development environment with `$ docker-compose up`.
        
This should open up the database for connections and make __http://localhost:8888__ access
the Jupyter notebook.

### Option 2 - No Docker
In exchange for containerization and seamless setup, Docker requires more overhead memory and 
comes with little quirks in the development environment with the current setup (like having to manually open the Jupyter
notebook). I find that a lot of times using Docker for small tweaks is a bit overkill, so I'm leaving this
option here.

* Install Python version 3.6. To downgrade from Python 3.7+ without overriding your current version,
  I recommend installing conda (https://www.anaconda.com/distribution/) and running 
  
        $ conda install python=3.6.8

* Set up a virtual environment. I recommend virtualenvwrapper for managing multiple environments.   

* Install dependencies with 

        $ pip3 install -r requirements.txt  

* Make sure Python can find custom packages for this repository with
 
        $ export PYTHONPATH=/path/to/repo/acoustic-keylogger-research/custom-packages
        
  I recommend adding this command to your `~/.bash_profile` or `~/.bashrc` so that it gets loaded between terminal sessions.

* Open Jupyter notebook with 
        
        $ jupyter notebook


This option can be simpler if you're unfamiliar with Docker or you don't need to access the database.
(Though the latter should still be possible using local postgres commands)


## Development
__Disclaimer__: Because this project is still in its very early stages and the repository is prone to frequent and
drastic refactorings, this section is likely to change the most. Many of the instructions or file locations
listed here may be completely wrong (though I'll do my best to actively keep this up-to-date).

__Last Updated__: April 14, 2019

### File I/O, keystroke extraction
The current method of processing sound data is recording audio, saving in WAV format, and extracting the keystroke
sounds as NumPy arrays. Functions to handle these processes are located in the __src/dataman__ package. To access the
functions, import the package with `from dataman.audio_processing import *` in the Python environment. Relevant functions
include `wav_read()`, `extract_keystrokes()`, and more. __src/dataman/audio_processing.py__ contains light documentation
in the docstrings.

### Database storage and retrieval
To avoid having the parse WAV files for keystrokes each time data is required, extracted keystroke data is stored
in a Postgres database (via SQLAlchemy). When using the data for training a model, the data is first retrieved from the 
database then preprocessed externally. Functions for storing and loading data are also located in the __src/dataman__
package. Relevant functions include `store_keystrokes()`, `load_keystrokes()`, and more.


### Other
Example for a basic keystroke classifier (currently at an incredible 4-5% accuracy at the time of writing) is located in
__src/supervised_learning.py__. 

In many cases, I find it useful to run a Python script within the Docker container specified by __Dockerfile__, but outside of
the Jupyter notebook (I've had cases where the Jupyter kernel dies during computationally intensive work, but work fine when
the same operations are run outside of the notebook). In such cases, run

    $ docker-compose run env <insert command>
    
Example:
    
    $ docker-compose run env python -i src/supervised_learning.py


## Testing [![CircleCI](https://circleci.com/gh/shoyo-inokuchi/acoustic-keylogger-research/tree/master.svg?style=svg)](https://circleci.com/gh/shoyo-inokuchi/acoustic-keylogger-research/tree/master)

Tests are being implemented for the __custom_packages/dataman__ package, which contains various functions for audio
processing and data management. These tests are contained in __tests/test_dataman__.

To run tests with the Docker configuration (Option 1), execute:

    $ docker-compose run env pytest -q tests
    
To run tests with no Docker configuration (Option 2), execute:

    $ python3.6 -m pytest -q tests

__Note:__ Both of the commands above are assumed to be executed from the base directory of this repository.


## Long-term TODO's
Here I list portions of this project that can always be improved.

#### Keystroke Extraction (src/dataman/audio_processing.py)
This refers to the process of inputting an audio clip of somebody typing, and outputting the individual keystrokes within the
audio encoded as NumPy arrays. The current implementation simply detects whenever a "silence threshold" is exceeded, and
slices 0.25 seconds from that point. Though this approach works well for basic audio recordings, it's very rigid. 
It can't handle clips where keys are pressed very rapidly in succession, or when there is too much background noise.

Since keystroke extraction is an essential functionality of this topic, any improvements to the algorithm will have
cascading benefits to the rest of this research. 

...


## Relevant Research Papers
Research papers for reference. For the most part, I intend to follow the methodology for the Zhuang, Zhou, Tygar paper 
(k-clustering, HMMs for guessing, iterating with trained classifier from inferred data).

### Supervised Methods
  * [*Keyboard acoustic emanations*](https://ieeexplore.ieee.org/document/1301311)
    by D. Asonov, R. Agrawal. 2004.

### Unsupervised Methods
  * [*Keyboard Acoustic Emanations Revisited*](https://www.cs.cornell.edu/~shmat/courses/cs6431/zhuang.pdf)
  by L. Zhuang, F. Zhou, J. D. Tygar. 2005.
  * [*Dictionary Attacks Using Keyboard Acoustic Emanations*](https://www.eng.tau.ac.il/~yash/p245-berger.pdf)
  by Y. Berger, A. Wool, A. Yeredor. 2006.

## Notes (to self)
  * The Dockerfile in this repository runs `jupyter notebook` with the `--allow-root` option
    to bypass errors the lazy way. This isn't really advised, so this should probably be
    reconsidered if this Docker container is ever run on a server.
