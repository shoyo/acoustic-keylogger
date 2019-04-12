# Keyboard Acoustic Emanations Attack (Research)

## Objectives
  * Create a proof-of-concept for a purely "audio-based" keylogging attack. 
    (referred to as a "keyboard acoustic emanations attack")
  * Ideally requires no supervised training, and instead uses unsupervised methods to
    approximate typed keys according to key-press sound waveform and wave intervals.

## Motivation
Many research papers were published in the mid-2000s concerning the topic of keyboard acoustic
enmanations attack. Some research, such as [*Keyboard Acoustic Emanations Revisited* by L. Zhuang,
F. Zhou, J. D. Tygar in 2005](https://www.cs.cornell.edu/~shmat/courses/cs6431/zhuang.pdf), demonstrated
extremely accurate results even without labeled training data. Technology surrounding machine learning
has advanced considerably since the time such research was published. This research therefore
aims to identify how __accurate__, __effective__, and __accessible__ such a security attack is in
the current machine learning landscape.

## Development Environment
### Option 1 - Virtual environment
Install Python version 3.6.  
Set up virtual environment. I recommend virtualenvwrapper for managing environments.   
Install dependencies with `pip3 install -r requirements.txt`.  
Open Jupyter notebook with `jupyter notebook`.


This option is simpler if you're unfamiliar with Docker or you understand the concept of
not over-complicating a simple local development environment.

### Option 2 - Docker
Install Docker.  
Build image (named "env" in this case) with

    docker build -t env .

Run "env" image with

    docker run -p 8888:8888 -v /path/to/local/directory:/env env

Close the Jupyter notebook with __Ctrl + c__. I mention this because the __Quit__ button
within the Jupyter UI sometimes doesn't work with this setup.

This option was born out of me not knowing how to downgrade my local Python version to
3.6 in a simple way. Also, Docker.

## Relevant Research Papers
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
    reconsidered if this docker container is ever run on a server.
