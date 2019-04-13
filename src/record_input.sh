#!/bin/bash

# Script to record a sound clip. Records and saves clip as WAV file in local
# "/recordings" directory for futher processing.

# This script requires SoX installed -- 'brew install sox'

FILENAME="recording-"`date +"%s".wav`

read -p "Press enter to start recording (5s of audio)" rec
if [ -z $rec ]; then
  rec --bits=16 --rate=44100 $LAB_ENV/datasets/recordings/$FILENAME trim 0 5
fi
echo WAV file \"$FILENAME\" created in \"datasets/recordings/\" directory
