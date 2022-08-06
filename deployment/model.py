#!/usr/bin/env python
# coding: utf-8

#################################################################################

#https://tts.readthedocs.io/en/latest/installation.html
!pip install -U pip
!pip install TTS
!pip install numpy
!apt-get install espeak
!pip install webrtcvad

#################################################################################

#Importing libraries

get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
import string
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import pickle
from scipy import stats
import gc
import missingno as msno
warnings.filterwarnings("ignore")
import os
import sys
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from multiprocessing import Pool
from matplotlib import pylab as plt
import math
import pandas as pd
import os
from os import path
from tqdm import tqdm
import json
import numpy as np
import urllib
import pathlib
from pydub.pydub import AudioSegment
#https://github.com/ngbala6/Audio-Processing/tree/master/Silence-Remove
import collections
import contextlib
import wave
import webrtcvad

#################################################################################

#Preprocessing

#all 100 raw audio has been stored under this directory after recording and manual cleanup by audasity software
root_dir='/content/drive/MyDrive/VoiceCloning/own voice'

#it will read all audio files ending with .wav under the root directory and store in a dataframe
def return_file_names_df(root_dir):
    
    #list for audio files
    flac_file_list=[]
    
    #https://stackoverflow.com/questions/9816816/get-absolute-paths-of-all-files-in-a-directory
    for filepath in pathlib.Path(root_dir).glob('**/*'):
        #formating the windows paths
        formatted_path=str(filepath.absolute()).replace('\\','/')
        #chnagin the absolute path to current directory as 'data'
        formatted_path=formatted_path.replace(root_dir,'')
        if (formatted_path.endswith('.wav')):
            flac_file_list.append(formatted_path)
   
    data_df=pd.DataFrame(flac_file_list, columns = ['audio'])
    return data_df
    
#this will have all the audio files in a dataframe format
data_df = return_file_names_df(root_dir)

#All the wav file should be under wavs directory     
root_dir_wav=os.path.join(root_dir,'wavs')

#create directory store the intermediate files
if not os.path.exists(os.path.join(root_dir,'wavs_intr')):
    os.makedirs(os.path.join(root_dir,'wavs_intr'))
    
out_dir=(os.path.join(root_dir,'wavs_intr'))

#convering the stero to mono channel for each audio 
#https://www.geeksforgeeks.org/splitting-stereo-audio-to-mono-with-pydub/
for file in tqdm(meta_data_df.file_name.values):
    wav_file=str(file)+'.wav'
    sound = AudioSegment.from_wav(os.path.join(root_dir_wav,wav_file))
    #convering the stero to mono channel
    sound = sound.set_channels(1)
    sound.export(os.path.join(out_dir,wav_file), format="wav")

#trimming silence part in the audio 
#https://github.com/ngbala6/Audio-Processing/tree/master/Silence-Remove  
def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])


def silence_remove(args):
    if len(args) != 2:
        sys.stderr.write(
            'Usage: Silence-Remove.py <aggressiveness> <path to wav file>\n')
        sys.exit(1)
    audio, sample_rate = read_wave(args[1])
    vad = webrtcvad.Vad(int(args[0]))
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 300, vad, frames)

    # Segmenting the Voice audio and save it in list as bytes
    concataudio = [segment for segment in segments]

    joinedaudio = b"".join(concataudio)
    #create directory to save the .wav files
    if not os.path.exists(os.path.join(root_dir,'wavs_trimmed/wavs')):
        os.makedirs(os.path.join(root_dir,'wavs_trimmed/wavs'))
    root_dir_='/content/drive/MyDrive/VoiceCloning/lj speech own voice/wavs'
    non_silenced_audio=str(str(str(args[1]).replace('\\','/')).split('/')[-1])
    write_wave(os.path.join(root_dir_,non_silenced_audio), joinedaudio, sample_rate)

#writing the audio files after removing the silence
root_dir_wav=os.path.join(root_dir,'wavs_intr')

for file in tqdm(meta_data_df.file_name.values):
    wav_file=str(file)+'.wav'
    #after experimenting with aggressiveness=2 , I found suitable for the raw audios, using 3 might remove certain words from the audio
    #please change accordingly
    silence_remove([2,os.path.join(root_dir_wav,wav_file)])
    
print ("After trimming the stored files are in :",os.path.join(root_dir,'wavs_trimmed/wavs'))   
  
#resampling script to convert the sample rate to 22050  
#!python resample.py --input_dir "\'/content/drive/MyDrive/VoiceCloning/lj speech own voice/wavs" --output_sr 22050 --output_dir #"/content/drive/MyDrive/VoiceCloning/datasets/wavs"

ouput_path='/content/drive/MyDrive/VoiceCloning/datasets/wavs/'

#the training scripts run 
#https://tts.readthedocs.io/en/latest/finetuning.html
#!python train_vits_v2.py --restore_path #/content/drive/MyDrive/VoiceCloning/datasets/vits-tts-finetune-August-02-2022_08+32AM-0000000/checkpoint_1005000.pth --coqpit.run_name "vits-tts-finetune"'

