#!/usr/bin/env python
# coding: utf-8

#########################################################################

#Installing necessary libraries
#https://tts.readthedocs.io/en/latest/installation.html
!pip install -U pip
!pip install TTS
!pip install numpy
!apt-get install espeak
!pip install webrtcvad
#https://www.geeksforgeeks.org/how-to-run-flask-app-on-google-colab/
!pip install flask-ngrok
!pip install flask
!pip install pyngrok
#Need to configure authentication to use flask-ngrok api
!ngrok authtoken 'your token'

#########################################################################

#Importing libraries
import flask
from flask import Flask, jsonify, request,Response, send_file
from flask_ngrok import run_with_ngrok
import os
import string
import time
import argparse
import json
import numpy as np
import IPython
from IPython.display import Audio
import torch
try:
  from TTS.utils.audio import AudioProcessor
except:
  from TTS.utils.audio import AudioProcessor
from TTS.tts.models import setup_model
from TTS.config import load_config
from TTS.tts.models.vits import *
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.models import setup_model
from TTS.config import load_config
from TTS.tts.models.vits import *

#################################################################################

from google.colab import drive
drive.mount('/content/drive')

#########################################################################

#create flask app 
app = Flask(__name__)

templates_path = '/content/templates'

# create teamplates path to store the html page
os.makedirs(templates_path, exist_ok=True)

#upload html file
!gdown 1wNRhlAXkMeAUBlEPc9YutmRlowueXf7Y -O '/content/templates/index.html'

OUT_PATH = 'out/'
# create output path if not exists
os.makedirs(OUT_PATH, exist_ok=True)

# model variables
MODEL_PATH = '/content/drive/MyDrive/VoiceCloning/models/checkpoint_1008000.pth'
CONFIG_PATH = '/content/drive/MyDrive/VoiceCloning/models/config.json'
USE_CUDA = torch.cuda.is_available()

# load the config
C = load_config(CONFIG_PATH)

# load the audio processor
ap = AudioProcessor(**C.audio)

speaker_embedding = None

model = setup_model(C)
#running on cpu 
cp = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

#########################################################################


# remove speaker encoder in single speaker setting
model_weights = cp['model'].copy()
for key in list(model_weights.keys()):
  if "speaker_encoder" in key:
    del model_weights[key]

#load model weights
model.load_state_dict(model_weights)
model.eval()

#based on cuda availablity
if USE_CUDA:
    model = model.cuda()

#########################################################################


# synthesize voice parameters
use_griffin_lim = False

#tuning parameters
model.inference_noise_scale = 0.2 # defines the noise variance applied to the random z vector at inference.
model.inference_noise_scale_dp = 0.2 # defines the noise variance applied to the duration predictor z vector at inference.


###################################################

@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    to_predict_list = request.form.to_dict()
    text=to_predict_list['input_text']
    speed_of_speech=float(to_predict_list['speed_of_speech'])
    print(" > text: {}".format(text))
    print(" > speed_of_speech: {}".format(speed_of_speech))
    model.length_scale = speed_of_speech
    wav, alignment, _, _ = synthesis(
                        model=model,
                        text=text,
                        CONFIG=C,
                        use_cuda="cuda" in str(next(model.parameters()).device),
                        speaker_id=None,
                        style_wav=None,
                        style_text=None,
                        use_griffin_lim=None,
                        d_vector=None,
                        language_id=None,
                    ).values()
    print("Generated Audio")
    IPython.display.display(Audio(wav, rate=ap.sample_rate))
    file_name = text.replace(" ", "_")
    attach_file_name = file_name.translate(str.maketrans('', '', string.punctuation.replace('_', ''))) + '.wav'
    file_name='tts_output.wav'
    out_path = os.path.join(OUT_PATH, file_name)
    print(" > Saving output to {}".format(out_path))
    ap.save_wav(wav, out_path)
    return send_file(
         '/content/out/tts_output.wav', 
         mimetype="audio/wav", 
         as_attachment=True, 
         attachment_filename=attach_file_name)

if __name__ == '__main__':
    run_with_ngrok(app) 
    app.run()    

#########################################################################
