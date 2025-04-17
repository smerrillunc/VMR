#!/usr/bin/env python
import numpy as np
import tqdm

import torch
import torchaudio
from torchaudio.prototype.pipelines import VGGISH
import torchvision.transforms as transforms
from scipy.interpolate import interp1d

import os
import gc
import argparse

def extract_audio_features(audio_path):   
    # Load and preprocess audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample if necessary
    if sample_rate != VGGISH.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, VGGISH.sample_rate)

    # Convert to mono
    waveform = waveform.mean(dim=0)
    
    # Process input into VGGish format
    input_batch = vggish_input_processor(waveform)

    # Extract features with VGGish
    with torch.no_grad():
        features = vggish(input_batch)  # Shape: (num_frames, 128)
    
    features = features.numpy()
    
    # VGGish generates one vector per 0.96s of audio
    frame_hop = 0.96
    num_vectors = features.shape[0]
    total_duration = num_vectors * frame_hop
    n_seconds = int(np.floor(total_duration))
    
    # Interpolate to get one 128-dim vector per second
    time_points = np.arange(num_vectors) * frame_hop
    interp = interp1d(time_points, features, axis=0, kind='linear', fill_value="extrapolate")
    second_marks = np.arange(n_seconds)
    per_second_features = interp(second_marks)  # Shape: (n_seconds, 128)

    return per_second_features


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Read file content.')
  parser.add_argument("-af", "--audio_file_path", type=str, default='/work/users/s/m/smerrill/Youtube8m/audio_files.txt', help='Path to audio file')
  parser.add_argument("-sp", "--save_path", type=str, default='/work/users/s/m/smerrill/Youtube8m', help='Save Path')
  args = vars(parser.parse_args())


  vggish_input_processor = VGGISH.get_input_processor()
  vggish = VGGISH.get_model()
  vggish.eval()
  print("loaded VGGish")

  os.makedirs(os.path.join(args['save_path'], 'vggish'), exist_ok=True)

  processed_vids = os.listdir(os.path.join(args['save_path'], 'vggish'))
  processed_vids = [x.split('.')[0] for x in processed_vids]

  # Here are the youtube ids used by original VM-NET
  with open(args['audio_file_path'], 'r') as file:
      content = file.read()  # Read the entire content of the file
  audio_files = content.split('\n')

  for audio_file in tqdm.tqdm(audio_files):
      try:
        # video id      
        vid = audio_file.split('/')[-1].split('.')[0]

        if vid in processed_vids:
          print(f"VID: {vid} already processed, skipping")
          continue

        vgg_feat = extract_audio_features(audio_file)
        np.save(os.path.join(args['save_path'], 'vggish', vid + '.npy'), vgg_feat)
        
        del vgg_feat
        gc.collect()

      except Exception as e:
        print(e)
      
  print("Audio Processing Complete")
