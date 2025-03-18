#!/usr/bin/env python

import yt_dlp
import os
import gc

import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3
import numpy as np
from sklearn.decomposition import PCA

import torchaudio
from torchaudio.prototype.pipelines import VGGISH

#import ffmpeg
from moviepy import AudioFileClip

import tqdm
import argparse

def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
        print(f"The file {filename} has been deleted.")
    else:
        print(f"The file {filename} does not exist.")


def extract_video_features(video_path, max_frames=360):

  # Define preprocessing transformations
  preprocess = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize((299, 299)),
      transforms.ToTensor(),
      # normalize based on mean and standard deviation of imagenet dataset
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  # Open video file
  cap = cv2.VideoCapture(video_path)
  fps = cap.get(cv2.CAP_PROP_FPS)

  features = []
  frame_count = 0

  with torch.no_grad():  # No gradients needed for inference
      while cap.isOpened() and frame_count < max_frames*fps:
          ret, frame = cap.read()
          if not ret:
              break

          # Process one frame per second
          if frame_count % int(fps) == 0:
              # Preprocess the frame
              frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
              input_tensor = preprocess(frame).unsqueeze(0)  # Add batch dimension

              # Extract features using the InceptionV3 model
              feature_vector = inception(input_tensor)
              features.append(feature_vector.squeeze(0).cpu().numpy())  # Convert to numpy array

          frame_count += 1

  cap.release()
  return np.array(features)


def extract_audio_features(audio_path):   
    # Load and preprocess audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample if necessary
    if sample_rate != VGGISH.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, VGGISH.sample_rate)

    waveform = waveform.mean(dim=0)
    
    # Process input
    input_batch = vggish_input_processor(waveform)
    
    # Extract features
    with torch.no_grad():
        features = vggish(input_batch)
    
    return features.numpy()



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Read file content.')

  parser.add_argument("-s", "--stat_index", type=int, default=0, help='YoutubeID Index to start on in YoutubeID File')
  parser.add_argument("-e", "--end_index", type=int, default=100, help='YoutubeID Index to end on in YoutubeID File')
  parser.add_argument("-p", "--path", type=str, default='/Users/scottmerrill/Desktop', help='Path to YoutubeID file.  This will also be where output featuers are saved')
  args = vars(parser.parse_args())


  inception = inception_v3(pretrained=True, transform_input=False)
  inception.fc = torch.nn.Identity()  # Remove the classification layer (we only need features)
  inception.eval()  # Set the model to evaluation mode
  print('Loaded InceptionV3')


  vggish_input_processor = VGGISH.get_input_processor()
  vggish = VGGISH.get_model()
  print("loaded VGGish")


  # In[28]:


  os.makedirs(args['path'] + '/video', exist_ok=True)
  os.makedirs(args['path'] + '/audio', exist_ok=True)


  # Here are the youtube ids used by original VM-NET
  with open(args['path'] + '/Youtube_ID.txt', 'r') as file:
      content = file.read()  # Read the entire content of the file
  youtube_urls = content.split('\n')

  youtube_urls = youtube_urls[args['stat_index']:args['end_index']]


  for video_url in tqdm.tqdm(youtube_urls):
      # video id
      vid = video_url.split('=')[1]
      print(f"processing VID: {vid}")
      # delete files if they exist
      delete_file(args['path'] + '/tmp.mp4')
      delete_file(args['path'] + '/tmp.mp3')
      
      try:
          ydl_opts = {
              'quiet': True,  # Suppresses verbose output
              'format': 'mp4',  # Directly download the best MP4 format available
              'outtmpl': 'tmp.%(ext)s',  # Customize output filename
              'noplaylist': True,  # Ensure only the video itself is downloaded, not a playlist
              'postprocessor_args': [
                  '-ss', '00:00:00',  # Start from the beginning of the video
                  '-t', '360',  # Limit to 360 seconds (6 minutes)
              ],
          }

          # download youtube video
          with yt_dlp.YoutubeDL(ydl_opts) as ydl:
              ydl.download([video_url])


          # once we download all training features, we have to do pca whitening
          video_feats = extract_video_features('tmp.mp4')
          
          # convert mp4 to mp3 and get audio features
          
          # Load MP4 file and extract audio
          audio = AudioFileClip(args['path'] + "/tmp.mp4")
            
          # Write audio to MP3 file
          audio.write_audiofile(args['path'] + "/tmp.mp3")

          #_ = ffmpeg.input('tmp.mp4').output('tmp.mp3').global_args('-loglevel', 'quiet', '-y').run()
          audio_feats = extract_audio_features(args['path'] + "/tmp.mp3")
      
          np.save(args['path'] + f'/video/{vid}.npy', video_feats)
          np.save(args['path'] + f'/audio/{vid}.npy', audio_feats)        
          del video_feats, audio_feats
          gc.collect()

      except Exception as e:
          print(video_url, e)   


      
  print("Download Complete")
