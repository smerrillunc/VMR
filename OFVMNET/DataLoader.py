import numpy as np
import pandas as pd
import os

import tensorflow as tf
import numpy as np
from IPython.display import YouTubeVideo

import requests
import json

import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from scipy.signal import peak_prominences
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import itertools
import random

import argparse



class VideoAudioDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.filenames = os.listdir(os.path.join(path, 'video'))
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        video_data = np.load(os.path.join(self.path, 'video', filename))
        audio_data = np.load(os.path.join(self.path, 'audio', filename))
        video_data = video_data[:, :1024]
        return video_data, audio_data

