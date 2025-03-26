import numpy as np
import os

from torch.utils.data import DataLoader, Dataset


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

