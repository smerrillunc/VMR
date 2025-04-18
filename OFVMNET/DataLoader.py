import numpy as np
import os

from torch.utils.data import DataLoader, Dataset


class VideoAudioDataset(Dataset):
    def __init__(self, meta_df):
        self.meta_df = meta_df.reset_index(drop=True)
                
    def __len__(self):
        return len(self.meta_df)
    
    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]

        video_data = np.load(row['vid_filename'])
        audio_data = np.load(row['aud_filename'])
        segments = eval(row['segments'])
        flow_ranks = row['ranks']

        return video_data, audio_data, segments, flow_ranks
