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


class VideoAudioTestset(Dataset):
    def __init__(self, meta_df, vid_dict, aud_dict, pad_size):
        self.meta_df = meta_df.reset_index(drop=True)
        self.vid_dict = vid_dict
        self.aud_dict = aud_dict
        self.pad_size = pad_size


    @staticmethod
    def pad_or_truncate(arr, pad_length=400):
        """
        Pads or truncates a 2D numpy array to the specified number of rows.

        If the array has fewer rows than `pad_length`, it repeats rows from the start
        to reach the target length. If it has more, it truncates to `pad_length`.

        Parameters:
            arr (np.ndarray): Input array of shape (N, D).
            pad_length (int): Desired number of rows in the output array.

        Returns:
            np.ndarray: Output array of shape (pad_length, D).
        """
        current_length = arr.shape[0]
        
        if current_length == pad_length:
            return arr
        elif current_length > pad_length:
            return arr[:pad_length]
        else:
            needed = pad_length - current_length
            repeat_times = (needed + current_length - 1) // current_length  # Ceiling division
            repeated = np.tile(arr, (repeat_times + 1, 1))[:needed]
            return np.vstack((arr, repeated))

    def __len__(self):
        return len(self.meta_df)
    
    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]

        video_data = np.load(row['vid_filename'])
        audio_data = np.load(row['aud_filename'])

        if self.pad_size > 0:
            video_data = VideoAudioTestset.pad_or_truncate(video_data)
            audio_data = VideoAudioTestset.pad_or_truncate(audio_data)
            
        segments = eval(row['segments'])

        return video_data, audio_data, segments, self.vid_dict[row['vid_filename']], self.aud_dict[row['aud_filename']]
