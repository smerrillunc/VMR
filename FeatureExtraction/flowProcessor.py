#!/usr/bin/env python
import pandas as pd
import numpy as np
import tqdm

import cv2
import ruptures as rpt
import torchaudio

import os
import argparse


class OpticalFlowProcessor:
    def __init__(self, num_segments=9, min_frames=10, target_fps=3, resize_dim=(320, 240)):
        self.num_segments = num_segments
        self.min_frames = min_frames
        self.target_fps = target_fps
        self.resize_dim = resize_dim
        
    def extract_sampled_frames(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        self.target_fps = min(self.target_fps, original_fps)

        if not cap.isOpened():
            raise ValueError("Error: Unable to open the video file.")

        frame_interval = int(original_fps / self.target_fps)
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_interval == 0:
                resized = cv2.resize(frame, self.resize_dim)
                frames.append(resized)
            idx += 1

        cap.release()
        return frames

    def get_of_ranks(self, flow_trajectory, segments):
        segment_means = self.compute_segment_means(segments, flow_trajectory)
        ranks = self.rank_averages(segment_means)
        return ranks

    def get_best_worst_flow_times(self, segments, ranks):
        top_start, top_end = segments[np.where(ranks == 1)[0][0]], segments[np.where(ranks == 1)[0][0] + 1]
        bottom_start, bottom_end = segments[np.where(ranks == max(ranks))[0][0]], segments[np.where(ranks == max(ranks))[0][0] + 1]
        return (top_start/self.target_fps, top_end/self.target_fps), (bottom_start/self.target_fps, bottom_end/self.target_fps)
    
    def get_flow_trajectory(self, frames):
        flow_trajectory = [self.compute_of(frames[0], frames[1])] + [self.compute_of(frames[i - 1], frames[i]) for i in range(1, len(frames))]
        return np.array(flow_trajectory)
        
    def get_optical_flow_segments(self, flow_trajectory):    
        algo = rpt.Dynp(model='normal', min_size=self.min_frames*self.target_fps, jump=self.target_fps).fit(flow_trajectory)
        change_points = algo.predict(n_bkps=self.num_segments)  # The 'pen' parameter controls sensitivity

        # insert zero for start segment
        change_points.insert(0,0)
        return change_points

    @staticmethod
    def compute_of(img1, img2):
        prev_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
        avg_magnitude = cv2.mean(magnitude)[0]
        return avg_magnitude

    @staticmethod
    def compute_segment_means(segments, values):
        return [values[start:end].mean() if start < end else 0 for start, end in zip(segments[:-1], segments[1:])]

    @staticmethod
    def rank_averages(averages):
        sorted_indices = np.argsort(averages)[::-1]
        ranks = np.zeros_like(sorted_indices) + 1
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank + 1
        return ranks


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Read file content.')
  parser.add_argument("-t", "--target_fps", type=int, default=3, help='Target FPS to sample OF')
  parser.add_argument("-mf", "--min_frames", type=int, default=10, help='Minimum number of 1-second frames to include for segments')
  parser.add_argument("-ns", "--num_segments", type=int, default=10, help='Number of Segments to create')

  parser.add_argument("-vf", "--video_file_path", type=str, default='/work/users/s/m/smerrill/Youtube8m/video_paths.txt', help='Path to video file')
  parser.add_argument("-sp", "--save_path", type=str, default='/work/users/s/m/smerrill/Youtube8m', help='Save Path')
  
  args = vars(parser.parse_args())

  os.makedirs(os.path.join(args['save_path'], 'flow'), exist_ok=True)

  processed_vids = os.listdir(os.path.join(args['save_path'], 'flow'))
  processed_vids = [x.split('.')[0] for x in processed_vids]

  # Here are the youtube ids used by original VM-NET
  with open(args['video_file_path'], 'r') as file:
      content = file.read()  # Read the entire content of the file
  video_paths = content.split('\n')

  df = pd.DataFrame()
  for video_path in tqdm.tqdm(video_paths):
      
      processor = OpticalFlowProcessor(target_fps=args['target_fps'],
                                   min_frames=args['min_frames'], 
                                   num_segments=args['num_segments'])

      try:
        tmp = {}

        # video id      
        vid = video_path.split('/')[-1].split('.')[0]

        if vid in processed_vids:
          print(f"VID: {vid} already processed, skipping")
          #continue
        else:
          print(f"Processing VID: {vid}")

        frames = processor.extract_sampled_frames(video_path)
        flow_trajetory = processor.get_flow_trajectory(frames)
        change_points = processor.get_optical_flow_segments(flow_trajetory)
        ranks = processor.get_of_ranks(flow_trajetory, change_points)
        best_flow, worst_flow = processor.get_best_worst_flow_times(change_points, ranks)
    
        # save the trajectories
        np.save(os.path.join(args['save_path'], 'flow', vid + '.npy'), flow_trajetory)
        
        # save the flow ranks
        tmp = {'vid':vid,
                'ranks':ranks,
                # divide by fps since we'll be sampling OF at this FPS but we still want per second segments because
                # this is what our audio and vidoes are sampled at
                'segments':[x//processor.target_fps for x in change_points],
                'top_start':best_flow[0],
                'top_end':best_flow[1],
                'bottom_start':worst_flow[0],
                'bottom_end':worst_flow[1]}
        df = pd.concat([df, pd.DataFrame([tmp])])
        df.to_csv(os.path.join(args['save_path'], 'flow', 'ranks.csv'))

      except Exception as e:
        print(e)
      
  print("Video Processing Complete")
