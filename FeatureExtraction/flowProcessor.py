#!/usr/bin/env python
import numpy as np
import tqdm

import cv2
from scipy.interpolate import interp1d

from collections import defaultdict
import os
import gc
import argparse

def compute_optical_flow_and_movement(video_path: str, resize_factor: float = 0.5):
    cap = cv2.VideoCapture(video_path)

    # Get FPS (frames per second)
    skip_frames = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        print("Error: Couldn't open video.")
        return []

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Couldn't read the first frame.")
        return []

    # Resize and convert first frame
    prev_frame = cv2.resize(prev_frame, (0, 0), fx=resize_factor, fy=resize_factor)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_metrics = []  # Store total movement for each frame
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        # Resize and convert current frame
        frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        # Detect objects (thresholding + contours)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_movement = 0  # Total movement for this frame

        # Process each object (contour)
        for contour in contours:
            if cv2.contourArea(contour) < 100:  # Ignore small objects/noise
                continue

            # Create mask for the object
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

            # Extract optical flow for object pixels
            flow_in_object = flow[mask == 255]
            if len(flow_in_object) > 0:
                # Calculate the average movement (displacement) for the object
                avg_flow = np.mean(flow_in_object, axis=0)  # Mean of dx, dy
                avg_movement = np.linalg.norm(avg_flow)  # Magnitude of movement vector
                total_movement += avg_movement  # Add to total movement for this frame

        # Append the total movement for the current frame
        frame_metrics.append(total_movement)
        prev_gray = gray

    cap.release()
    return frame_metrics


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Read file content.')
  parser.add_argument("-vf", "--video_file_path", type=str, default='/work/users/s/m/smerrill/Youtube8m/video_files.txt', help='Path to video file')
  parser.add_argument("-sp", "--save_path", type=str, default='/work/users/s/m/smerrill/Youtube8m', help='Save Path')
  parser.add_argument("-rf", "--resize_factor", type=float, default=0.75, help='Resize factor for optical flow processor')
  args = vars(parser.parse_args())

  os.makedirs(os.path.join(args['save_path'], 'flow'), exist_ok=True)

  processed_vids = os.listdir(os.path.join(args['save_path'], 'flow'))
  processed_vids = [x.split('.')[0] for x in processed_vids]

  # Here are the youtube ids used by original VM-NET
  with open(args['video_file_path'], 'r') as file:
      content = file.read()  # Read the entire content of the file
  video_files = content.split('\n')

  for video_file in tqdm.tqdm(video_files):
      try:
        # video id      
        vid = video_file.split('/')[-1].split('.')[0]

        if vid in processed_vids:
          print(f"VID: {vid} already processed, skipping")
          continue

        of_feat = compute_optical_flow_and_movement(video_file, resize_factor=args['resize_factor'])
        np.save(os.path.join(args['save_path'], 'flow', vid + '.npy'), of_feat)
        
        del of_feat
        gc.collect()

      except Exception as e:
        print(e)
      
  print("Video Processing Complete")
