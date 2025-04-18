import numpy as np
#from scipy.signal import peak_prominences
#from scipy.signal import find_peaks
import ruptures as rpt
import cv2
import torchaudio


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
        return frames, target_fps

    def get_of_ranks(self, flow_trajectory, segments):
        segment_means = self.compute_segment_means(segments, flow_trajectory)
        ranks = self.rank_averages(segment_means)
        return ranks

    def get_best_worst_flow_times(self, segments, ranks):
        top_start, top_end = segments[np.where(ranks == 1)[0][0]], segments[np.where(ranks == 1)[0][0] + 1]
        bottom_start, bottom_end = segments[np.where(ranks == max(ranks))[0][0]], segments[np.where(ranks == max(ranks))[0][0] + 1]
        return (top_start/self.target_fps, top_end/self.target_fps), (bottom_start/self.target_fps, bottom_end/self.target_fps)
    
    def get_flow_trajectory(self, frames):
        flow_trajectory = [compute_of(frames[0], frames[1])] + [compute_of(frames[i - 1], frames[i]) for i in range(1, len(frames))]
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