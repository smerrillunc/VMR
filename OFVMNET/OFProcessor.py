import numpy as np
#from scipy.signal import peak_prominences
#from scipy.signal import find_peaks
import ruptures as rpt

class OpticalFlowProcessor:
    def __init__(self, method='video', window_size=10, segments=20, min_frames=5):
        self.method = method
        self.window_size = window_size
        self.segments = segments
        self.min_frames = min_frames

    def get_of_ranks(self, rgb, audio):
        flow = self._compute_flow(rgb, audio)
        segments = self._optical_flow_segments(flow)
        ranks = self._rank_averages(self._compute_segment_means(segments, flow))
        return ranks

    def get_best_worst_flow(self, rgb, audio):
        flow = self._compute_flow(rgb, audio)
        segments = self._optical_flow_segments(flow)
        ranks = self._rank_averages(self._compute_segment_means(segments, flow))
        return self._extract_best_worst_segments(segments, ranks)

    def _compute_flow(self, rgb, audio):
        if self.method == 'video':
            return self._moving_average(self._calculate_optical_flow_euclidean(rgb))
        elif self.method == 'audio':
            return self._moving_average(self._calculate_optical_flow_euclidean(audio))
        else:
            raise ValueError("Method must be 'video' or 'audio'")

    @staticmethod
    def _calculate_optical_flow_euclidean(embedding_seq):
        return np.linalg.norm(embedding_seq[1:] - embedding_seq[:-1], axis=1)

    @staticmethod
    def _moving_average(arr, window_size=5):
        return np.convolve(arr, np.ones(window_size) / window_size, mode='valid')

    def _optical_flow_segments_old(self, optical_flow):
        peaks, _ = find_peaks(optical_flow)
        prominences = peak_prominences(optical_flow, peaks)[0]
        peak_index = peaks[np.argsort(prominences)[-self.max_segments:]]
        peak_index = self._merge_intervals(np.sort(peak_index))
        return np.insert(np.append(peak_index, len(optical_flow)), 0, 0)

    def _optical_flow_segments(self, optical_flow_video, max_seq_len=100):

        # minimum of 10 seconds of video
        algo = rpt.Dynp(model='l2', min_size=self.min_frames, jump=3).fit(optical_flow_video)
        change_points = algo.predict(n_bkps=self.segments)  # The 'pen' parameter controls sensitivity

        change_points.insert(0,0)
        return change_points

    
    def _merge_intervals(self, arr):
        merged = [arr[0]]
        for i in range(1, len(arr)):
            if arr[i] - merged[-1] >= self.min_frames:
                merged.append(arr[i])
        return np.array(merged)

    @staticmethod
    def _compute_segment_means(segments, values):
        return [values[start:end].mean() if start < end else 0 for start, end in zip(segments[:-1], segments[1:])]

    @staticmethod
    def _rank_averages(averages):
        sorted_indices = np.argsort(averages)[::-1]
        ranks = np.zeros_like(sorted_indices) + 1
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank + 1
        return ranks

    def _extract_best_worst_segments(self, segments, ranks):
        top_start, top_end = segments[np.where(ranks == 1)[0][0]], segments[np.where(ranks == 1)[0][0] + 1]
        bottom_start, bottom_end = segments[np.where(ranks == max(ranks))[0][0]], segments[np.where(ranks == max(ranks))[0][0] + 1]
        return (top_start, top_end), (bottom_start, bottom_end)