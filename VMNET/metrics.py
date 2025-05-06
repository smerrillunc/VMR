import torch
import numpy as np
from scipy import linalg
import cv2
import numpy as np
import torchaudio
from scipy.signal import find_peaks


## AV-ALIGN Functions
# Function to extract frames from a video file
def detect_video_peaks(frames, fps, threshold=1, distance=3):
    """
    Detect motion peaks using optical flow and find_peaks.
    
    Args:
        frames: list of grayscale frames
        fps: frames per second (adjusted for skipping)
        threshold: minimum peak height (motion strength)
        distance: minimum number of frames between peaks

    Returns:
        flow_trajectory: List of average flow magnitudes
        peak_times: Timestamps of detected peaks (in seconds)
    """
    distance = distance * fps
    # Compute optical flow magnitudes between consecutive frames
    flow_trajectory = [
        compute_of(frames[i - 1], frames[i])
        for i in range(1, len(frames))
    ]

    # Use scipy to detect peaks
    peak_indices, properties = find_peaks(
        flow_trajectory,
        height=threshold,       # Minimum motion strength
        distance=distance       # Minimum distance between peaks
    )

    # Convert frame indices to time
    peak_times = [idx / fps for idx in peak_indices]

    return flow_trajectory, peak_times


def extract_frames(video_path, frame_skip=5, downsample_factor=2):
    """Extract grayscale, downsampled frames from video with frame skipping."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Unable to open video {video_path}.")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_idx = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_idx % frame_skip == 0:
            if downsample_factor > 1:
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (w // downsample_factor, h // downsample_factor))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        frame_idx += 1

    cap.release()

    effective_fps = original_fps / frame_skip
    return frames, effective_fps

def detect_audio_peaks(audio_file):
    waveform, sr = torchaudio.load(audio_file)
    waveform = waveform.mean(dim=0)  # Convert to mono

    # Compute the short-time energy
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = int(0.010 * sr)    # 10ms hop

    energy = waveform.unfold(0, frame_length, hop_length).pow(2).mean(dim=1)

    # Normalize energy
    energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-6)

    # Detect peaks: simple threshold
    peaks = (energy[1:-1] > energy[:-2]) & (energy[1:-1] > energy[2:]) & (energy[1:-1] > 0.3)
    peak_indices = torch.nonzero(peaks).squeeze() + 1  # +1 because of slicing

    onset_times = (peak_indices * hop_length) / sr
    return onset_times.tolist()


# Function to find local maxima with threshold
def find_local_max_indexes(arr, fps):
    arr = torch.tensor(arr)
    local_max_mask = (arr[1:-1] > arr[:-2]) & (arr[1:-1] > arr[2:]) & (arr[1:-1] >= 0.1)
    indexes = torch.where(local_max_mask)[0] + 1  # offset by 1 due to slicing
    times = indexes.float() / fps
    return times.tolist()

# Function to compute the optical flow magnitude between two frames
def compute_of(prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
    avg_magnitude = magnitude.mean()
    return avg_magnitude



# Function to calculate Intersection over Union (IoU)
def calc_intersection_over_union(audio_peaks, video_peaks, fps):
    intersection_length = 0
    used_video_peaks = [False] * len(video_peaks)
    for audio_peak in audio_peaks:
        for j, video_peak in enumerate(video_peaks):
            if not used_video_peaks[j] and abs(video_peak - audio_peak) <= (1 / fps):
                intersection_length += 1
                used_video_peaks[j] = True
                break
    union = len(audio_peaks) + len(video_peaks) - intersection_length
    return intersection_length / union if union > 0 else 0


### FAD Function
def compute_covariance(matrix):
    """Compute covariance matrix manually (samples in rows)."""
    matrix = matrix - matrix.mean(dim=0)
    return (matrix.T @ matrix) / (matrix.size(0) - 1)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate the Fréchet Distance between two multivariate Gaussians."""
    mu1 = mu1.cpu().numpy()
    mu2 = mu2.cpu().numpy()
    sigma1 = sigma1.cpu().numpy()
    sigma2 = sigma2.cpu().numpy()

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        print("Adding epsilon to diagonal of covariance matrices for stability.")
        sigma1 += np.eye(sigma1.shape[0]) * eps
        sigma2 += np.eye(sigma2.shape[0]) * eps
        covmean = linalg.sqrtm(sigma1 @ sigma2)

    # Numerical error might give slight imaginary part
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    distance = (diff @ diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return torch.tensor(distance, dtype=torch.float32)


def compute_fad(real_embeddings, retrieved_embeddings):
    """Compute FAD between real and generated embeddings."""
    mu_real = real_embeddings.mean(dim=0)
    sigma_real = compute_covariance(real_embeddings.T)

    mu_gen = retrieved_embeddings.mean(dim=0)
    sigma_gen = compute_covariance(retrieved_embeddings.T)

    fad = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    return fad


## KLD
def calculate_kl_divergence(mu1, logvar1, mu2, logvar2):
    """
    Calculate KL divergence between two diagonal Gaussians:
    N(mu1, sigma1^2) and N(mu2, sigma2^2)

    logvar1/logvar2: log of variance vectors (assumed diagonal)
    Returns scalar KL divergence.
    """
    var1 = logvar1.exp()
    var2 = logvar2.exp()
    kl = 0.5 * (logvar2 - logvar1 + (var1 + (mu1 - mu2).pow(2)) / var2 - 1)
    return kl.sum()  # sum over dimensions

def compute_gaussian_stats(embeddings, eps=1e-6):
    """Returns mean and log-variance for embeddings assuming diagonal Gaussian."""
    mu = embeddings.mean(dim=0)
    var = embeddings.var(dim=0, unbiased=False) + eps
    logvar = var.log()
    return mu, logvar

def compute_kld(real_embeddings, retrieved_embeddings):
    """
    Compute KL divergence between two sets of embeddings modeled as diagonal Gaussians.
    KL(N_real || N_retrieved)
    """
    mu_real, logvar_real = compute_gaussian_stats(real_embeddings)
    mu_gen, logvar_gen = compute_gaussian_stats(retrieved_embeddings)

    kld = calculate_kl_divergence(mu_real, logvar_real, mu_gen, logvar_gen)
    return kld

