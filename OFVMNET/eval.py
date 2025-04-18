import torch
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import peak_prominences
import itertools

class Eval():
    def __init__(self):
        pass

    # Function to detect local maxima in embeddings using scipy's find_peaks
    def find_local_maxima_in_embeddings(self, embeddings, prominence_threshold=0.1):
        """
        Detect local maxima in the embeddings using scipy's find_peaks and sklearn's peak_prominences.

        Args:
            embeddings (torch.Tensor): Tensor of shape (batch_size, feature_dim).
            prominence_threshold (float): Minimum prominence required to consider a peak.

        Returns:
            peaks_list (list): List of indices where the local maxima (peaks) occur in the embeddings.
        """
        embeddings = embeddings.cpu().detach().numpy()  # Convert to numpy for peak detection

        peaks_list = []
        for i in range(embeddings.shape[0]):  # Iterate through each embedding (e.g., audio or video)
            # Find local maxima (peaks) in the embedding
            peaks, _ = find_peaks(embeddings[i])
            prominences = peak_prominences(embeddings[i], peaks)[0]

            # Filter peaks based on prominence
            significant_peaks = peaks[prominences >= prominence_threshold]
            peaks_list.append(significant_peaks)

        return peaks_list

    # Function to calculate Intersection over Union (IoU) between audio and video peaks
    def calc_intersection_over_union(self, audio_peaks, video_peaks):
        """
        Calculate Intersection over Union (IoU) between the audio and video peaks.

        Args:
            audio_peaks (list): Indices of audio peaks.
            video_peaks (list): Indices of video peaks.

        Returns:
            float: IoU score between audio and video peaks.
        """
        intersection = len(set(audio_peaks).intersection(set(video_peaks)))
        union = len(set(audio_peaks).union(set(video_peaks)))
        iou_score = intersection / union
        return iou_score

    # Function to compute the AV-Align score from audio and video embeddings
    def compute_av_align_score(self, audio_embeddings, video_embeddings, prominence_threshold=0.1):
        """
        Compute the AV-Align score between the audio and video embeddings.

        Args:
            audio_embeddings (torch.Tensor): Audio embeddings of shape (batch_size, feature_dim).
            video_embeddings (torch.Tensor): Video embeddings of shape (batch_size, feature_dim).
            prominence_threshold (float): Minimum prominence to filter significant peaks.

        Returns:
            float: AV-Align score (IoU between audio and video peaks).
        """
        # Detect peaks in the audio and video embeddings
        audio_peaks = self.find_local_maxima_in_embeddings(audio_embeddings, prominence_threshold)
        video_peaks = self.find_local_maxima_in_embeddings(video_embeddings, prominence_threshold)

        # Flatten the list of peaks before calculating the IoU
        audio_peaks_flattened = list(itertools.chain(*audio_peaks))
        video_peaks_flattened = list(itertools.chain(*video_peaks))

        # Calculate the Intersection over Union (IoU) for the audio and video peaks
        iou_score = self.calc_intersection_over_union(audio_peaks_flattened, video_peaks_flattened)

        return iou_score

    def compute_mean_and_covariance(self, embeddings):
        """
        Computes the mean and covariance matrix of the embeddings.

        Args:
            embeddings (torch.Tensor): A tensor of shape (num_samples, feature_dim)

        Returns:
            mean (torch.Tensor): The mean of the embeddings.
            covariance (torch.Tensor): The covariance matrix of the embeddings.
        """
        mean = embeddings.mean(dim=0)
        centered_embeddings = embeddings - mean
        covariance = torch.matmul(centered_embeddings.T, centered_embeddings) / (embeddings.size(0) - 1)
        return mean, covariance

    def calculate_frechet_audio_distance(self, embeddings_ground_truth, embeddings_retrieved, epsilon=1e-6):
        """
        Calculates the Fréchet Audio Distance (FAD) between two sets of embeddings.

        Args:
            embeddings_ground_truth (torch.Tensor): Ground truth audio embeddings (num_samples, feature_dim).
            embeddings_retrieved (torch.Tensor): Retrieved audio embeddings (num_samples, feature_dim).
            epsilon (float): Regularization term to ensure positive semi-definiteness of covariance matrices.

        Returns:
            fad (float): The Fréchet Audio Distance between the two sets of embeddings.
        """
        # Compute the mean and covariance for ground truth and retrieved embeddings
        mu_x, sigma_x = self.compute_mean_and_covariance(embeddings_ground_truth)
        mu_y, sigma_y = self.compute_mean_and_covariance(embeddings_retrieved)

        # Calculate the term: ||mu_x - mu_y||^2
        diff_mu = mu_x - mu_y
        mu_term = torch.sum(diff_mu ** 2)

        # Add epsilon to the covariance matrices to ensure they are positive semi-definite
        sigma_x += epsilon * torch.eye(sigma_x.size(0), device=sigma_x.device)
        sigma_y += epsilon * torch.eye(sigma_y.size(0), device=sigma_y.device)

        # Perform Cholesky decomposition to compute the matrix square root of covariance matrices
        try:
            sigma_x_sqrt = torch.linalg.cholesky(sigma_x)
            sigma_y_sqrt = torch.linalg.cholesky(sigma_y)
        except RuntimeError:
            raise ValueError("Covariance matrices are not positive semi-definite, even with regularization")
            return float('inf')

        # Calculate the term: Tr(sigma_x + sigma_y - 2(sigma_x^0.5 * sigma_y * sigma_x^0.5)^0.5)
        term = torch.trace(sigma_x + sigma_y - 2 * torch.matmul(sigma_x_sqrt, torch.matmul(sigma_y_sqrt, sigma_x_sqrt.T)))

        # The FAD is the sum of the two terms
        fad = mu_term + term
        return fad.item()

    def top_k_recall(self, similarity_matrix, k):
        """
        Compute top-k recall from a similarity matrix (vectorized version).

        similarity_matrix: Tensor of shape (batch_size, batch_size), where
                           diagonal elements represent positive pairs.
        k: The value of k for top-k recall.

        Returns:
            recall: The top-k recall (mean recall across all examples).
        """
        batch_size = similarity_matrix.size(0)

        # Get the indices of the top-k most similar items for each row (video/audio example)
        _, top_k_indices = torch.topk(similarity_matrix, k, dim=1)

        # Create a tensor for diagonal indices (i, i) for each row
        diagonal_indices = torch.arange(batch_size, device=similarity_matrix.device)

        # Check if the diagonal index of each row is in the top-K indices of that row
        # `top_k_indices` is of shape (batch_size, k)
        is_true_positive_in_top_k = (top_k_indices == diagonal_indices.unsqueeze(1))

        # Calculate recall: for each row, check if the diagonal index is in the top-K
        recall_per_row = is_true_positive_in_top_k.any(dim=1).float()

        # Return the mean recall across all examples (rows)
        return recall_per_row.mean()
