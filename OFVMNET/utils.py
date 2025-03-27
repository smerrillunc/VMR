import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from OFProcessor import OpticalFlowProcessor
from DataLoader import VideoAudioDataset
from torch.utils.data import DataLoader, Dataset
from eval import Eval

def collate_fn(batch, processor):
    video_batch, audio_batch = zip(*batch)
    video_batch = [torch.tensor(v, dtype=torch.float32) for v in video_batch]
    audio_batch = [torch.tensor(a, dtype=torch.float32) for a in audio_batch]
    flow_ranks = [processor.get_of_ranks(video_batch[i], audio_batch[i]) for i in range(len(video_batch))]
    return video_batch, audio_batch, flow_ranks


def get_dataloader(path, filenames, batch_size=32, shuffle=True, method='video', window_size=20):
    dataset = VideoAudioDataset(path, filenames)
    processor = OpticalFlowProcessor(method=method, window_size=window_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda batch: collate_fn(batch, processor))


def perform_feature_padding(video_features, audio_features, start_segment, end_segment, max_seq_len):
    vf = video_features.clone().detach()
    af = audio_features.clone().detach()
    vf =vf[start_segment:end_segment,:]
    af = af[start_segment:end_segment,:]

    pvf = torch.zeros(max_seq_len, 1024)
    pvf[:vf.shape[0], :] = vf

    paf = torch.zeros(max_seq_len, 128)
    paf[:af.shape[0], :] = af

    # Create mask (True for padding positions)
    mask = torch.arange(max_seq_len) >= vf.shape[0]
    mask = mask.unsqueeze(0)  # Convert to 2D (batch_size=1, seq_len)
    return pvf, paf, mask


def save_checkpoint(model, optimizer, epoch, filename):
    """Saves model and optimizer state dict."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch} to {filename}")


def get_segmentd_embeddings(video_model, audio_model, vid, aud, max_seq_len, window_size, segments, min_frames):
    vid_segment_embeddings = []
    
    of = OpticalFlowProcessor(method='video', window_size=window_size, segments=segments, min_frames=min_frames)
    flow = of._compute_flow(vid, aud)
    segments = of._optical_flow_segments(flow)

    vid_segment_embeddings = []
    aud_segment_embeddings = []
    for i in range(1, len(segments)):
        start = segments[i-1]
        end = segments[i]

        vid_emb, aud_emb, mask = perform_feature_padding(vid, aud, start, end, max_seq_len)
        
        vid_segment_embeddings.append(video_model(vid_emb, mask))
        aud_segment_embeddings.append(audio_model(aud_emb, mask))
    return vid_segment_embeddings, aud_segment_embeddings



def get_batch_embeddings(video_model, audio_model, video_batch, audio_batch, max_seq_len, window_size, segments, min_frames):
    # We precompute the segment embeddings in each batch.  We do this once and then proceed to processing batch
    batch_vid_embeddings = []
    batch_aud_embeddings = []
    for i in range(len(video_batch)):
        vid = video_batch[i]
        aud = audio_batch[i]

        vid_sgmt_emb, aud_sgmt_emb = get_segmentd_embeddings(video_model, audio_model, vid, aud, max_seq_len, window_size, segments, min_frames)
        batch_vid_embeddings.extend(vid_sgmt_emb)
        batch_aud_embeddings.extend(aud_sgmt_emb)
        
    # Shape will by (total segments X embedding dim)
    # total segments is clip dependent
    batch_aud_embeddings = torch.stack(batch_aud_embeddings)
    batch_vid_embeddings = torch.stack(batch_vid_embeddings)
    
    # MAKE SURE VECTORS ARE NORMALIZED FIRST idk if I want to do here or later..
    batch_aud_embeddings = torch.nn.functional.normalize(batch_aud_embeddings, p=2, dim=1)
    batch_vid_embeddings = torch.nn.functional.normalize(batch_vid_embeddings, p=2, dim=1)

    return batch_aud_embeddings, batch_vid_embeddings



def get_intermodal_loss(batch_vid_embeddings, batch_aud_embeddings, k=5, min_val=0):
    # batch_vid_embeddings and  batch_aud_embeddings should already be normalized so 
    # multiplying them is a similarity metric

    # convert simliarity to distance by (-1) >> high value indicates the distance between the samples is long
    dist_xy = (-1) *torch.matmul(batch_vid_embeddings, batch_aud_embeddings.T)

    positive_pairs = torch.diag(dist_xy)

    # Get non-diagonal elements (negative examples)
    # First, create a mask for non-diagonal elements
    mask = ~torch.eye(dist_xy.size(0), dtype=torch.bool)

    # Apply the mask to extract non-diagonal elements
    negative_pairs = dist_xy[mask]

    # First we find the positive pairs that are furthest in embedding space
    topk_pos_values, _ = torch.topk(positive_pairs.flatten(), k, largest=True)

    # next we find the negative pairs that are closest in embedding space
    topk_neg_values, _ = torch.topk(negative_pairs.flatten(), k, largest=False)

    # expand so we compare all possible combinations of pos/neg pairs
    topk_pos_values_expanded = topk_pos_values.unsqueeze(1)  # Shape: (k, 1)
    topk_neg_values_expanded = topk_neg_values.unsqueeze(0)  # Shape: (1, k)

    # min_val or this loss
    loss = torch.maximum(torch.tensor(min_val), topk_pos_values_expanded - topk_neg_values_expanded)
    loss = loss.mean()
    return loss

def compute_evaluations(video_model, audio_model, batch_size, max_seq_len, window_size, \
                        segments, min_frames, path, filenames, epoch, ks=[1, 5]):

    metrics = Eval()
    testloader = get_dataloader(path, filenames, batch_size=batch_size, shuffle=True, method='video', window_size=window_size)
    audio_model.eval()
    video_model.eval()

    recalls = []
    fads = []
    av_aligns = []

    
    for video_batch, audio_batch, flow_ranks in testloader:
        batch_aud_embeddings, batch_vid_embeddings = get_batch_embeddings(video_model, audio_model, video_batch, audio_batch, max_seq_len, window_size, segments, min_frames)

        # These were in (#segments*batchsize, 256)
        # Now they are in (batchsize, 256 * #segments)
        batch_vid_embeddings = batch_vid_embeddings.reshape(batch_size, -1)
        batch_aud_embeddings = batch_aud_embeddings.reshape(batch_size, -1)

        # we are going to do a naive cosine similarity based retrieval strategy
        similarity_matrix = torch.matmul(batch_vid_embeddings, batch_aud_embeddings.T)

        # Get the most similar audio embeddings for each video
        _, most_similar_indices = torch.max(similarity_matrix, dim=1)

        # recall@k
        recall = [metrics.top_k_recall(similarity_matrix, k) for k in ks]

        retrieved_audio_embeddings = batch_aud_embeddings[most_similar_indices]
        fad = metrics.calculate_frechet_audio_distance(batch_aud_embeddings, retrieved_audio_embeddings)
        av_align = metrics.compute_av_align_score(batch_vid_embeddings, retrieved_audio_embeddings, prominence_threshold=0.1)


        recalls.append(recall)
        fads.append(fad)
        av_aligns.append(av_align)
        break
    mean_recalls = np.mean(recalls, axis=0)
    print(f'Mean Recall@{ks}: {mean_recalls}')
    print(f'Mean AV-Align: {np.mean(av_aligns)}')
    print(f'Mean AV-Align: {np.mean(fads)}')
    
    tmp = pd.DataFrame({'epoch':epoch,
       'fad':np.mean(av_aligns),
       'av_align':np.mean(fads)
      }, index=[0])

    mean_recalls = np.mean(recalls, axis=0)
    for i, k in enumerate(ks):
        tmp[f'recall@{k}'] = mean_recalls[i]
        
    return tmp
