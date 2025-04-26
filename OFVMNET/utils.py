import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from eval import Eval
import os
import ast

def create_feature_to_file_dicts(vid_path, vid_feature_path, aud_path, aud_feature_path):
    """
    Create dictionaries mapping feature file paths to raw file paths.

    Args:
        vid_path (str): Path to the directory containing video files.
        vid_feature_path (str): Path to the directory containing video feature files.
        aud_path (str): Path to the directory containing audio files.
        aud_feature_path (str): Path to the directory containing audio feature files.

    Returns:
        vid_dict (dict): Mapping from video feature file paths to video file paths.
        aud_dict (dict): Mapping from audio feature file paths to audio file paths.
    """
    vid_files = [os.path.join(vid_path, x) for x in os.listdir(vid_path) if '.part' not in x]
    vid_features = [os.path.join(vid_feature_path, os.path.splitext(x)[0] + '.npy') for x in os.listdir(vid_path) if '.part' not in x]

    aud_files = [os.path.join(aud_path, x) for x in os.listdir(aud_path) if '.part' not in x]
    aud_features = [os.path.join(aud_feature_path, os.path.splitext(x)[0] + '.npy') for x in os.listdir(aud_path) if '.part' not in x]

    vid_dict = dict(zip(vid_features, vid_files))
    aud_dict = dict(zip(aud_features, aud_files))

    return vid_dict, aud_dict



def get_meta_df(video_feature_path, audio_feature_path, flow_rank_file, max_seq_len):
    def parse_ranks_str(ranks_str):
        # Remove tuple and quotes, then extract numbers
        cleaned = ranks_str.strip("(),'")  # removes parentheses, commas, quotes
        return list(map(int, cleaned.strip('[]').split()))

    video_files = os.listdir(video_feature_path)
    audio_files = os.listdir(audio_feature_path)

    video_file_map = {filename.split('.')[0]: os.path.join(video_feature_path, filename) for filename in video_files }
    audio_file_map = {filename.split('.')[0]: os.path.join(audio_feature_path, filename) for filename in audio_files}
    
    flow_ranks = pd.read_csv(flow_rank_file)
    flow_ranks_dict = dict(zip(flow_ranks['vid'], flow_ranks['ranks']))
    
    vid_df = pd.DataFrame(list(video_file_map.items()), columns=['vid', 'vid_filename'])
    aud_df = pd.DataFrame(list(audio_file_map.items()), columns=['vid', 'aud_filename'])
    flow_ranks = pd.read_csv(flow_rank_file)[['vid', 'ranks', 'segments']]

    df = vid_df.merge(aud_df, on='vid').merge(flow_ranks, on='vid')

    df['ranks'] = df['ranks'].apply(
        lambda x: parse_ranks_str(x) if isinstance(x, str) else list(map(int, x)))

    # drop entire videos who's max sequence length doesn't comply
    df = df[df['segments'].apply(lambda s: max(np.diff(ast.literal_eval(s))) <= max_seq_len)].reset_index(drop=True)

    return df

def custom_collate(batch):
    videos, audios, segments, ranks = zip(*batch)
    # Return as lists so you can deal with variable shapes manually
    return list(videos), list(audios), list(segments), list(ranks)

def custom_collate_testset(batch):
    videos, audios, segments, vidfile, audfile = zip(*batch)
    # Return as lists so you can deal with variable shapes manually
    return list(videos), list(audios), list(segments), list(vidfile), list(audfile)

def perform_feature_padding(video_features, audio_features, start_segment, end_segment, max_seq_len):
    #vf = video_features.clone().detach()
    #af = audio_features.clone().detach()

    vf =video_features[start_segment:end_segment,:]
    af = audio_features[start_segment:end_segment,:]

    pvf = torch.zeros(max_seq_len, vf.shape[1])
    pvf[:vf.shape[0], :] = torch.tensor(vf)

    paf = torch.zeros(max_seq_len, af.shape[1])
    paf[:af.shape[0], :] = torch.tensor(af)

    # Create mask (True for padding positions)
    mask = torch.arange(max_seq_len) >= vf.shape[0]
    mask = mask.unsqueeze(0)  # Convert to 2D (batch_size=1, seq_len)
    return pvf, paf, mask


def save_checkpoint(model, optimizer, epoch, filename):
    """Saves model and optimizer state dict."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'model_args': model.args,
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch} to {filename}")


def get_segmentd_embeddings(video_model, audio_model, vid, aud, max_seq_len, segments):

    vid_segment_embeddings = []
    aud_segment_embeddings = []
    for i in range(1, len(segments)):
        start = segments[i-1]
        end = segments[i]

        vid_emb, aud_emb, mask = perform_feature_padding(vid, aud, start, end, max_seq_len)
        
        vid_segment_embeddings.append(video_model(vid_emb, mask))
        aud_segment_embeddings.append(audio_model(aud_emb, mask))
    return vid_segment_embeddings, aud_segment_embeddings



def get_batch_embeddings(video_model, audio_model, video_batch, audio_batch, max_seq_len, segments):
    # We precompute the segment embeddings in each batch.  We do this once and then proceed to processing batch
    batch_vid_embeddings = []
    batch_aud_embeddings = []
    for i in range(len(video_batch)):
        vid = video_batch[i]
        aud = audio_batch[i]

        vid_sgmt_emb, aud_sgmt_emb = get_segmentd_embeddings(video_model, audio_model, vid, aud, max_seq_len, segments[i])
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
                        segments, min_frames, path, filenames, epoch, ks=[1, 2]):

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
        

    mean_recalls = np.mean(recalls, axis=0)
    print(f'Mean Recall@{ks}: {mean_recalls}')
    print(f'Mean AV-Align: {np.mean(av_aligns)}')
    print(f'Mean FAD: {np.mean(fads)}')
    
    tmp = pd.DataFrame({'epoch':epoch,
       'fad':np.mean(av_aligns),
       'av_align':np.mean(fads)
      }, index=[0])

    mean_recalls = np.mean(recalls, axis=0)
    for i, k in enumerate(ks):
        tmp[f'recall@{k}'] = mean_recalls[i]
        
    return tmp
