import numpy as np
import pandas as pd
import os

import tensorflow as tf
import numpy as np
from IPython.display import YouTubeVideo

import requests
import json

import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from scipy.signal import peak_prominences
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import itertools
import random

import argparse
from OFProcessor import OpticalFlowProcessor
from DataLoader import VideoAudioDataset



def collate_fn(batch, processor):
    video_batch, audio_batch = zip(*batch)
    video_batch = [torch.tensor(v, dtype=torch.float32) for v in video_batch]
    audio_batch = [torch.tensor(a, dtype=torch.float32) for a in audio_batch]
    flow_ranks = [processor.get_of_ranks(video_batch[i], audio_batch[i]) for i in range(len(video_batch))]
    return video_batch, audio_batch, flow_ranks

def get_dataloader(path, batch_size=32, shuffle=True, method='video', window_size=20):
    dataset = VideoAudioDataset(path)
    processor = OpticalFlowProcessor(method=method, window_size=window_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda batch: collate_fn(batch, processor))

def perform_feature_padding(video_features, audio_features, start_segment, end_segment, max_seq_len):
    vf = torch.tensor(video_features[start_segment:end_segment,:])
    af = torch.tensor(audio_features[start_segment:end_segment,:])

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


def get_segmentd_embeddings(video_model, audio_model, vid, aud, max_seq_len):
    vid_segment_embeddings = []
    

    of = OpticalFlowProcessor()
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



def get_batch_embeddings(video_model, audio_model, video_batch, audio_batch, max_seq_len):
    # We precompute the segment embeddings in each batch.  We do this once and then proceed to processing batch
    batch_vid_embeddings = []
    batch_aud_embeddings = []
    for i in range(len(video_batch)):
        vid = video_batch[i]
        aud = audio_batch[i]
        vid_sgmt_emb, aud_sgmt_emb = get_segmentd_embeddings(video_model, audio_model, vid, aud, max_seq_len)
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


# In[200]:


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
    
    topk_pos_values, _ = torch.topk(positive_pairs.flatten(), k)
    topk_neg_values, _ = torch.topk(negative_pairs.flatten(), k)
    
    # max accross each pos/neg pair
    loss = torch.max(topk_pos_values - topk_neg_values, torch.tensor(min_val))
    return torch.mean(loss)

