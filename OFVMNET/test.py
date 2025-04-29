import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import torch.optim as optim

import itertools
import random

import argparse
import tqdm

import utils
from models import Transformer
from DataLoader import VideoAudioTestset
from torch.utils.data import DataLoader, Dataset

import metrics


if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)

    parser = argparse.ArgumentParser(description='Read file content.')

    # TRANSFORMER PARAMS
    parser.add_argument("-ms", "--max_seq_len", type=int, default=200, help='Max sequence laength for Transformer Encoders')
    parser.add_argument("-bs", "--batch_size", type=int, default=128, help='Train Batch Size')

    # Longleaf
    parser.add_argument("-vmp", "--model_path", type=str, default='/work/users/s/m/smerrill/OFVMNET/models/resnet', help='model checkpoint path')        

    parser.add_argument("-sp", "--save_path", type=str, default='/work/users/s/m/smerrill/OFVMNET/models/resnet/', help='save path')    


    parser.add_argument("-rvp", "--raw_video_path", type=str, default='/work/users/s/m/smerrill/Youtube8m/video', help='Path to video Features File')
    parser.add_argument("-rap", "--raw_audio_path", type=str, default='/work/users/s/m/smerrill/Youtube8m/audio', help='Path to video Features File')
    
    parser.add_argument("-vfp", "--video_feature_path", type=str, default='/work/users/s/m/smerrill/Youtube8m/resnet/resnet101', help='Path to video Features File')
    parser.add_argument("-afp", "--audio_feature_path", type=str, default='/work/users/s/m/smerrill/Youtube8m/vggish', help='Path to audio Features File')
    parser.add_argument("-frf", "--flow_ranks_file", type=str, default='/work/users/s/m/smerrill/Youtube8m/flow/ranks.csv', help='Path to OF ranks file')

    
    args = vars(parser.parse_args())

    # make checkpoint dir
    os.makedirs(args['save_path'], exist_ok=True)
    summary_file = os.path.join(args['save_path'], "summaries.txt")

    video_model_path, audio_model_path = utils.get_latest_models(args['model_path'])
    video_checkpoint = torch.load(video_model_path)
    video_model = Transformer(**video_checkpoint['model_args'])
    video_model.load_state_dict(video_checkpoint['model_state_dict'])

    audio_checkpoint = torch.load(audio_model_path)
    audio_model = Transformer(**audio_checkpoint['model_args'])
    audio_model.load_state_dict(audio_checkpoint['model_state_dict'])
    print("Video and Audio Models Loaded!")


    meta_df = utils.get_meta_df(args['video_feature_path'], args['audio_feature_path'], args['flow_ranks_file'], args['max_seq_len'])
    print(f"Total Training Examples: {len(meta_df)}")

    vid_dict, aud_dict = utils.create_feature_to_file_dicts(args['raw_video_path'], args['video_feature_path'], args['raw_audio_path'], args['audio_feature_path'])
    dataset = VideoAudioTestset(meta_df, vid_dict, aud_dict)
    dataloader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True, collate_fn=utils.custom_collate_testset)


    ks = [1, 5, 10]
    for batch, (video_batch, audio_batch, segments, vidfiles, audfiles) in enumerate(dataloader):
        print(f"Starting Test Batch {batch}")
        batch_aud_embeddings, batch_vid_embeddings = utils.get_batch_embeddings(video_model, audio_model, video_batch, audio_batch, args['max_seq_len'], segments)

        # These were in (#segments*batchsize, 256)
        # Now they are in (batchsize, 256 * #segments)
        batch_vid_embeddings = batch_vid_embeddings.reshape(args['batch_size'], -1)
        batch_aud_embeddings = batch_aud_embeddings.reshape(args['batch_size'], -1)

        # we are going to do a naive cosine similarity based retrieval strategy
        similarity_matrix = torch.matmul(batch_vid_embeddings, batch_aud_embeddings.T)

        # Get the most similar audio embeddings for each video
        _, most_similar_indices = torch.max(similarity_matrix, dim=1)

        # recall@k
        recall = [metrics.top_k_recall(similarity_matrix, k) for k in ks]
        print(f"Recalls {recall}")

        retrieved_audio_embeddings = batch_aud_embeddings[most_similar_indices]

        # Compute FAD/AV-Align
        gots = []
        retrievals = []
        av_aligns = []
        got_aligns = []
        for idx, retrieval in enumerate(most_similar_indices):
            # for FAD
            gots.append(batch_aud_embeddings[idx])
            retrievals.append(batch_aud_embeddings[retrieval])

            # Av-align
            frames, fps = metrics.extract_frames(vidfiles[idx])
            _, video_peaks = metrics.detect_video_peaks(frames, fps)

            audio_peaks1 = metrics.detect_audio_peaks(audfiles[retrieval])
            audio_peaks2 = metrics.detect_audio_peaks(audfiles[idx])
            av_align = metrics.calc_intersection_over_union(audio_peaks1, video_peaks, fps)
            got_av_align = metrics.calc_intersection_over_union(audio_peaks2, video_peaks, fps)

            print(f'idx {idx}, AV-ALIGN: {av_align}, GOT-AV-ALIGN: {got_av_align}')
            av_aligns.append(av_align)
            got_aligns.append(got_av_align)

            
        fad_score = metrics.compute_fad(torch.stack(gots).detach(), torch.stack(retrievals).detach())
        print(f'FAD: {fad_score}')

        with open(summary_file, "a") as f:
            f.write(f"Batch {batch}\n")
            f.write(f"Recalls (for ks={ks}): {recall}\n")
            f.write(f"FAD: {fad_score.item() if hasattr(fad_score, 'item') else fad_score}\n")
            f.write(f"Average AV-ALIGN: {sum(av_aligns) / len(av_aligns)}\n")
            f.write(f"Average GOT-AV-ALIGN: {sum(got_aligns) / len(got_aligns)}\n")
            f.write("-" * 50 + "\n")
        print("File Saved")
