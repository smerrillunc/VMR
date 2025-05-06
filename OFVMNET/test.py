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
import wandb
import cv2

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)

    parser = argparse.ArgumentParser(description='Read file content.')

    # TRANSFORMER PARAMS
    parser.add_argument("-ms", "--max_seq_len", type=int, default=200, help='Max sequence laength for Transformer Encoders')
    parser.add_argument("-bs", "--batch_size", type=int, default=128, help='Train Batch Size')

    # Longleaf
    parser.add_argument("-vmp", "--model_path", type=str, default='/work/users/s/m/smerrill/OFVMNET/models/resnet2', help='model checkpoint path')        

    parser.add_argument("-sp", "--save_path", type=str, default='/work/users/s/m/smerrill/OFVMNET/models/resnet/', help='save path')    


    parser.add_argument("-rvp", "--raw_video_path", type=str, default='/work/users/s/m/smerrill/Youtube8m/video', help='Path to video Features File')
    parser.add_argument("-rap", "--raw_audio_path", type=str, default='/work/users/s/m/smerrill/Youtube8m/audio', help='Path to video Features File')
    
    parser.add_argument("-vfp", "--video_feature_path", type=str, default='/work/users/s/m/smerrill/Youtube8m/resnet/resnet101', help='Path to video Features File')
    parser.add_argument("-afp", "--audio_feature_path", type=str, default='/work/users/s/m/smerrill/Youtube8m/vggish', help='Path to audio Features File')
    parser.add_argument("-frf", "--flow_ranks_file", type=str, default='/work/users/s/m/smerrill/Youtube8m/flow/ranks.csv', help='Path to OF ranks file')
    parser.add_argument("-tcsv", "--test_csv_file", type=str, default='/work/users/s/m/smerrill/Youtube8m/test_resnet.csv', help='Path to OF ranks file')

    parser.add_argument("-ps", "--pad_size", type=int, default=0, help='Pad Size')

    args = vars(parser.parse_args())

    features = args['video_feature_path'].split('/')[-1]
    dataset = args['video_feature_path'].split('smerrill/')[1].split('/')[0]
    run_name = f'{dataset}_{features}_{str(args['pad_size'])}'
    print(features, dataset)

    base_directory = os.path.dirname(args['flow_ranks_file'])  # Get the directory
    file_name = os.path.basename(args['flow_ranks_file'])  # Get the base file name (e.g., 'ranks.csv')

    # Modify the file name based on pad_size
    if args['pad_size'] == 0:
        flow_ranks_file = os.path.join(base_directory, file_name)
    else:
        file_name_with_pad = f'ranks_{args['pad_size']}.csv'  # Adjust the file name based on pad_size
        flow_ranks_file = os.path.join(base_directory, file_name_with_pad)


    wandb.init(
        project="OFVMNET",  # Change this to your preferred project name
        name=run_name,  # Set custom run name
        config={
            "padding": args['pad_size'],
            "dataset": dataset,
            "video_feature_dir": args['video_feature_path'],
            'flow_ranks_file':flow_ranks_file})

    # make checkpoint dir
    os.makedirs(args['save_path'], exist_ok=True)
    summary_file = os.path.join(args['save_path'], "summaries.txt")

    video_model_path, audio_model_path = utils.get_latest_models(args['model_path'])
    print(video_model_path, audio_model_path)
    video_checkpoint = torch.load(video_model_path, map_location=torch.device('cpu') )
    video_model = Transformer(**video_checkpoint['model_args'])
    video_model.load_state_dict(video_checkpoint['model_state_dict'])

    audio_checkpoint = torch.load(audio_model_path, map_location=torch.device('cpu') )
    audio_model = Transformer(**audio_checkpoint['model_args'])
    audio_model.load_state_dict(audio_checkpoint['model_state_dict'])
    print("Video and Audio Models Loaded!")

    # limit to 1600 examples for Youtube8m
    meta_df = utils.get_meta_df(args['video_feature_path'], args['audio_feature_path'], flow_ranks_file, args['max_seq_len'], args['test_csv_file'])
    print(f"Total Training Examples: {len(meta_df)}")

    vid_dict, aud_dict = utils.create_feature_to_file_dicts(args['raw_video_path'], args['video_feature_path'], args['raw_audio_path'], args['audio_feature_path'])
    dataset = VideoAudioTestset(meta_df, vid_dict, aud_dict, args['pad_size'])
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

        wandb.log({
            "batch": int(batch),
            "XY_R@1": recall[0],
            "XY_R@5": recall[1],
            "XY_R@10": recall[2],
        })

        retrieved_audio_embeddings = batch_aud_embeddings[most_similar_indices]

        # Compute FAD/AV-Align
        gots = []
        retrievals = []
        av_aligns = []
        got_aligns = []
        for idx, retrieval in enumerate(most_similar_indices):
            try:
                # for FAD
                gots.append(batch_aud_embeddings[idx])
                retrievals.append(batch_aud_embeddings[retrieval])
    
            except Exception as e:
                print(e)


        got_embeddings = torch.stack(gots).detach()
        retrieved_embeddings = torch.stack(retrievals).detach()     
        fad_score = metrics.compute_fad(got_embeddings, retrieved_embeddings)
        kld1_score = metrics.compute_kld(got_embeddings, retrieved_embeddings)
        kld2_score = metrics.compute_kld(retrieved_embeddings, got_embeddings)

        print(f'FAD: {fad_score}')
        wandb.log({
            "batch": int(batch),
            "FAD": fad_score,
            "KLD1": kld1_score,
            "KLD2": kld2_score,
        })


        for idx, retrieval in enumerate(most_similar_indices):
            try:
                video_file = vidfiles[idx]
                audio_file = audfiles[retrieval]
                got_audio_file = audfiles[idx]

                cap = cv2.VideoCapture(video_file)
                if not cap.isOpened():
                    raise ValueError(f"Error: Unable to open video {video_file}.")

                frame_skip = 5
                original_fps = cap.get(cv2.CAP_PROP_FPS)
                fps = original_fps / frame_skip


                video_peaks_file = video_file.replace('video', 'video_peaks').split('.')[0] + '.npy'
                audio_peaks_file = audio_file.replace('audio', 'audio_peaks').split('.')[0] + '.npy'
                got_audio_peaks_file = got_audio_file.replace('audio', 'audio_peaks').split('.')[0] + '.npy'
    
                # Av-align
                video_peaks = np.load(video_peaks_file)

                audio_peaks1 = np.load(audio_peaks_file)
                audio_peaks2 = np.load(got_audio_peaks_file)

                av_align = metrics.calc_intersection_over_union(audio_peaks1, video_peaks, fps)
                got_av_align = metrics.calc_intersection_over_union(audio_peaks2, video_peaks, fps)
    
                if got_av_align == 0:
                    print("Error computin AV-ALGIN")
                    
                print(f'idx {idx}, AV-ALIGN: {av_align}, GOT-AV-ALIGN: {got_av_align}')

                wandb.log({
                    "batch": int(batch),
                    "idx": int(idx),
                    "av_align": av_align,
                    "got_av_align": got_av_align,
                })

                av_aligns.append(av_align)
                got_aligns.append(got_av_align)
            except Exception as e:
                print(e)


        wandb.log({
            "batch": int(batch),
            "Average AV-ALIGN": sum(av_aligns) / len(av_aligns),
            "Average GOT-AV-ALIGN": sum(got_aligns) / len(got_aligns),
        })

        with open(summary_file, "a") as f:
            f.write(f"Batch {batch}\n")
            f.write(f"Recalls (for ks={ks}): {recall}\n")
            f.write(f"FAD: {fad_score.item() if hasattr(fad_score, 'item') else fad_score}\n")
            f.write(f"Average AV-ALIGN: {sum(av_aligns) / len(av_aligns)}\n")
            f.write(f"Average GOT-AV-ALIGN: {sum(got_aligns) / len(got_aligns)}\n")
            f.write("-" * 50 + "\n")
        print("File Saved")
