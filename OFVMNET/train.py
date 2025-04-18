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
from DataLoader import VideoAudioDataset
from torch.utils.data import DataLoader, Dataset

if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)

    parser = argparse.ArgumentParser(description='Read file content.')

    # TRANSFORMER PARAMS
    parser.add_argument("-ms", "--max_seq_len", type=int, default=200, help='Max sequence laength for Transformer Encoders')
    parser.add_argument("-nh", "--num_heads", type=int, default=1, help='Number of Heads for Transformer Encoders')
    parser.add_argument("-nl", "--num_layers", type=int, default=1, help='Number of Layers for Transformer Encoders')

    parser.add_argument("-ida", "--input_dim_audio", type=int, default=128, help='Audio input dimension')
    parser.add_argument("-idv", "--input_dim_video", type=int, default=2048, help='Video input dimension')
    parser.add_argument("-ed", "--embed_dim", type=int, default=32, help='Embedding dimension')

    # segmentation params
    parser.add_argument("-mf", "--min_frames", type=int, default=5, help='Minimum Frames in each Segment')
    parser.add_argument("-sg", "--segments", type=int, default=20, help='Number of segments to create in the video')

    # Loss Params
    parser.add_argument("-l1", "--lambda1", type=float, default=0.33, help='Inter-modal loss weight')
    parser.add_argument("-l2", "--lambda2", type=float, default=0.33, help='OF Top matching loss weight')
    parser.add_argument("-l3", "--lambda3", type=float, default=0.33, help='OF Bottom matching loss weight')
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4, help='Learning Rate for Encoders')
    parser.add_argument("-m", "--margin", type=float, default=0.1, help='Margin parameter for OF triplet loss')

    # Learning Params
    parser.add_argument("-bs", "--batch_size", type=int, default=10, help='Train Batch Size')
    parser.add_argument("-e", "--epochs", type=int, default=1000, help='Epochs')
    parser.add_argument("-k", "--top_k", type=int, default=10, help='Top k violating examples')
    parser.add_argument("-fk", "--flow_k", type=int, default=10, help='Mine this many top and bottom flow examples')

    # Longleaf
    parser.add_argument("-sp", "--save_path", type=str, default='/nas/longleaf/home/smerrill/PD/data', help='save path')
    parser.add_argument("-dp", "--data_path", type=str, default='/nas/longleaf/home/smerrill/PD/data', help='dataset path')
    
    parser.add_argument("-vfp", "--video_feature_path", type=str, default='/work/users/s/m/smerrill/Youtube8m/resnet/resnet101', help='Path to video Features File')
    parser.add_argument("-afp", "--audio_feature_path", type=str, default='/work/users/s/m/smerrill/Youtube8m/vggish', help='Path to audio Features File')
    parser.add_argument("-frf", "--flow_ranks_file", type=str, default='/work/users/s/m/smerrill/Youtube8m/flow/ranks.csv', help='Path to OF ranks file')

    
    args = vars(parser.parse_args())

    # make checkpoint dir
    os.makedirs(args['save_path'], exist_ok=True)
    audio_model = Transformer(input_dim=args['input_dim_audio'], embed_dim=args['embed_dim'], \
                             num_heads=args['num_heads'], num_layers=args['num_layers'], max_seq_len=args['max_seq_len'])

    video_model = Transformer(input_dim=args['input_dim_video'], embed_dim=args['embed_dim'], \
                             num_heads=args['num_heads'], num_layers=args['num_layers'], max_seq_len=args['max_seq_len'])

    # Define the Adam optimizer for the audio model
    audio_optimizer = optim.Adam(audio_model.parameters(), lr=args['learning_rate'])

    # Define the Adam optimizer for the video model
    video_optimizer = optim.Adam(video_model.parameters(), lr=args['learning_rate'])

    meta_df = utils.get_meta_df(args['video_feature_path'], args['audio_feature_path'], args['flow_ranks_file'])
    dataset = VideoAudioDataset(meta_df)
    dataloader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=True, collate_fn=utils.custom_collate)

    #train_filenames = pd.read_csv(args['data_path']+'/train.csv')['filename'].values
    #test_filenames = pd.read_csv(args['data_path']+'/test.csv')['filename'].values
    #dataloader = utils.get_dataloader(args['data_path'], train_filenames, batch_size=args['batch_size'], shuffle=True, method='video', window_size=args['window_size'])



    triplet_loss = nn.TripletMarginLoss(margin=args['margin'])

    df = pd.DataFrame()

    # Batch iterator
    for epoch in tqdm.tqdm(range(args['epochs'])):
        print(f"Starting Epoch {epoch}")
        for idx, (video_batch, audio_batch, segments, flow_ranks) in enumerate(dataloader):
            try:

                audio_optimizer.zero_grad()
                video_optimizer.zero_grad()

                # create segments for each batch and compute embeddings for the segments
                # stack all the embeddings into single tensors
                batch_aud_embeddings, batch_vid_embeddings = utils.get_batch_embeddings(video_model, audio_model, video_batch, audio_batch, args['max_seq_len'], segments)

                
                # 1. Inter-modal loss
                inter_modal_loss = utils.get_intermodal_loss(batch_vid_embeddings, batch_aud_embeddings, k=args['top_k'], min_val=0)


                # 2. optical flow loss
                # this code finds the top and bottom ranked optical flow for a particular
                # video.  This is specified in flow ranks.  It then converts these indexes to 
                # their corresponding position in the stacked embeddings
                top_rank_idxs = []
                bottom_rank_idxs = []
                current_index = 0
                for ranks in flow_ranks:
                    top_rank_idxs.append(current_index + np.argmin(ranks))
                    bottom_rank_idxs.append(current_index + np.argmax(ranks))
                    current_index += len(ranks)

                # Randomly choose num_flow_matching pairs to match (top_rank, top_rank, bottom rank)
                top_matching_samples = list(itertools.product(top_rank_idxs, top_rank_idxs, bottom_rank_idxs))
                top_matching_samples = [random.choice(top_matching_samples) for _ in range(args['flow_k'])]

                # Randomly choose num_flow_matching pairs to match (bottom, bottom, top_rank rank)
                bottom_matching_samples = list(itertools.product(bottom_rank_idxs, bottom_rank_idxs, top_rank_idxs))
                bottom_matching_samples = [random.choice(bottom_matching_samples) for _ in range(args['flow_k'])]

                of_loss_top = 0
                for anchor, pos, neg in top_matching_samples:
                    of_loss_top += triplet_loss(batch_vid_embeddings[anchor], batch_vid_embeddings[pos], batch_vid_embeddings[neg])
                    of_loss_top += triplet_loss(batch_aud_embeddings[anchor], batch_aud_embeddings[pos], batch_aud_embeddings[neg])

                of_loss_bottom = 0
                for anchor, pos, neg in bottom_matching_samples:
                    of_loss_bottom += triplet_loss(batch_vid_embeddings[anchor], batch_vid_embeddings[pos], batch_vid_embeddings[neg])
                    of_loss_bottom += triplet_loss(batch_aud_embeddings[anchor], batch_aud_embeddings[pos], batch_aud_embeddings[neg])

                loss = args['lambda1']*inter_modal_loss + args['lambda2']*of_loss_top + args['lambda3']*of_loss_bottom
                loss.backward()
                audio_optimizer.step()
                video_optimizer.step()
                print(f'Epoch {epoch}, Batch {idx}, Train Loss {loss}')

            except Exception as e:
                # adding a wrapper just in case
                print(f"Epoch {epoch}, Batch {idx}, failed with Exception {e}")

        if epoch % 10 == 0:
            utils.save_checkpoint(audio_model, audio_optimizer, epoch, args['save_path'] + f'/audio_{epoch}.pth')
            utils.save_checkpoint(video_model, video_optimizer, epoch, args['save_path'] + f'/video_{epoch}.pth')
