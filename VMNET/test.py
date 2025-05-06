#!/usr/bin/python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import os
import scipy.io
from utils.dataiter import FeatLoaderTestset
from network_structure import Model_structure
import metrics
from utils.utils import create_feature_to_file_dicts
import csv
import torch
import gc
import wandb
import cv2

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_layer_x', 3, 'Constraint Weight xy')
flags.DEFINE_integer('num_layer_y',2, 'Constraint Weight yx')
flags.DEFINE_integer('test_batch_size', 1024, 'Test batch size.') #flags.DEFINE_integer('test_batch_size', 1000, 'Test batch size.')

# CHANGE YOUR OWN PATH CONFIGS BEFORE RUNNING!!!!!!
# local
#flags.DEFINE_string('test_data_dir', "/Users/scottmerrill/Documents/UNC/MultiModal/VMR/Youtube8m/", 'Directory to contain audio and rgb for test samples.')
#flags.DEFINE_string('test_csv_path', "/Users/scottmerrill/Documents/UNC/MultiModal/VMR/Youtube8m/test.csv", 'Path to the csv recording all test samples')
#flags.DEFINE_string('summaries_dir', "./models/MV_9k_efficient_b5_Avgpool_MUSICNN_penultimate_Structure_Nonlinear_single_loss_margin_0.5_emb_512_epochs_101_GlobalAvg", 'Directory to put the summary and log data.')

#longleaf
flags.DEFINE_string('video_dir', "/work/users/s/m/smerrill/SymMV/video", 'Directory to contain vid features.')
flags.DEFINE_string('audio_dir', "/work/users/s/m/smerrill/SymMV/audio", 'Directory to contain audio features.')
flags.DEFINE_string('video_feature_dir', "/work/users/s/m/smerrill/SymMV/resnet/resnet101", 'Directory to contain vid features.')
flags.DEFINE_string('audio_feature_dir', "/work/users/s/m/smerrill/SymMV/vggish", 'Directory to contain audio features.')
flags.DEFINE_string('test_csv_path', "/work/users/s/m/smerrill/SymMV/test_resnet.csv", 'Path to the csv recording all test samples')
flags.DEFINE_string('summaries_dir', "/proj/mcavoy_lab/Youtube8m/models/resnet", 'Directory to put the summary and log data.')


flags.DEFINE_integer('constraint_xy', 3, 'Constraint Weight xy')
flags.DEFINE_integer('constraint_yx',1, 'Constraint Weight yx')
flags.DEFINE_float('constraint_x', 0.2, 'Constraint Structure Weight x')
flags.DEFINE_float('constraint_y', 0.2, 'Constraint Structure Weight y')
flags.DEFINE_integer('vid_dim', 2048, 'Video Dim.')
flags.DEFINE_integer('aud_dim', 128, 'Audio Dim.')
flags.DEFINE_integer('pad_size', 0, 'PadSize')

net_opts = Model_structure.OPTS()
net_opts.network_name = 'Wrapping Network'
net_opts.x_dim = FLAGS.aud_dim
net_opts.y_dim = FLAGS.vid_dim  # CHANGE THE VIDEO FEAT DIM IF YOU USE DIFFERENT MODEL FOR VISUAL FEATURE EXTARCTION
net_opts.x_num_layer = FLAGS.num_layer_x
net_opts.y_num_layer = FLAGS.num_layer_y
net_opts.constraint_weights = [FLAGS.constraint_xy, FLAGS.constraint_yx, FLAGS.constraint_x, FLAGS.constraint_y]
net_opts.is_linear = False
net = Model_structure(net_opts)
net.construct()

features = FLAGS.video_feature_dir.split('/')[-1]
dataset = FLAGS.video_feature_dir.split('smerrill/')[1].split('/')[0]
run_name = f'{dataset}_{features}_{str(FLAGS.pad_size)}'
print(features, dataset)

wandb.init(
    project="VMNET",  # Change this to your preferred project name
    name=run_name,  # Set custom run name
    config={
        "padding": FLAGS.pad_size,
        "video_feature_dir": FLAGS.video_feature_dir,
        "test_csv_path": FLAGS.test_csv_path,
        "summaries_dir": FLAGS.summaries_dir})


print(FLAGS.video_feature_dir)
vid_dict, aud_dict = create_feature_to_file_dicts(FLAGS.video_dir, FLAGS.video_feature_dir, FLAGS.audio_dir, FLAGS.audio_feature_dir)


x_batch, y_batch, aff_xy, audio_files, video_files = FeatLoaderTestset(FLAGS.test_csv_path, \
                                                                       FLAGS.video_feature_dir,\
                                                                       FLAGS.audio_feature_dir,\
                                                                       vid_dict,\
                                                                       aud_dict,
                                                                       FLAGS.pad_size)


saver = tf.train.Saver(tf.global_variables())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    lamb = 0
    K = 100
    
    checkpoint_dir = os.path.join(FLAGS.summaries_dir, 'checkpoints')
    step = 0
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.all_model_checkpoint_paths:
        for path in ckpt.all_model_checkpoint_paths:
            aff_xy = sess.run(aff_xy)  
            x_batch = sess.run(x_batch)  
            y_batch = sess.run(y_batch)  
            saver.restore(sess, path)
            step = path.split('/')[-1].split('-')[-1]
            print('Session restored successfully. step: {0}'.format(step))
            xy, yx, xy_idx, yx_idx, xembed, yembed, top1 = sess.run([net.recall_xy, net.recall_yx, net.xy_idx, net.yx_idx, net.x_embed, net.y_embed, net.top1_yx_idx], 
                                                    feed_dict={
                                                        net.x_data: x_batch,
                                                        net.y_data: y_batch,
                                                        net.K: K,
                                                        net.aff_xy: aff_xy,
                                                        net.keep_prob: 1.,
                                                        net.is_training: False
                                                    })
            
            print("Overall xy R@1={}, R@5={}, R@10={}, R@20={}, R@50={}, R@100={}".format(xy[0], xy[1], xy[2], xy[3], xy[4], xy[5]))
            print("Overall yx R@1={}, R@5={}, R@10={}, R@20={}, R@50={}, R@100={}".format(yx[0], yx[1], yx[2], yx[3], yx[4], yx[5]))
            
            real_embeddings = []
            retrieved_embeddings = []
            for i in range(len(yembed)):
                real_embedding = xembed[i]

                retrieval_idx = top1[i][0]
                retrieved_embedding = xembed[retrieval_idx]

                real_embeddings.append(torch.tensor(real_embedding))
                retrieved_embeddings.append(torch.tensor(retrieved_embedding))

            real_embeddings = torch.stack(real_embeddings)
            retrieved_embeddings = torch.stack(retrieved_embeddings)
            fad_score = metrics.compute_fad(real_embeddings, retrieved_embeddings)
            KLD1 = metrics.compute_kld(real_embeddings, retrieved_embeddings)
            KLD2 = metrics.compute_kld(retrieved_embeddings, real_embeddings)

            print(f"FAD: {fad_score}")
            
            wandb.log({
                "step": int(step),
                "FAD": fad_score,
                "KLD1": KLD1,
                "KLD2": KLD2,
                "XY_R@1": xy[0],
                "XY_R@5": xy[1],
                "XY_R@10": xy[2],
                "XY_R@20": xy[3],
                "XY_R@50": xy[4],
                "XY_R@100": xy[5],
                "YX_R@1": yx[0],
                "YX_R@5": yx[1],
                "YX_R@10": yx[2],
                "YX_R@20": yx[3],
                "YX_R@50": yx[4],
                "YX_R@100": yx[5]
            })

            
            av_aligns = []
            got_aligns = []

            for idx, retrieval in enumerate(top1):
                try:
                    video_file = video_files[idx]
                    audio_file = audio_files[retrieval[0]]
                    got_audio_file = audio_files[idx]

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
                    av_aligns.append(av_align)
                    got_aligns.append(got_av_align)

                    # Log each individual AV-ALIGN to W&B
                    wandb.log({
                        "step": int(step),
                        "Sample_ID": idx,
                        "Top1_Index": retrieval[0],
                        "AV_ALIGN": av_align,
                        "GOT_AV_ALIGN": got_av_align
                    })

                    gc.collect()
                except Exception as e:
                    print(e)
                    
            print(f"Overall AV-ALIGN {np.mean(av_aligns)}; Overall GOT-AV-ALIGN {np.mean(got_aligns)}")

            wandb.log({
                "step": int(step),
                "Mean_AV_ALIGN": np.mean(av_aligns),
                "Mean_GOT_AV_ALIGN": np.mean(got_aligns),
            })