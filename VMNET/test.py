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
flags.DEFINE_string('video_dir', "/work/users/s/m/smerrill/Youtube8m/video", 'Directory to contain vid features.')
flags.DEFINE_string('audio_dir', "/work/users/s/m/smerrill/Youtube8m/audio", 'Directory to contain audio features.')
flags.DEFINE_string('video_feature_dir', "/work/users/s/m/smerrill/Youtube8m/resnet/resnet101", 'Directory to contain vid features.')
flags.DEFINE_string('audio_feature_dir', "/work/users/s/m/smerrill/Youtube8m/vggish", 'Directory to contain audio features.')
flags.DEFINE_string('test_csv_path', "/proj/mcavoy_lab/Youtube8m/test.csv", 'Path to the csv recording all test samples')
flags.DEFINE_string('summaries_dir', "/proj/mcavoy_lab/Youtube8m/models/resnet", 'Directory to put the summary and log data.')


flags.DEFINE_integer('constraint_xy', 3, 'Constraint Weight xy')
flags.DEFINE_integer('constraint_yx',1, 'Constraint Weight yx')
flags.DEFINE_float('constraint_x', 0.2, 'Constraint Structure Weight x')
flags.DEFINE_float('constraint_y', 0.2, 'Constraint Structure Weight y')
flags.DEFINE_integer('vid_dim', 2048, 'Video Dim.')
flags.DEFINE_integer('aud_dim', 128, 'Audio Dim.')

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


vid_dict, aud_dict = create_feature_to_file_dicts(FLAGS.video_dir, FLAGS.video_feature_dir, FLAGS.audio_dir, FLAGS.audio_feature_dir)

x_batch, y_batch, aff_xy, audio_files, video_files = FeatLoaderTestset(FLAGS.test_csv_path, \
                                                                       FLAGS.video_feature_dir,\
                                                                       FLAGS.audio_feature_dir,\
                                                                       vid_dict,\
                                                                       aud_dict)


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

            fad_score = metrics.compute_fad(torch.stack(real_embeddings), torch.stack(retrieved_embeddings))
            print(f"FAD: {fad_score}")
            
            
            av_aligns = []
            got_aligns = []

            for idx, retrieval in enumerate(top1):
                # Av-align
                frames, fps = metrics.extract_frames(video_files[idx])
                _, video_peaks = metrics.detect_video_peaks(frames, fps)

                audio_peaks1 = metrics.detect_audio_peaks(audio_files[retrieval[0]])
                audio_peaks2 = metrics.detect_audio_peaks(audio_files[idx])
                av_align = metrics.calc_intersection_over_union(audio_peaks1, video_peaks, fps)
                got_av_align = metrics.calc_intersection_over_union(audio_peaks2, video_peaks, fps)

                print(f'idx {idx}, AV-ALIGN: {av_align}, GOT-AV-ALIGN: {got_av_align}')
                av_aligns.append(av_align)
                got_aligns.append(got_av_align)
            print(f"Overall AV-ALIGN {np.mean(av_aligns)}; Overall GOT-AV-ALIGN {np.mean(got_aligns)}")

            csv_path = os.path.join(checkpoint_dir, f"metrics_step_{step}.csv")

            with open(csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Write header
                writer.writerow(['Index', 'Top1_Index', 'AV_ALIGN', 'GOT_AV_ALIGN'])

                for idx, retrieval in enumerate(top1):
                    writer.writerow([idx, retrieval[0], av_aligns[idx], got_aligns[idx]])

                # Add summary statistics at the end
                writer.writerow([])
                writer.writerow(['Summary'])
                writer.writerow(['FAD', fad_score])
                writer.writerow(['Overall_XY_R@1', xy[0]])
                writer.writerow(['Overall_XY_R@5', xy[1]])
                writer.writerow(['Overall_XY_R@10', xy[2]])
                writer.writerow(['Overall_XY_R@20', xy[3]])
                writer.writerow(['Overall_XY_R@50', xy[4]])
                writer.writerow(['Overall_XY_R@100', xy[5]])
                writer.writerow(['Overall_YX_R@1', yx[0]])
                writer.writerow(['Overall_YX_R@5', yx[1]])
                writer.writerow(['Overall_YX_R@10', yx[2]])
                writer.writerow(['Overall_YX_R@20', yx[3]])
                writer.writerow(['Overall_YX_R@50', yx[4]])
                writer.writerow(['Overall_YX_R@100', yx[5]])
                writer.writerow(['Mean_AV_ALIGN', np.mean(av_aligns)])
                writer.writerow(['Mean_GOT_AV_ALIGN', np.mean(got_aligns)])


