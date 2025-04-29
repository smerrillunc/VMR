import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import pandas as pd
import numpy as np
import os
from os.path import join
import scipy.io
import random


def FeatLoader(csv_path, video_dir, audio_dir, batch_size):
    x_feats_list = []
    y_feats_list = []
    labels_list = []

    with open(csv_path, 'r') as file:
        all_filenames = [line.strip() for line in file.readlines()]

    # Shuffle for randomness
    random.shuffle(all_filenames)
    used_filenames = set()

    def load_features(filename):
        audio_path = os.path.join(audio_dir, filename)
        video_path = os.path.join(video_dir, filename)

        try:
            audio_feat = np.load(audio_path)
            video_feat = np.load(video_path)

            if audio_feat.size == 0 or video_feat.size == 0:
                print(f"Skipping {filename} due to empty features")
                return None

            a_mean = np.mean(audio_feat, axis=0)
            v_mean = np.mean(video_feat, axis=0)

            if np.ndim(a_mean) == 0 or np.ndim(v_mean) == 0:
                print(f"Skipping {filename} due to scalar feature")
                return None

            return tf.convert_to_tensor(a_mean), tf.convert_to_tensor(v_mean)

        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            return None

    i = 0
    while i < len(all_filenames):
        batch_filenames = []
        j = 0

        while j < batch_size and i < len(all_filenames):
            fname = all_filenames[i]
            if fname not in used_filenames:
                batch_filenames.append(fname)
                used_filenames.add(fname)
                j += 1
            i += 1

        a_f = []
        v_f = []
        label = np.eye(batch_size)

        retries = 0
        max_retries = 20

        for idx, fname in enumerate(batch_filenames):
            result = load_features(fname)
            if result:
                a_feat, v_feat = result
                a_f.append(a_feat)
                v_f.append(v_feat)
            else:
                # Try to sample a replacement
                while retries < max_retries:
                    replacement = random.choice(all_filenames)
                    if replacement not in used_filenames:
                        result = load_features(replacement)
                        if result:
                            used_filenames.add(replacement)
                            a_feat, v_feat = result
                            a_f.append(a_feat)
                            v_f.append(v_feat)
                            break
                        retries += 1

        if len(a_f) != batch_size or len(v_f) != batch_size:
            print(f"Skipping incomplete batch due to unrecoverable issues.")
            continue

        x_feats_list.append(tf.stack(a_f, axis=0))
        y_feats_list.append(tf.stack(v_f, axis=0))
        labels_list.append(tf.convert_to_tensor(label))

        print(f'Finished loading {len(x_feats_list)} batches')

    if not x_feats_list:
        raise ValueError("No valid batches loaded.")

    x_feats = tf.concat(x_feats_list, axis=0)
    y_feats = tf.concat(y_feats_list, axis=0)
    aff_xy = tf.stack(labels_list, axis=0)
    aff_xy = tf.reshape(aff_xy, [-1, batch_size])

    assert x_feats.shape[0] == y_feats.shape[0] == aff_xy.shape[0]
    return [x_feats, y_feats, aff_xy], list(used_filenames)
        

def GetBatch(feats_list, num_epochs, batch_size, shuffle = False):
    input_queue = tf.train.slice_input_producer([feats_list[0], feats_list[1], feats_list[2]], num_epochs = num_epochs, shuffle = shuffle, capacity=batch_size )
    x_batch, y_batch, aff_xy = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=batch_size, 
                                                                                  allow_smaller_final_batch=False)
    return x_batch, y_batch, aff_xy


def FeatLoaderTestset(csv_path, video_dir, audio_dir, vid_dict, aud_dict):
    x_feats_list = []
    y_feats_list = []
    x_file_list = []
    y_file_list = []
    labels_list = []

    with open(csv_path, 'r') as file:
        filenames = [line.strip() for line in file.readlines()]

    for j, filename in enumerate(filenames):
        audio_path = os.path.join(audio_dir, filename)
        video_path = os.path.join(video_dir, filename)

        try:
            audio_feat = np.load(audio_path)
            video_feat = np.load(video_path)

            if audio_feat.size == 0 or video_feat.size == 0:
                print(f"Skipping {filename} due to empty features")
                continue

            a_mean = np.mean(audio_feat, axis=0)
            v_mean = np.mean(video_feat, axis=0)

            # Check if mean resulted in scalar
            if np.ndim(a_mean) == 0 or np.ndim(v_mean) == 0:
                print(f"Skipping {filename} due to scalar feature")
                continue
                
            y_feats_list.append(tf.convert_to_tensor(v_mean))
            x_feats_list.append(tf.convert_to_tensor(a_mean))

            y_file_list.append(vid_dict[video_path])
            x_file_list.append(aud_dict[audio_path])
            
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            continue

    labels_list  = tf.convert_to_tensor(np.eye(len(x_feats_list)))
    y_feats = tf.convert_to_tensor(y_feats_list)
    x_feats = tf.convert_to_tensor(x_feats_list)    
    
    print(x_feats.shape[0], y_feats.shape[0], labels_list.shape[0])
    assert x_feats.shape[0] == y_feats.shape[0] == labels_list.shape[0]
    return [x_feats, y_feats, labels_list, x_file_list, y_file_list]