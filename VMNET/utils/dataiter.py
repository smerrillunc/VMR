import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import pandas as pd
import numpy as np
import os
from os.path import join
import scipy.io


def FeatLoader(csv_path, video_dir, audio_dir, batch_size):
    x_feats_list = []
    y_feats_list = []
    labels_list = []

    with open(csv_path, 'r') as file:
        filenames = [line.strip() for line in file.readlines()]

    batchs = [filenames[i:i + batch_size] for i in range(0, len(filenames), batch_size)]

    if len(batchs[-1]) != batch_size:
        batchs = batchs[:-1]

    for i, batch in enumerate(batchs):
        a_f = []
        v_f = []
        label = np.eye(batch_size)

        for j, filename in enumerate(batch):
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

                a_f.append(tf.convert_to_tensor(a_mean))
                v_f.append(tf.convert_to_tensor(v_mean))

            except Exception as e:
                print(f"Failed to load {filename}: {e}")
                continue

        if len(a_f) != batch_size or len(v_f) != batch_size:
            print(f"Skipping incomplete batch {i+1} (size mismatch)")
            continue

        x_feats_list.append(tf.stack(a_f, axis=0))
        y_feats_list.append(tf.stack(v_f, axis=0))
        labels_list.append(tf.convert_to_tensor(label))

        print(f'Finished loading {i + 1} / {len(batchs)} batches')

    x_feats = tf.concat(x_feats_list, axis=0)
    y_feats = tf.concat(y_feats_list, axis=0)
    aff_xy = tf.stack(labels_list, axis=0)
    aff_xy = tf.reshape(aff_xy, [-1, batch_size])

    assert x_feats.shape[0] == y_feats.shape[0] == aff_xy.shape[0]
    return [x_feats, y_feats, aff_xy], batchs
        

def GetBatch(feats_list, num_epochs, batch_size, shuffle = False):
    input_queue = tf.train.slice_input_producer([feats_list[0], feats_list[1], feats_list[2]], num_epochs = num_epochs, shuffle = shuffle, capacity=batch_size )
    x_batch, y_batch, aff_xy = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=batch_size, 
                                                                                  allow_smaller_final_batch=False)
    return x_batch, y_batch, aff_xy