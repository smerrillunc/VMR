import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import pandas as pd
import numpy as np
import os
from os.path import join
import scipy.io


def FeatLoader(csv_path, video_dir, audio_dir, batch_size):
    a_f=[]
    v_f=[]
    labels = []
    
    with open(csv_path, 'r') as file:
        filenames = [line.strip() for line in file.readlines()]

    batchs = [filenames[i:i + batch_size] for i in range(0, len(filenames), batch_size)]

    if len(batchs[-1]) != batch_size:
        batchs = batchs[:-1] # delete the last batch if it cannot fill a complete one
    else:
        pass
    
    for i, batch in enumerate(batchs):
        label = np.eye(batch_size)
        for filename in batch:
            audio_path = os.path.join(audio_dir, filename)
            video_path = os.path.join(video_dir, filename)
            a_f.append(tf.convert_to_tensor(np.mean(np.load(audio_path),axis=0)))
            v_f.append(tf.convert_to_tensor(np.mean(np.load(video_path),axis=0)))
        labels.append(tf.convert_to_tensor(label))
        print('Finished loading {} / {} batchs'.format(i + 1 , len(batchs)))
    x_feats = tf.stack(a_f, axis = 0)
    y_feats = tf.stack(v_f, axis = 0)
    aff_xy = tf.stack(labels, axis = 0)    
    aff_xy = tf.reshape(aff_xy, [-1, batch_size])

    assert x_feats.get_shape()[0] == y_feats.get_shape()[0] == aff_xy.get_shape()[0]
    return [x_feats, y_feats, aff_xy], batchs
    

def GetBatch(feats_list, num_epochs, batch_size, shuffle = False):
    input_queue = tf.train.slice_input_producer([feats_list[0], feats_list[1], feats_list[2]], num_epochs = num_epochs, shuffle = shuffle, capacity=batch_size )
    x_batch, y_batch, aff_xy = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1, capacity=batch_size, 
                                                                                  allow_smaller_final_batch=False)
    return x_batch, y_batch, aff_xy