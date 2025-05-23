# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains a collection of util functions for training and evaluating.
"""

import numpy
import tensorflow.compat.v1 as tf
import os
tf.disable_v2_behavior() 

def Dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value
    return feat_vector * scalar + bias


def MakeSummary(name, value):
    """Creates a tf.Summary proto with the given name and value."""
    summary = tf.Summary()
    val = summary.value.add()
    val.tag = str(name)
    val.simple_value = float(value)
    return summary


def AddGlobalStepSummary(summary_writer,
                         global_step_val,
                         global_step_info_dict,
                         summary_scope="Eval"):
    """Add the global_step summary to the Tensorboard.
    Args:
      summary_writer: Tensorflow summary_writer.
      global_step_val: a int value of the global step.
      global_step_info_dict: a dictionary of the evaluation metrics calculated for
        a mini-batch.
      summary_scope: Train or Eval.
    Returns:
      A string of this global_step summary
    """
    if "hit_at_one_emb" in global_step_info_dict.keys():
        this_hit_at_one_emb = global_step_info_dict["hit_at_one_emb"]
        summary_writer.add_summary(
            MakeSummary("GlobalStep/" + summary_scope + "_Hit@1Embedding", this_hit_at_one_emb),
            global_step_val)
    this_hit_at_one = global_step_info_dict["hit_at_one"]
    this_perr = global_step_info_dict["perr"]
    this_loss = global_step_info_dict["loss"]
    examples_per_second = global_step_info_dict.get("examples_per_second", -1)

    summary_writer.add_summary(
      MakeSummary("GlobalStep/" + summary_scope + "_Hit@1", this_hit_at_one),
      global_step_val)
    summary_writer.add_summary(
      MakeSummary("GlobalStep/" + summary_scope + "_Perr", this_perr),
      global_step_val)
    summary_writer.add_summary(
      MakeSummary("GlobalStep/" + summary_scope + "_Loss", this_loss),
      global_step_val)

    if examples_per_second != -1:
        summary_writer.add_summary(
            MakeSummary("GlobalStep/" + summary_scope + "_Example_Second",
                        examples_per_second), global_step_val)

    summary_writer.flush()
    if "hit_at_one_emb" in global_step_info_dict.keys():
        info = ("global_step {0} | Batch Hit@1: {1:.3f} | Batch PERR: {2:.3f} | Batch Loss: {3:.3f} "
              "| Examples_per_sec: {4:.3f} | Batch Hit@1Embeddingg: {5:.0f}").format(
                  global_step_val, this_hit_at_one, this_perr, this_loss,
                  examples_per_second, this_hit_at_one_emb)
    else:
        info = ("global_step {0} | Batch Hit@1: {1:.3f} | Batch PERR: {2:.3f} | Batch Loss: {3:.3f} "
              "| Examples_per_sec: {4:.3f}").format(
                  global_step_val, this_hit_at_one, this_perr, this_loss,
                  examples_per_second)
    return info


def AddEpochSummary(summary_writer,
                    global_step_val,
                    epoch_info_dict,
                    summary_scope="Eval"):
    epoch_id = epoch_info_dict["epoch_id"]
    avg_hit_at_one = epoch_info_dict["avg_hit_at_one"]
    avg_perr = epoch_info_dict["avg_perr"]
    avg_loss = epoch_info_dict["avg_loss"]
    aps = epoch_info_dict["aps"]
    gap = epoch_info_dict["gap"]
    mean_ap = numpy.mean(aps)

    if "avg_hit_at_one_emb" in epoch_info_dict.keys():
        avg_hit_at_one_emb = epoch_info_dict["avg_hit_at_one_emb"]
        summary_writer.add_summary(
            MakeSummary("Epoch/" + summary_scope + "_Avg_Hit@1Embedding", avg_hit_at_one_emb),
            global_step_val)

    summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_Avg_Hit@1", avg_hit_at_one),
      global_step_val)
    summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_Avg_Perr", avg_perr),
      global_step_val)
    summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_Avg_Loss", avg_loss),
      global_step_val)
    summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_MAP", mean_ap),
          global_step_val)
    summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_GAP", gap),
          global_step_val)
    summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_GAP", gap),
          global_step_val)
    summary_writer.flush()

    if "avg_hit_at_one_emb" in epoch_info_dict.keys():
        info = ("epoch/eval number {0} | Avg_Hit@1: {1:.3f} | Avg_PERR: {2:.3f} "
              "| MAP: {3:.3f} | GAP: {4:.3f} | Avg_Loss: {5:3f} | Avg_Hit@1Emb (%): {6:.5f}").format(
              epoch_id, avg_hit_at_one, avg_perr, mean_ap, gap, avg_loss, avg_hit_at_one_emb*100)
    else:
        info = ("epoch/eval number {0} | Avg_Hit@1: {1:.3f} | Avg_PERR: {2:.3f} "
              "| MAP: {3:.3f} | GAP: {4:.3f} | Avg_Loss: {5:3f}").format(
              epoch_id, avg_hit_at_one, avg_perr, mean_ap, gap, avg_loss)
    return info

def GetListOfFeatureNamesAndSizes(feature_names, feature_sizes):

    list_of_feature_names = [
      feature_names.strip() for feature_names in feature_names.split(',')]
    list_of_feature_sizes = [int(feature_sizes) for feature_sizes in feature_sizes.split(',')]

    return list_of_feature_names, list_of_feature_sizes


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

