#!/usr/bin/python

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np

from flip_gradient import *
# import embed_network as en
import embed_network_music as enm
import embed_network_video as env
import embed_loss as el
import embed_structure_loss as esl
import eval
import tensorflow_probability as tfp
import OPTS


class Model_structure:
    class OPTS(OPTS.OPTS):
        def __init__(self):
            OPTS.OPTS.__init__(self, 'Model OPTS')
            self.network_name = None
            self.x_dim = None
            self.y_dim = None
            self.x_num_layer = 2
            self.y_num_layer = 2
            self.constraint_weights = [2, 1]
            self.batch_size = 1024
            self.is_linear = False

    def __init__(self, opts):
        if opts is None:
            opts = self.OPTS()
        self.opts = opts
        self.opts.assert_all_keys_valid()

    def construct(self):

        self.x_data = tf.placeholder(tf.float32, [None, self.opts.x_dim], name='X_data')
        self.y_data = tf.placeholder(tf.float32, [None, self.opts.y_dim], name='Y_data')

        # Build embedding
        self.aff_xy = tf.placeholder(tf.bool, [None, None], name='aff_xy')
        self.K = tf.placeholder(tf.int32, name='K')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.l = tf.placeholder(tf.float32, name='lambda_for_adv')

        x_net_opts = enm.Music_Model.OPTS()
        x_net_opts.network_name = 'X_network'
        x_net_opts.num_layer = self.opts.x_num_layer
        self.x_net = enm.Music_Model(x_net_opts)

        y_net_opts = env.Video_Model.OPTS()
        y_net_opts.network_name = 'Y_network'
        y_net_opts.num_layer = self.opts.y_num_layer
        self.y_net = env.Video_Model(y_net_opts)

        self.x_embed = self.x_net.construct(self.x_data, keep_prob=self.keep_prob, is_linear=self.opts.is_linear, is_training=self.is_training)
        self.y_embed = self.y_net.construct(self.y_data, keep_prob=self.keep_prob, is_linear=self.opts.is_linear, is_training=self.is_training)



        # Cross-modal Triplet loss
        el_opts = el.Triplet.OPTS()
        el_opts.network_name = 'Triplet'
        el_opts.distance = el_opts.DIST.COS

        el_net = el.Triplet(el_opts)
        print(self.x_embed.shape, self.y_embed.shape)
        self.loss_cross_xy, self.loss_cross_yx = el_net.construct(self.x_embed, self.y_embed, self.aff_xy, self.K, 'Triplet_Net')

        # Single-modal structure loss
        esl_opts = esl.Triplet_Structure.OPTS()
        esl_opts.network_name = 'Triplet_Structure'
        esl_opts.distance = el_opts.DIST.COS

        esl_net = esl.Triplet_Structure(esl_opts)
        self.loss_single_x = esl_net.construct(self.x_data, self.x_embed, self.K, 'Triplet_Strcture_x_Net')
        self.loss_single_y = esl_net.construct(self.y_data, self.y_embed, self.K, 'Triplet_Strcture_y_Net')

        # Adversarial loss
        print(self.x_embed)
        print(self.y_embed)
        self.concat_feat = tf.concat([self.x_embed, self.y_embed], 0)
        self.dom_label = tf.concat([tf.tile([0.], [tf.shape(self.x_embed)[0]]),
                                    tf.tile([1.], [tf.shape(self.y_embed)[0]])], 0)
        self.dom_label = tf.expand_dims(self.dom_label, 1)
        self.dom_loss, self.dom_acc = self.dom_classifier(self.concat_feat, self.dom_label, self.l, self.keep_prob)


        # Final loss
        self.loss = self.loss_cross_xy * self.opts.constraint_weights[0] + self.loss_cross_yx * self.opts.constraint_weights[1] \
                        + self.loss_single_x * self.opts.constraint_weights[2] + self.loss_single_y * self.opts.constraint_weights[3]

        # calculate gradients
        _, self.xy_rank_x = tf.nn.moments(tf.gradients(self.loss_cross_xy, [self.x_embed])[0], [0, 1])
        _, self.yx_rank_x = tf.nn.moments(tf.gradients(self.loss_cross_yx, [self.x_embed])[0], [0, 1])
        _, self.x_rank_x = tf.nn.moments(tf.gradients(self.loss_single_x, [self.x_embed])[0], [0, 1])
        _, self.xy_rank_y = tf.nn.moments(tf.gradients(self.loss_cross_xy, [self.y_embed])[0], [0, 1])
        _, self.yx_rank_y = tf.nn.moments(tf.gradients(self.loss_cross_yx, [self.y_embed])[0], [0, 1])
        _, self.y_rank_y = tf.nn.moments(tf.gradients(self.loss_single_y, [self.y_embed])[0], [0, 1])


        eval_opts = eval.Recall.OPTS()
        eval_opts.network_name = 'Recall'

        eval_net = eval.Recall(eval_opts)

        self.recall_xy, self.recall_yx, self.xy_idx, self.yx_idx, self.top1_xy_idx, self.top1_yx_idx = eval_net.construct(self.x_embed, self.y_embed, self.aff_xy, [1, 5, 10, 20, 50, 100])

        # Calculate FAD
        self.fad = self.compute_fad()


        # calculate av-align score
        self.av_align_score = self.compute_av_align_score()

        # self.debug_list = el_net.debug_list

    def dom_classifier(self, tensor, labels, l=1., keep_prob=1.):
        with tf.name_scope("dom_classifier"):
            feature = flip_gradient(tensor, l)
            
            d_fc1 = self.fc_layer(feature, 512, "dom_fc1")
            d_bn1 = tf.layers.batch_normalization(inputs=d_fc1, axis = -1, center=True, scale=True, epsilon=1e-3, \
                                                    training=self.is_training, name='dom_bn1')
            d_relu1 = tf.nn.relu(d_bn1, name='d_relu1')
            
            d_relu1 = tf.nn.dropout(d_relu1, keep_prob)
            
            d_fc2 = self.fc_layer(d_relu1, 512, "dom_fc2")
            d_bn2 = tf.layers.batch_normalization(inputs=d_fc2, axis = -1, center=True, scale=True, epsilon=1e-3, \
                                                    training=self.is_training, name='dom_bn2')
            d_relu2 = tf.nn.relu(d_bn2, name='d_relu2')
            d_relu2 = tf.nn.dropout(d_relu2, keep_prob)

            # d_logits = self.fc_layer(d_fc2, 2, "dom_logits")
            d_logit = self.fc_layer(d_relu2, 1, "dom_logit")
            d_logit = tf.nn.relu(d_logit, name='d_relu_logit')
            

        with tf.name_scope("domain_acc_and_loss"):
            # self.domain_prediction = tf.nn.softmax(d_logits)
            domain_prediction = tf.sigmoid(d_logit)

            # self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(d_logits, self.domain)
            domain_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logit, labels = labels)
            domain_loss = tf.reduce_mean(domain_loss)

            # domain acc
            correct_domain_prediction = tf.equal(tf.round(domain_prediction), labels)
            domain_acc = tf.reduce_mean(tf.cast(correct_domain_prediction, tf.float32))
        return domain_loss, domain_acc

    def fc_layer(self, tensor, dim, name, reuse=False):
        in_shape = tensor.get_shape()

        with tf.variable_scope(name, reuse=reuse):
            weights = tf.get_variable("weights", shape=[in_shape[1], dim],
                                      initializer=tf.initializers.glorot_normal())
            biases = tf.get_variable("biases", shape=[dim],
                                     initializer=tf.initializers.constant(0.0))
            fc = tf.nn.xw_plus_b(tensor, weights, biases)
            return fc

    def compute_fad_alt(self):
        # Get the top 1 retrieved audio embeddings
        top1_retrieved_audio = tf.gather(self.x_embed, self.top1_yx_idx[:, 0])
        
        # Calculate mean and covariance with added epsilon for numerical stability
        epsilon = 1e-6
        mu_retrieved = tf.reduce_mean(top1_retrieved_audio, axis=0)
        sigma_retrieved = tfp.stats.covariance(top1_retrieved_audio) + epsilon * tf.eye(tf.shape(self.x_embed)[1])
        
        mu_ground_truth = tf.reduce_mean(self.y_embed, axis=0)
        sigma_ground_truth = tfp.stats.covariance(self.y_embed) + epsilon * tf.eye(tf.shape(self.y_embed)[1])
        
        # Calculate Fréchet distance with safeguards
        diff = mu_retrieved - mu_ground_truth
        covmean_sq = tf.matmul(tf.matmul(sigma_retrieved, sigma_ground_truth), sigma_retrieved)
        covmean = tf.linalg.sqrtm(covmean_sq + epsilon * tf.eye(tf.shape(covmean_sq)[0]))
        
        tr_covmean = tf.linalg.trace(covmean)
        
        fad = tf.reduce_sum(tf.square(diff)) + tf.linalg.trace(sigma_retrieved) + tf.linalg.trace(sigma_ground_truth) - 2 * tr_covmean
        
        return tf.maximum(fad, 0.0)  # Ensure non-negative


    def compute_fad(self):
        top1_retrieved_audio = tf.gather(self.x_embed, self.top1_yx_idx[:, 0])
        
        mu1, var1 = tf.nn.moments(top1_retrieved_audio, axes=[0])
        mu2, var2 = tf.nn.moments(self.y_embed, axes=[0])
        
        diff = mu1 - mu2
        covmean = tf.sqrt(tf.multiply(var1, var2))
        
        fad = tf.reduce_sum(tf.square(diff)) + tf.reduce_sum(var1) + tf.reduce_sum(var2) - 2 * tf.reduce_sum(covmean)
        return tf.maximum(fad, 0.0)

    def compute_av_align_score(self):
        # Normalize embeddings
        x_embed_norm = tf.nn.l2_normalize(self.x_embed, axis=1)
        y_embed_norm = tf.nn.l2_normalize(self.y_embed, axis=1)
        
        # Get top-1 retrieved audio embeddings for each video query
        _, top1_indices = tf.nn.top_k(tf.matmul(y_embed_norm, x_embed_norm, transpose_b=True), k=1)
        top1_retrieved = tf.gather(x_embed_norm, tf.squeeze(top1_indices))
        
        # Compute cosine similarity between top-1 retrieved and ground truth
        cosine_similarities = tf.reduce_sum(top1_retrieved * y_embed_norm, axis=1)
        
        # Compute average similarity (AV-align score)
        self.av_align_score = tf.reduce_mean(cosine_similarities)
        
        return self.av_align_score



