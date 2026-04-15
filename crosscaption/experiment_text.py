import tensorflow as tf
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import time
import os

try:
    from .experiment import *
except ImportError:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from crosscaption.experiment import *


def create_graph_placeholders_with_text(dataset, use_desc=True, with_tags=True,
                                        with_attention=True, use_subgraph=False):
    placeholders = []
    V_shape = [None] + list(dataset[0].shape[1:])
    V = tf.compat.v1.placeholder(tf.as_dtype(dataset[0].dtype), shape=V_shape, name='V_input')
    placeholders.append(V)

    A_shape = [None] + list(dataset[1].shape[1:])
    A = tf.compat.v1.placeholder(tf.as_dtype(dataset[1].dtype), shape=A_shape, name='AdjMat_input')
    placeholders.append(A)

    labels_shape = [None]
    labels = tf.compat.v1.placeholder(tf.as_dtype(dataset[2].dtype), shape=labels_shape, name='labels_input')
    placeholders.append(labels)

    mask_shape = [None] + list(dataset[3].shape[1:])
    masks = tf.compat.v1.placeholder(tf.as_dtype(dataset[3].dtype), shape=mask_shape, name='masks_input')
    placeholders.append(masks)

    if with_attention:
        graph_size_shape = [None]
        graph_size = tf.compat.v1.placeholder(tf.as_dtype(dataset[4].dtype), shape=graph_size_shape,
                                              name='graph_size_input')
        placeholders.append(graph_size)

    if with_tags:
        tags_shape = [None]
        tags = tf.compat.v1.placeholder(tf.as_dtype(dataset[5].dtype), shape=tags_shape, name='tags_input')
        placeholders.append(tags)

    if use_desc:
        global_state_shape = [None] + list(dataset[6].shape[1:])
        global_state = tf.compat.v1.placeholder(tf.as_dtype(dataset[6].dtype), shape=global_state_shape,
                                                name='global_state_input')
        placeholders.append(global_state)

    if use_subgraph:
        subgraph_size_shape = [None, 2]
        subgraph_size = tf.compat.v1.placeholder(tf.as_dtype(dataset[7].dtype), shape=subgraph_size_shape,
                                                 name='subgraph_size_input')
        placeholders.append(subgraph_size)

    text_idx = 8 if use_subgraph else (7 if use_desc else 6)
    if len(dataset) > text_idx:
        text_shape = [None] + list(dataset[text_idx].shape[1:])
        text_features = tf.compat.v1.placeholder(tf.as_dtype(dataset[text_idx].dtype),
                                                 shape=text_shape, name='text_features_input')
        placeholders.append(text_features)

    return placeholders


class ModelText(Model):
    def __init__(self, model, train_data, valid_data, with_test=False,
                 test_data=None, build_fc=False, model_name='model',
                 dataset_name='dataset', with_tags=True, use_desc=True,
                 use_subgraph=False, with_attention=True, use_text=True,
                 snapshot_path='./snapshot/', summary_path='./summary/',
                 return_fused=False):

        tf.compat.v1.reset_default_graph()
        self.train_data = train_data
        self.test_data = valid_data
        self.val = with_test
        self.use_text = use_text
        self.return_fused = return_fused

        if self.val:
            self.val_data = test_data

        self.is_training = tf.compat.v1.placeholder(tf.bool, shape=(), name='is_training')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.build_fc = build_fc

        if build_fc:
            self.inputs = create_fc_placeholders(train_data)
        else:
            self.inputs = create_graph_placeholders_with_text(
                train_data, use_desc=use_desc, with_tags=with_tags,
                with_attention=with_attention, use_subgraph=use_subgraph
            )

        self.pred_out, self.labels = model.build_model(self.inputs, self.is_training, self.global_step)

        self.snapshot_path = snapshot_path + '/%s/%s/' % (model_name, dataset_name)
        self.test_summary_path = summary_path + '/%s/test/%s' % (model_name, dataset_name)
        self.train_summary_path = summary_path + '/%s/train/%s' % (model_name, dataset_name)
        self.is_finetuning = False

        if return_fused:
            self.pred_out, self.labels, self.fused_op = model.build_model(
                self.inputs, self.is_training, self.global_step, return_fused=True
            )
        else:
            self.pred_out, self.labels = model.build_model(
                self.inputs, self.is_training, self.global_step
            )