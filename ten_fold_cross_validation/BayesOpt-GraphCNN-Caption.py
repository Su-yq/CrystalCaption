import runpy
import sys
import os
import multiprocessing
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import random
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from crosscaption.experiment_text import ModelText
from crosscaption.Dataset_text import DatasetText
from crosscaption import layers


def verify_dir_exists(dirname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def make_dataset(data_abs_path, text_feature_dir=None):
    data1 = DatasetText(
        os.path.join(data_abs_path, 'CC_Table/CC_Table.tab'),
        mol_blocks_dir=os.path.join(data_abs_path, 'Mol_Blocks.dir'),
        text_feature_dir=text_feature_dir
    )
    data1.make_graph_dataset(Desc=0, A_type='OnlyCovalentBond', hbond=0,
                             pipi_stack=0, contact=0, make_dataframe=True)
    return data1


def build_graphcnn_with_text(
        graphcnn_layer_1_size,
        graphcnn_layer_2_size,
        graphcnn_layer_3_size,
        graphcnn_act_fun,
        graph_pool_1_size,
        graph_pool_2_size,
        graph_pool_3_size,
        graph_pool_act_fun,
        dense_layer_1_size,
        dense_layer_2_size,
        dense_layer_3_size,
        dense_act_func,
        dense_dropout,
        fusion_type='concat',
        text_hidden_dim=256
):
    mask_judge = (graph_pool_1_size, graph_pool_2_size, graph_pool_3_size)
    print(f"Pool层配置：{mask_judge}")

    class Model(object):
        def build_model(self, inputs, is_training, global_step):
            V = inputs[0]
            A = inputs[1]
            labels = inputs[2]
            mask = inputs[3]
            graph_size = inputs[4]
            tags = inputs[5]
            text_features = inputs[6]

            V = layers.make_graphcnn_layer(V, A, graphcnn_layer_1_size)
            V = layers.make_bn(V, is_training, mask=mask, num_updates=global_step)
            V = graphcnn_act_fun(V)

            if graph_pool_1_size is not None:
                V_pool, A = layers.make_graph_embed_pooling(V, A, mask=mask, no_vertices=graph_pool_1_size)
                V = layers.make_bn(V_pool, is_training, mask=None, num_updates=global_step)
                V = graph_pool_act_fun(V)

            if graphcnn_layer_2_size is not None:
                m = None if mask_judge[0] is not None else mask
                V = layers.make_graphcnn_layer(V, A, graphcnn_layer_2_size)
                V = layers.make_bn(V, is_training, mask=m, num_updates=global_step)
                V = graphcnn_act_fun(V)

            if graph_pool_2_size is not None:
                m = None if mask_judge[0] is not None else mask
                V_pool, A = layers.make_graph_embed_pooling(V, A, mask=m, no_vertices=graph_pool_2_size)
                V = layers.make_bn(V_pool, is_training, mask=None, num_updates=global_step)
                V = graph_pool_act_fun(V)

            if graphcnn_layer_3_size is not None:
                m = None if (mask_judge[1] is not None or mask_judge[0] is not None) else mask
                V = layers.make_graphcnn_layer(V, A, graphcnn_layer_3_size)
                V = layers.make_bn(V, is_training, mask=m, num_updates=global_step)
                V = graphcnn_act_fun(V)

            m = None if (mask_judge[1] is not None or mask_judge[0] is not None) else mask
            V_pool, A = layers.make_graph_embed_pooling(V, A, mask=m, no_vertices=graph_pool_3_size)
            V = layers.make_bn(V_pool, is_training, mask=None, num_updates=global_step)
            V = graph_pool_act_fun(V)

            no_input_features = int(np.prod(V.get_shape()[1:]))
            V = tf.reshape(V, [-1, no_input_features])

            with tf.compat.v1.variable_scope('Text_Processing') as scope:
                text_proj = tf.compat.v1.layers.dense(text_features, text_hidden_dim)
                text_proj = tf.compat.v1.layers.batch_normalization(text_proj, training=is_training)
                text_proj = dense_act_func(text_proj)
                text_proj = tf.compat.v1.layers.dropout(text_proj, dense_dropout, training=is_training)

            if fusion_type == 'concat':
                fused = tf.concat([V, text_proj], axis=1)
                dense_input_size = no_input_features + text_hidden_dim
            elif fusion_type == 'add':
                if no_input_features != text_hidden_dim:
                    V_proj = tf.compat.v1.layers.dense(V, text_hidden_dim)
                else:
                    V_proj = V
                fused = V_proj + text_proj
                dense_input_size = text_hidden_dim
            elif fusion_type == 'gate':
                if no_input_features != text_hidden_dim:
                    V_proj = tf.compat.v1.layers.dense(V, text_hidden_dim)
                else:
                    V_proj = V
                gate_input = tf.concat([V_proj, text_proj], axis=1)
                gate = tf.compat.v1.layers.dense(gate_input, text_hidden_dim)
                gate = tf.nn.sigmoid(gate)
                fused = gate * V_proj + (1 - gate) * text_proj
                dense_input_size = text_hidden_dim
            else:
                fused = V
                dense_input_size = no_input_features

            V = layers.make_embedding_layer(fused, dense_layer_1_size, name='FC-1')
            V = layers.make_bn(V, is_training, mask=None, num_updates=global_step)
            V = dense_act_func(V)
            V = tf.compat.v1.layers.dropout(V, dense_dropout, training=is_training)

            if dense_layer_2_size is not None:
                V = layers.make_embedding_layer(V, dense_layer_2_size, name='FC-2')
                V = layers.make_bn(V, is_training, mask=None, num_updates=global_step)
                V = dense_act_func(V)
                V = tf.compat.v1.layers.dropout(V, dense_dropout, training=is_training)

            if dense_layer_3_size is not None:
                V = layers.make_embedding_layer(V, dense_layer_3_size, name='FC-3')
                V = layers.make_bn(V, is_training, mask=None, num_updates=global_step)
                V = dense_act_func(V)
                V = tf.compat.v1.layers.dropout(V, dense_dropout, training=is_training)

            out = layers.make_embedding_layer(V, 2, name='final')
            return out, labels

    return Model()


def black_box_function(args_dict, root_abs_path, train_data, valid_data, text_feature_dir):
    tf.compat.v1.reset_default_graph()

    batch_size = args_dict['batch_size']
    graphcnn_layer_1_size = args_dict['graphcnn_layer_1_size']
    graphcnn_layer_2_size = args_dict['graphcnn_layer_2_size']
    graphcnn_layer_3_size = args_dict['graphcnn_layer_3_size']
    graphcnn_act_fun = args_dict['graphcnn_act_fun']
    graph_pool_1_size = args_dict['graph_pool_1_size']
    graph_pool_2_size = args_dict['graph_pool_2_size']
    graph_pool_3_size = args_dict['graph_pool_3_size']
    graph_pool_act_fun = args_dict['graph_pool_act_fun']
    dense_layer_1_size = args_dict['dense_layer_1_size']
    dense_layer_2_size = args_dict['dense_layer_2_size']
    dense_layer_3_size = args_dict['dense_layer_3_size']
    dense_act_func = args_dict['dense_act_func']
    dense_dropout = args_dict['dense_dropout']
    fusion_type = args_dict.get('fusion_type', 'concat')

    if len(train_data) < 7:
        text_dim = 768
        train_data = list(train_data) + [np.zeros((train_data[0].shape[0], text_dim), dtype=np.float32)]
        valid_data = list(valid_data) + [np.zeros((valid_data[0].shape[0], text_dim), dtype=np.float32)]

    snapshot_path = os.path.join(root_abs_path, 'bayes_snapshot/')
    model_name = 'ten_fold_cross_validation-GraphCNN-Text/'
    verify_dir_exists(os.path.join(snapshot_path, model_name))

    if os.listdir(os.path.join(snapshot_path, model_name)) == []:
        dataset_name = 'Step_0/'
    else:
        l_ = [int(i.split('_')[1]) for i in os.listdir(os.path.join(snapshot_path, model_name)) if 'Step_' in i]
        dataset_name = 'Step_{}/'.format(max(l_) + 1)

    model = build_graphcnn_with_text(
        graphcnn_layer_1_size, graphcnn_layer_2_size, graphcnn_layer_3_size,
        graphcnn_act_fun, graph_pool_1_size, graph_pool_2_size, graph_pool_3_size,
        graph_pool_act_fun, dense_layer_1_size, dense_layer_2_size, dense_layer_3_size,
        dense_act_func, dense_dropout, fusion_type=fusion_type
    )

    exp_model = ModelText(
        model, train_data, valid_data, with_test=False,
        snapshot_path=snapshot_path, use_subgraph=False, use_desc=False, build_fc=False,
        model_name=model_name, dataset_name=os.path.join(dataset_name, 'time_0'),
        use_text=True
    )

    history = exp_model.fit(
        num_epoch=100, save_info=True, save_att=False, silence=False,
        train_batch_size=batch_size, max_to_keep=1, metric='loss'
    )

    loss = min(history['valid_cross_entropy'])
    tf.compat.v1.reset_default_graph()
    print(f'\nLoss: {loss}')
    return loss


if __name__ == '__main__':
    multiprocessing.freeze_support()
    tf.compat.v1.disable_eager_execution()

    from hyperopt import fmin, tpe, Trials, hp

    root_abs_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_abs_path = os.path.join(root_abs_path, "data")
    text_feature_dir = os.path.join(root_abs_path, "data/processed_features")

    fold_10_path = os.path.join(data_abs_path, 'Fold_10.dir')
    fold_10 = eval(open(fold_10_path, 'r', encoding='utf-8').read())

    print("预加载数据集，加载文本特征...")
    temp_data = make_dataset(data_abs_path, text_feature_dir=text_feature_dir)
    valid_sample_names = list(temp_data.dataframe.keys())

    fold_samples = fold_10['fold-0']['train'] + fold_10['fold-0']['valid']
    Samples = [name for name in fold_samples if name in valid_sample_names]

    random.shuffle(Samples)
    num_sample = len(Samples)
    train_num = int(0.9 * num_sample)
    train_samples = Samples[:train_num]
    valid_samples = Samples[train_num:]

    data = temp_data
    train_data, valid_data = data.split(train_samples=train_samples, valid_samples=valid_samples)

    args_dict = {
        'batch_size': hp.choice('batch_size', (128,)),
        'graphcnn_layer_1_size': hp.choice('graphcnn_layer_1_size', (16, 32, 64, 128, 256)),
        'graphcnn_layer_2_size': hp.choice('graphcnn_layer_2_size', (16, 32, 64, 128, 256, None)),
        'graphcnn_layer_3_size': hp.choice('graphcnn_layer_3_size', (16, 32, 64, 128, 256, None)),
        'graphcnn_act_fun': hp.choice('graphcnn_act_fun', (tf.nn.relu,)),
        'graph_pool_1_size': hp.choice('graph_pool_1_size', (8, 16, 32, None)),
        'graph_pool_2_size': hp.choice('graph_pool_2_size', (8, 16, 32, None)),
        'graph_pool_3_size': hp.choice('graph_pool_3_size', (8, 16, 32)),
        'graph_pool_act_fun': hp.choice('graph_pool_act_fun', (tf.nn.relu,)),
        'dense_layer_1_size': hp.choice('dense_layer_1_size', (64, 128, 256, 512)),
        'dense_layer_2_size': hp.choice('dense_layer_2_size', (64, 128, 256, 512, None)),
        'dense_layer_3_size': hp.choice('dense_layer_3_size', (64, 128, 256, 512, None)),
        'dense_act_func': hp.choice('dense_act_func', (tf.nn.relu,)),
        'dense_dropout': hp.uniform('dense_dropout', 0.0, 0.75),
        'fusion_type': hp.choice('fusion_type', ('concat', 'add', 'gate'))
    }

    trials = Trials()
    best = fmin(
        fn=lambda args: black_box_function(args, root_abs_path, train_data, valid_data, text_feature_dir),
        space=args_dict,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
        trials_save_file='trials_save_file-graphcnn_text'
    )
    print('\n最优参数:')
    print(best)