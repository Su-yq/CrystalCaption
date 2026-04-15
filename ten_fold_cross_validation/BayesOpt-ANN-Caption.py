import sys
import os
import runpy
import multiprocessing

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import tensorflow as tf
from crosscaption.experiment_text import ModelText
from crosscaption.Dataset_text import DatasetText
from crosscaption import numpy as np
import random
import time

def verify_dir_exists(dirname):
    if not os.path.isdir(os.path.dirname(dirname)):
        os.makedirs(os.path.dirname(dirname))


def make_dataset(data_abs_path, text_feature_dir=None):
    data1 = DatasetText(
        os.path.join(data_abs_path, 'CC_Table/CC_Table.tab'),
        mol_blocks_dir=os.path.join(data_abs_path, 'Mol_Blocks.dir'),
        text_feature_dir=text_feature_dir
    )
    data1.make_graph_dataset(Desc=1, A_type='OnlyCovalentBond', hbond=0,
                             pipi_stack=0, contact=0, make_dataframe=True)
    return data1


def build_ann_with_text(
        layer_1_size,
        layer_2_size,
        layer_3_size,
        layer_4_size,
        layer_5_size,
        layer_6_size,
        layer_7_size,
        act_func,
        dropout,
        text_layer_size=256,
        fusion_type='concat'
):
    class DNN_Text(object):
        def build_model(self, inputs, is_training, global_step=None):
            if len(inputs) < 4:
                desc = inputs[0]
                labels = inputs[1]
                tags = inputs[2]
                text_dim = 768
                batch_size = tf.shape(desc)[0]
                text_features = tf.zeros([batch_size, text_dim])
            else:
                desc = inputs[0]
                labels = inputs[1]
                tags = inputs[2]
                text_features = inputs[3]
                if len(text_features.shape) == 1:
                    text_features = tf.expand_dims(text_features, axis=1)

            with tf.compat.v1.variable_scope('Text_Processing') as scope:
                text_proj = tf.compat.v1.layers.dense(text_features, text_layer_size)
                text_proj = tf.compat.v1.layers.batch_normalization(text_proj, training=is_training)
                text_proj = act_func(text_proj)
                text_proj = tf.compat.v1.layers.dropout(text_proj, dropout, training=is_training)

            if fusion_type == 'concat':
                fused = tf.concat([desc, text_proj], axis=1)
                layer_1_input_size = desc.get_shape()[-1].value + text_layer_size
            elif fusion_type == 'add':
                if desc.get_shape()[-1] != text_layer_size:
                    desc_proj = tf.compat.v1.layers.dense(desc, text_layer_size)
                else:
                    desc_proj = desc
                fused = desc_proj + text_proj
                layer_1_input_size = text_layer_size
            elif fusion_type == 'gate':
                if desc.get_shape()[-1] != text_layer_size:
                    desc_proj = tf.compat.v1.layers.dense(desc, text_layer_size)
                else:
                    desc_proj = desc
                gate_input = tf.concat([desc_proj, text_proj], axis=1)
                gate = tf.compat.v1.layers.dense(gate_input, text_layer_size)
                gate = tf.nn.sigmoid(gate)
                fused = gate * desc_proj + (1 - gate) * text_proj
                layer_1_input_size = text_layer_size
            else:
                fused = desc
                layer_1_input_size = desc.get_shape()[-1].value

            with tf.compat.v1.variable_scope('FC_1') as scope:
                fused = tf.compat.v1.layers.dense(fused, layer_1_size)
                fused = tf.compat.v1.layers.batch_normalization(fused, training=is_training)
                fused = act_func(fused)
                fused = tf.compat.v1.layers.dropout(fused, dropout, training=is_training)

            layers_config = [
                (layer_2_size, 'FC_2'),
                (layer_3_size, 'FC_3'),
                (layer_4_size, 'FC_4'),
                (layer_5_size, 'FC_5'),
                (layer_6_size, 'FC_6'),
                (layer_7_size, 'FC_7')
            ]

            for layer_size, scope_name in layers_config:
                if layer_size is not None:
                    with tf.compat.v1.variable_scope(scope_name) as scope:
                        fused = tf.compat.v1.layers.dense(fused, layer_size)
                        fused = tf.compat.v1.layers.batch_normalization(fused, training=is_training)
                        fused = act_func(fused)
                        fused = tf.compat.v1.layers.dropout(fused, dropout, training=is_training)

            fused = layers.make_fc_layer(fused, 2, is_training=is_training,
                                         with_bn=False, act_func=None, name='final')
            return fused, labels

    return DNN_Text()


def black_box_function(args_dict, root_abs_path, train_data, valid_data, text_feature_dir):
    tf.compat.v1.reset_default_graph()

    batch_size = args_dict['batch_size']
    layer_1_size = args_dict['layer_1_size']
    layer_2_size = args_dict['layer_2_size']
    layer_3_size = args_dict['layer_3_size']
    layer_4_size = args_dict['layer_4_size']
    layer_5_size = args_dict['layer_5_size']
    layer_6_size = args_dict['layer_6_size']
    layer_7_size = args_dict['layer_7_size']
    act_func = args_dict['act_fun']
    dropout = args_dict['dropout']
    fusion_type = args_dict.get('fusion_type', 'concat')

    if len(train_data) < 4:
        text_dim = 768
        train_data = list(train_data) + [np.zeros((train_data[0].shape[0], text_dim), dtype=np.float32)]
        valid_data = list(valid_data) + [np.zeros((valid_data[0].shape[0], text_dim), dtype=np.float32)]

    snapshot_path = os.path.join(root_abs_path, 'bayes_snapshot/')
    model_name = 'ten_fold_cross_validation-ANN-Text/'
    verify_dir_exists(os.path.join(snapshot_path, model_name))

    if os.listdir(os.path.join(snapshot_path, model_name)) == []:
        dataset_name = 'Step_0/'
    else:
        l_ = [int(i.split('_')[1]) for i in os.listdir(os.path.join(snapshot_path, model_name)) if 'Step_' in i]
        dataset_name = 'Step_{}/'.format(max(l_) + 1)

    model = build_ann_with_text(
        layer_1_size, layer_2_size, layer_3_size, layer_4_size,
        layer_5_size, layer_6_size, layer_7_size, act_func, dropout,
        fusion_type=fusion_type
    )

    model = ModelText(model, train_data, valid_data, snapshot_path=snapshot_path,
                      use_subgraph=False, use_desc=True, build_fc=True,
                      model_name=model_name, dataset_name=dataset_name + '/time_0',
                      use_text=True)

    history = model.fit(num_epoch=100, save_info=True, silence=0,
                        train_batch_size=batch_size, metric='loss')

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
        'batch_size': hp.choice('batch_size', (64, 128, 256)),
        'layer_1_size': hp.choice('layer_1_size', (16, 32, 64, 128, 256, 512)),
        'layer_2_size': hp.choice('layer_2_size', (16, 32, 64, 128, 256, 512, None)),
        'layer_3_size': hp.choice('layer_3_size', (16, 32, 64, 128, 256, 512, None)),
        'layer_4_size': hp.choice('layer_4_size', (16, 32, 64, 128, 256, 512, None)),
        'layer_5_size': hp.choice('layer_5_size', (16, 32, 64, 128, 256, 512, None)),
        'layer_6_size': hp.choice('layer_6_size', (16, 32, 64, 128, 256, 512, None)),
        'layer_7_size': hp.choice('layer_7_size', (16, 32, 64, 128, 256, 512, None)),
        'act_fun': hp.choice('act_fun', (tf.nn.relu, tf.nn.elu, tf.nn.tanh)),
        'dropout': hp.uniform('dropout', 0.0, 0.75),
        'fusion_type': hp.choice('fusion_type', ('concat', 'add', 'gate'))
    }

    trials = Trials()
    best = fmin(
        fn=lambda args: black_box_function(args, root_abs_path, train_data, valid_data, text_feature_dir),
        space=args_dict,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
        trials_save_file='trials_save_file-ann_text'
    )

    print('\n最优参数:')
    print(best)