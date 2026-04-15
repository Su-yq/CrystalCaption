import runpy
import sys
import os
import multiprocessing
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
from rdkit import Chem
from crosscaption import experiment_text
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from crosscaption.experiment_text import ModelText
from crosscaption.Dataset_text import DatasetText
from crosscaption import layers
import numpy as np
import time
import random
from sklearn.metrics import balanced_accuracy_score


def verify_dir_exists(dirname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def make_dataset(data_abs_path, text_feature_dir=None):

    table_path = os.path.join(data_abs_path, 'CC_Table/CC_Table.tab')
    mol_blocks_path = os.path.join(data_abs_path, 'Mol_Blocks.dir')

    data1 = DatasetText(table_path, mol_blocks_dir=mol_blocks_path,
                        text_feature_dir=text_feature_dir)

    data1.make_graph_dataset(Desc=1, A_type='OnlyCovalentBond', hbond=0,
                             pipi_stack=0, contact=0, make_dataframe=True)

    return data1


def build_model_with_text(
        blcok_1_size,
        blcok_2_size,
        blcok_3_size,
        blcok_4_size,
        blcok_5_size,
        mp_act_func,
        n_head,
        pred_layer_1_size,
        pred_layer_2_size,
        pred_layer_3_size,
        pred_act_func,
        pred_dropout_rate,
        fusion_type='early',
        text_hidden_dim=256
):
    class Model(object):
        def build_model(self, inputs, is_training, global_step=None):
            V = inputs[0]
            A = inputs[1]
            labels = inputs[2]
            mask = inputs[3]
            graph_size = inputs[4]
            tags = inputs[5]
            global_state = inputs[6]
            subgraph_size = inputs[7]
            text_features = inputs[8]

            V, global_state = layers.CCGBlockText(
                V, A, global_state, subgraph_size, text_features=text_features,
                no_filters=blcok_1_size, act_func=mp_act_func,
                mask=mask, num_updates=global_step, is_training=is_training
            )

            if blcok_2_size is not None:
                V, global_state = layers.CCGBlockText(
                    V, A, global_state, subgraph_size, text_features=text_features,
                    no_filters=blcok_2_size, act_func=mp_act_func,
                    mask=mask, num_updates=global_step, is_training=is_training
                )

            if blcok_3_size is not None:
                V, global_state = layers.CCGBlockText(
                    V, A, global_state, subgraph_size, text_features=text_features,
                    no_filters=blcok_3_size, act_func=mp_act_func,
                    mask=mask, num_updates=global_step, is_training=is_training
                )

            if blcok_4_size is not None:
                V, global_state = layers.CCGBlockText(
                    V, A, global_state, subgraph_size, text_features=text_features,
                    no_filters=blcok_4_size, act_func=mp_act_func,
                    mask=mask, num_updates=global_step, is_training=is_training
                )

            if blcok_5_size is not None:
                V, global_state = layers.CCGBlockText(
                    V, A, global_state, subgraph_size, text_features=text_features,
                    no_filters=blcok_5_size, act_func=mp_act_func,
                    mask=mask, num_updates=global_step, is_training=is_training
                )

            V = layers.ReadoutFunction(V, global_state, graph_size,
                                       num_head=n_head, is_training=is_training)

            if fusion_type == 'early':
                fused = layers.EarlyFusionLayer(V, text_features,
                                                hidden_dim=text_hidden_dim)
            elif fusion_type == 'late':
                fused, _ = layers.LateFusionLayer(V, text_features,
                                                  hidden_dim=text_hidden_dim)
            elif fusion_type == 'concat':
                fused = tf.concat([V, text_features], axis=-1)
            else:
                fused = V

            with tf.compat.v1.variable_scope('Predictive_FC_1') as scope:
                V = layers.make_embedding_layer(fused, pred_layer_1_size)
                V = layers.make_bn(V, is_training, mask=None, num_updates=global_step)
                V = pred_act_func(V)
                V = tf.compat.v1.layers.dropout(V, pred_dropout_rate, training=is_training)

            if pred_layer_2_size is not None:
                with tf.compat.v1.variable_scope('Predictive_FC_2') as scope:
                    V = layers.make_embedding_layer(V, pred_layer_2_size)
                    V = layers.make_bn(V, is_training, mask=None, num_updates=global_step)
                    V = pred_act_func(V)
                    V = tf.compat.v1.layers.dropout(V, pred_dropout_rate, training=is_training)

            if pred_layer_3_size is not None:
                with tf.compat.v1.variable_scope('Predictive_FC_3') as scope:
                    V = layers.make_embedding_layer(V, pred_layer_3_size)
                    V = layers.make_bn(V, is_training, mask=None, num_updates=global_step)
                    V = pred_act_func(V)
                    V = tf.compat.v1.layers.dropout(V, pred_dropout_rate, training=is_training)

            out = layers.make_embedding_layer(V, 2, name='final')
            #fused = tf.concat([V, text_features], axis=-1)
            #return out, labels, fused
            return out, labels

    return Model()


def black_box_function(args_dict, root_abs_path, train_data, valid_data, text_feature_dir):
    print('\n' + str(args_dict))
    tf.compat.v1.reset_default_graph()

    batch_size = args_dict['batch_size']
    blcok_1_size = args_dict['blcok_1_size']
    blcok_2_size = args_dict['blcok_2_size']
    blcok_3_size = args_dict['blcok_3_size']
    blcok_4_size = args_dict['blcok_4_size']
    blcok_5_size = args_dict['blcok_5_size']
    mp_act_func = args_dict['mp_act_func']
    n_head = args_dict['n_head']
    pred_layer_1_size = args_dict['pred_layer_1_size']
    pred_layer_2_size = args_dict['pred_layer_2_size']
    pred_layer_3_size = args_dict['pred_layer_3_size']
    pred_act_func = args_dict['pred_act_func']
    pred_dropout_rate = args_dict['pred_dropout_rate']
    fusion_type = args_dict.get('fusion_type', 'early')

    snapshot_path = os.path.join(root_abs_path, 'bayes_snapshot/')
    model_name = 'ten_fold_cross_validation-CrossCaption/'
    verify_dir_exists(os.path.join(snapshot_path, model_name))

    if os.listdir(os.path.join(snapshot_path, model_name)) == []:
        dataset_name = 'Step_0/'
    else:
        l_ = [int(i.split('_')[1]) for i in os.listdir(os.path.join(snapshot_path, model_name)) if 'Step_' in i]
        dataset_name = 'Step_{}/'.format(max(l_) + 1)

    model = build_model_with_text(
        blcok_1_size, blcok_2_size, blcok_3_size, blcok_4_size, blcok_5_size,
        mp_act_func, n_head, pred_layer_1_size, pred_layer_2_size,
        pred_layer_3_size, pred_act_func, pred_dropout_rate,
        fusion_type=fusion_type
    )

    model = ModelText(model, train_data, valid_data, with_test=False,
                      snapshot_path=snapshot_path, use_subgraph=True,
                      use_text=True,
                      model_name=model_name, dataset_name=dataset_name + '/time_0')

    history = model.fit(num_epoch=100, save_info=True, save_att=False, silence=True,
                        train_batch_size=batch_size, max_to_keep=1, metric='loss')

    loss = min(history['valid_cross_entropy'])
    tf.compat.v1.reset_default_graph()
    print('\nLoss: {}'.format(loss))
    return loss

if __name__ == '__main__':
    multiprocessing.freeze_support()
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    root_abs_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_abs_path = os.path.join(root_abs_path, "data")
    text_feature_dir = os.path.join(root_abs_path, "data/processed_features")

    fold_10_path = os.path.join(data_abs_path, 'Fold_10.dir')
    if not os.path.exists(fold_10_path):
        raise FileNotFoundError(f"未找到Fold_10.dir文件: {fold_10_path}")
    fold_10 = eval(open(fold_10_path, 'r', encoding='utf-8').read())

    print("预加载数据集，加载文本特征...")
    temp_data = make_dataset(data_abs_path, text_feature_dir=text_feature_dir)
    valid_sample_names = list(temp_data.dataframe.keys())

    fold_samples = fold_10['fold-0']['train'] + fold_10['fold-0']['valid']
    Samples = [name for name in fold_samples if name in valid_sample_names]
    print(f"有效样本数: {len(Samples)}")

    random.shuffle(Samples)
    num_sample = len(Samples)
    train_num = int(0.9 * num_sample)
    train_samples = Samples[:train_num]
    valid_samples = Samples[train_num:]

    data = temp_data
    train_data, valid_data = data.split(train_samples=train_samples, valid_samples=valid_samples)

    if len(train_data) < 9:
        text_dim = 768
        train_data = list(train_data) + [np.zeros((len(train_samples), text_dim), dtype=np.float32)]
        valid_data = list(valid_data) + [np.zeros((len(valid_samples), text_dim), dtype=np.float32)]

    from hyperopt import fmin, tpe, Trials, hp

    args_dict = {
        'batch_size': hp.choice('batch_size', (128,)),
        'blcok_1_size': hp.choice('blcok_1_size', (16, 32, 64, 128, 256)),
        'blcok_2_size': hp.choice('blcok_2_size', (16, 32, 64, 128, 256, None)),
        'blcok_3_size': hp.choice('blcok_3_size', (16, 32, 64, 128, 256, None)),
        'blcok_4_size': hp.choice('blcok_4_size', (16, 32, 64, 128, 256, None)),
        'blcok_5_size': hp.choice('blcok_5_size', (16, 32, 64, 128, 256, None)),
        'mp_act_func': hp.choice('mp_act_func', (tf.nn.relu,)),
        'n_head': hp.choice('n_head', (1, 2, 3, 4, 5, 6, 7, 8)),
        'pred_layer_1_size': hp.choice('pred_layer_1_size', (64, 128, 256)),
        'pred_layer_2_size': hp.choice('pred_layer_2_size', (64, 128, 256, None)),
        'pred_layer_3_size': hp.choice('pred_layer_3_size', (64, 128, 256, None)),
        'pred_act_func': hp.choice('pred_act_func', (tf.nn.relu,)),
        'pred_dropout_rate': hp.uniform('pred_dropout_rate', 0.0, 0.5),
        'fusion_type': hp.choice('fusion_type', ('early', 'late', 'concat'))
    }

    trials = Trials()
    best = fmin(
        fn=lambda args: black_box_function(args, root_abs_path, train_data, valid_data, text_feature_dir),
        space=args_dict,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
        trials_save_file='trials_save_file-crosscaption'
    )

    print('\n最优参数:')
    print(best)
    best_params_path = os.path.join(root_abs_path, 'bayes_snapshot/ten_fold_cross_validation-CrossCaption/best_params.npy')
    verify_dir_exists(os.path.dirname(best_params_path))
    np.save(best_params_path, best)
    print(f'最优超参数已保存至: {best_params_path}')