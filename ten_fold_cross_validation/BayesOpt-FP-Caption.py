import pickle
import runpy
import sys
import os
import multiprocessing

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import tensorflow as tf
from crosscaption.experiment_text import ModelText
from crosscaption.Dataset_text import DatasetText
from crosscaption import layers
import numpy as np
import time
import random


def verify_dir_exists(dirname):
    dir_path = os.path.dirname(dirname)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def make_dataset(fp_size, radii, data_abs_path, text_feature_dir=None):
    table_path = os.path.join(data_abs_path, 'CC_Table/CC_Table.tab')
    mol_blocks_path = os.path.join(data_abs_path, 'Mol_Blocks.dir')

    data1 = DatasetText(table_path, mol_blocks_dir=mol_blocks_path,
                        text_feature_dir=text_feature_dir)

    data1.make_embedding_dataset(fp_type='ecfp', nBits=fp_size, radii=radii,
                                 processes=4, make_dataframe=True)
    return data1


def build_model_with_text(
        layer_1_size,
        layer_2_size,
        layer_3_size,
        act_func,
        dropout,
        merge,
        forward_layer_1_size,
        forward_layer_2_size,
        forward_layer_3_size,
        forward_act_func,
        forward_dropout,
        text_layer_size=256,
        fusion_type='concat'
):
    class DNN_5_Text(object):
        def build_model(self, inputs, is_training, global_step=None):
            if len(inputs) < 4:
                fps = inputs[0]
                labels = inputs[1]
                tags = inputs[2]
                text_dim = 768
                batch_size = tf.shape(fps)[0]
                text_features = tf.zeros([batch_size, text_dim])
            else:
                fps = inputs[0]
                labels = inputs[1]
                tags = inputs[2]
                text_features = inputs[3]
                if len(text_features.shape) == 1:
                    text_features = tf.expand_dims(text_features, axis=1)

            fps = tf.reshape(fps, [-1, int(fps.get_shape()[-1].value / 2)])

            with tf.compat.v1.variable_scope('FC_1') as scope:
                fps = tf.compat.v1.layers.dense(fps, layer_1_size)
                fps = tf.compat.v1.layers.batch_normalization(fps, training=is_training)
                fps = act_func(fps)
                fps = tf.compat.v1.layers.dropout(fps, dropout, training=is_training)

            if layer_2_size != None:
                with tf.compat.v1.variable_scope('FC_2') as scope:
                    fps = tf.compat.v1.layers.dense(fps, layer_2_size)
                    fps = tf.compat.v1.layers.batch_normalization(fps, training=is_training)
                    fps = act_func(fps)
                    fps = tf.compat.v1.layers.dropout(fps, dropout, training=is_training)

            if layer_3_size != None:
                with tf.compat.v1.variable_scope('FC_3') as scope:
                    fps = tf.compat.v1.layers.dense(fps, layer_3_size)
                    fps = tf.compat.v1.layers.batch_normalization(fps, training=is_training)
                    fps = act_func(fps)
                    fps = tf.compat.v1.layers.dropout(fps, dropout, training=is_training)

            if merge == 'add':
                with tf.compat.v1.variable_scope('merge_add') as scope:
                    fp_size = fps.get_shape()[-1].value
                    fps = tf.reshape(fps, [-1, 2, fp_size])
                    fps = tf.reduce_sum(fps, axis=1)
            elif merge == 'concat':
                with tf.compat.v1.variable_scope('merge_concat') as scope:
                    fp_size = fps.get_shape()[-1].value
                    fps = tf.reshape(fps, [-1, fp_size * 2])

            if len(text_features.shape) == 1:
                text_features = tf.expand_dims(text_features, axis=0)

            with tf.compat.v1.variable_scope('Text_Processing') as scope:
                text_proj = tf.compat.v1.layers.dense(text_features, text_layer_size)
                text_proj = tf.compat.v1.layers.batch_normalization(text_proj, training=is_training)
                text_proj = act_func(text_proj)
                text_proj = tf.compat.v1.layers.dropout(text_proj, dropout, training=is_training)

            if fusion_type == 'concat':
                fused = tf.concat([fps, text_proj], axis=1)
            elif fusion_type == 'add':
                if fps.get_shape()[-1] != text_proj.get_shape()[-1]:
                    fps = tf.compat.v1.layers.dense(fps, text_layer_size)
                fused = fps + text_proj
            elif fusion_type == 'gate':
                if fps.get_shape()[-1] != text_proj.get_shape()[-1]:
                    fps_proj = tf.compat.v1.layers.dense(fps, text_layer_size)
                else:
                    fps_proj = fps

                gate_input = tf.concat([fps_proj, text_proj], axis=1)
                gate = tf.compat.v1.layers.dense(gate_input, text_layer_size)
                gate = tf.nn.sigmoid(gate)
                fused = gate * fps_proj + (1 - gate) * text_proj
            else:
                fused = fps

            with tf.compat.v1.variable_scope('Forward_FC_1') as scope:
                fused = tf.compat.v1.layers.dense(fused, forward_layer_1_size)
                fused = tf.compat.v1.layers.batch_normalization(fused, training=is_training)
                fused = forward_act_func(fused)
                fused = tf.compat.v1.layers.dropout(fused, forward_dropout, training=is_training)

            if forward_layer_2_size != None:
                with tf.compat.v1.variable_scope('Forward_FC_2') as scope:
                    fused = tf.compat.v1.layers.dense(fused, forward_layer_2_size)
                    fused = tf.compat.v1.layers.batch_normalization(fused, training=is_training)
                    fused = forward_act_func(fused)
                    fused = tf.compat.v1.layers.dropout(fused, forward_dropout, training=is_training)

            if forward_layer_3_size != None:
                with tf.compat.v1.variable_scope('Forward_FC_3') as scope:
                    fused = tf.compat.v1.layers.dense(fused, forward_layer_3_size)
                    fused = tf.compat.v1.layers.batch_normalization(fused, training=is_training)
                    fused = forward_act_func(fused)
                    fused = tf.compat.v1.layers.dropout(fused, forward_dropout, training=is_training)

            fps = layers.make_fc_layer(fused, 2, is_training=is_training,
                                       with_bn=False, act_func=None)
            return fps, labels

    return DNN_5_Text()

def black_box_function(args_dict, train_samples, valid_samples, data_abs_path,
                       root_abs_path, text_feature_dir):
    tf.compat.v1.reset_default_graph()

    fp_size = args_dict['fp_size']
    radii = args_dict['fp_radii']
    batch_size = args_dict['batch_size']
    layer_1_size = args_dict['layer_1_size']
    layer_2_size = args_dict['layer_2_size']
    layer_3_size = args_dict['layer_3_size']
    act_fun = args_dict['act_fun']
    dropout = args_dict['dropout']
    merge = args_dict['merge']
    forward_layer_1_size = args_dict['forward_layer_1_size']
    forward_layer_2_size = args_dict['forward_layer_2_size']
    forward_layer_3_size = args_dict['forward_layer_3_size']
    forward_act_fun = args_dict['forward_act_fun']
    forward_dropout = args_dict['forward_dropout']
    fusion_type = args_dict.get('fusion_type', 'concat')

    data = make_dataset(fp_size, radii, data_abs_path, text_feature_dir=text_feature_dir)
    train_data, valid_data = data.split(train_samples=train_samples,
                                        valid_samples=valid_samples, with_fps=True)

    if len(train_data) < 4:
        text_dim = 768
        train_data = list(train_data) + [np.zeros((len(train_samples), text_dim), dtype=np.float32)]
        valid_data = list(valid_data) + [np.zeros((len(valid_samples), text_dim), dtype=np.float32)]

    snapshot_path = os.path.join(root_abs_path, 'bayes_snapshot/')
    model_name = 'ten_fold_cross_validation-FP-Text/'
    verify_dir_exists(os.path.join(snapshot_path, model_name))

    model_full_path = os.path.join(snapshot_path, model_name)
    if not os.path.exists(model_full_path):
        os.makedirs(model_full_path)

    if os.listdir(model_full_path) == []:
        dataset_name = 'Step_0/'
    else:
        l_ = [int(i.split('_')[1]) for i in os.listdir(model_full_path) if 'Step_' in i]
        dataset_name = 'Step_{}/'.format(max(l_) + 1)

    tf.compat.v1.reset_default_graph()
    model = build_model_with_text(layer_1_size, layer_2_size, layer_3_size,
                                  act_fun, dropout, merge, forward_layer_1_size,
                                  forward_layer_2_size, forward_layer_3_size,
                                  forward_act_fun, forward_dropout,
                                  fusion_type=fusion_type)

    exp_model = ModelText(model, train_data, valid_data, with_test=False,
                          snapshot_path=snapshot_path, use_subgraph=False,
                          use_desc=False, build_fc=True, use_text=True,
                          model_name=model_name, dataset_name=dataset_name + '/time_0')

    history = exp_model.fit(num_epoch=100, save_info=True, save_att=False, silence=0,
                            train_batch_size=batch_size, max_to_keep=1, metric='loss')

    loss = min(history['valid_cross_entropy'])
    tf.compat.v1.reset_default_graph()
    print('\nLoss: {}'.format(loss))
    return loss

if __name__ == '__main__':
    multiprocessing.freeze_support()
    tf.compat.v1.disable_eager_execution()

    from hyperopt import fmin, tpe, Trials, hp

    args_dict = {
        'fp_size': hp.choice('fp_size', [128, 256, 512, 1024, 2048, 4096]),
        'fp_radii': hp.choice('fp_radii', (1, 2, 3)),
        'batch_size': hp.choice('batch_size', (64, 128, 256)),
        'layer_1_size': hp.choice('layer_1_size', (128, 256, 512, 1024, 2048)),
        'layer_2_size': hp.choice('layer_2_size', (128, 256, 512, 1024, 2048, None)),
        'layer_3_size': hp.choice('layer_3_size', (128, 256, 512, 1024, 2048, None)),
        'act_fun': hp.choice('act_fun', (tf.nn.relu, tf.nn.elu, tf.nn.tanh)),
        'dropout': hp.uniform('dropout', 0.0, 0.75),
        'merge': hp.choice('merge', ('add', 'concat')),
        'forward_layer_1_size': hp.choice('forward_layer_1_size', (128, 256, 512, 1024, 2048)),
        'forward_layer_2_size': hp.choice('forward_layer_2_size', (128, 256, 512, 1024, 2048, None)),
        'forward_layer_3_size': hp.choice('forward_layer_3_size', (128, 256, 512, 1024, 2048, None)),
        'forward_act_fun': hp.choice('forward_act_fun', (tf.nn.relu, tf.nn.elu, tf.nn.tanh)),
        'forward_dropout': hp.uniform('forward_dropout', 0.0, 0.75),
        'fusion_type': hp.choice('fusion_type', ('concat', 'add', 'gate', 'none'))
    }

    root_abs_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_abs_path = os.path.join(root_abs_path, "data")
    text_feature_dir = os.path.join(root_abs_path, "data/processed_features")

    fold_10_path = os.path.join(data_abs_path, 'Fold_10.dir')
    if not os.path.exists(fold_10_path):
        raise FileNotFoundError(f"未找到Fold_10.dir文件: {fold_10_path}")
    fold_10 = eval(open(fold_10_path).read())

    print("预加载数据集，过滤无效样本...")
    temp_data = DatasetText(os.path.join(data_abs_path, 'CC_Table/CC_Table.tab'),
                            mol_blocks_dir=os.path.join(data_abs_path, 'Mol_Blocks.dir'),
                            text_feature_dir=text_feature_dir)
    temp_data.make_embedding_dataset(fp_type='ecfp', nBits=1024, radii=2,
                                     processes=4, make_dataframe=True)

    valid_sample_names = list(temp_data.dataframe.keys())

    fold_samples = fold_10['fold-0']['train'] + fold_10['fold-0']['valid']
    Samples = [name for name in fold_samples if name in valid_sample_names]
    print(f"过滤前样本数: {len(fold_samples)}，过滤后有效样本数: {len(Samples)}")

    random.shuffle(Samples)
    num_sample = len(Samples)
    train_num = int(0.9 * num_sample)
    train_samples = Samples[:train_num]
    valid_samples = Samples[train_num:]

    trials_save_file = 'trials_save_file-FP_text.pkl'
    if os.path.exists(trials_save_file):
        print(f"找到已保存的trials文件: {trials_save_file}")
        print("正在加载之前的trials数据...")

        with open(trials_save_file, 'rb') as f:
            trials = pickle.load(f)

        completed_trials = len([t for t in trials.trials if t['result']['status'] == 'ok'])
        print(f"已完成的试验数量: {completed_trials}/10")

        remaining_evals = max(0, 10 - completed_trials)

        if remaining_evals > 0:
            print(f"从中断处继续运行，剩余试验数量: {remaining_evals}")
        else:
            print("所有试验已完成，显示最优结果:")
            print(trials.best_trial['result'])
            sys.exit(0)
    else:
        print("未找到已保存的trials文件，开始新的贝叶斯优化...")
        trials = Trials()
        remaining_evals = 10

    print("开始贝叶斯优化搜索超参数...")

    try:
        best = fmin(
            fn=lambda args: black_box_function(args, train_samples, valid_samples,
                                               data_abs_path, root_abs_path, text_feature_dir),
            space=args_dict,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials,
            trials_save_file='trials_save_file-FP_text'
        )

        print('\n最优超参数:')
        print(best)
        np.save(os.path.join(root_abs_path, 'bayes_snapshot/ten_fold_cross_validation-FP-Text/best_params.npy'), best)
        print('最优超参数已保存')

    except KeyboardInterrupt:
        print("\n用户中断，保存当前进度...")
        with open(trials_save_file, 'wb') as f:
            pickle.dump(trials, f)
        print(f"trials已保存到: {trials_save_file}")
        print("下次运行将从当前进度继续")

    except Exception as e:
        print(f"\n发生错误: {e}")
        print("保存当前进度...")
        with open(trials_save_file, 'wb') as f:
            pickle.dump(trials, f)
        print(f"trials已保存到: {trials_save_file}")
        raise