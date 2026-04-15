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
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import deepchem as dc
from deepchem.models import GraphConvModel
from deepchem.data import NumpyDataset
import pybel
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def verify_dir_exists(dirname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


class GetDeepChemCocrystalDatasetText:

    def __init__(self, mol_blocks, cc_table, text_feature_path=None, removeHs=False):
        self.text_features = None
        self.text_feature_map = {}

        if text_feature_path and os.path.exists(text_feature_path):
            self.text_features = np.load(text_feature_path).astype(np.float32)
            print(f"加载文本特征: {self.text_features.shape}")

        table = [line.strip().split('\t') for line in open(cc_table).readlines()]
        mol_block = eval(open(mol_blocks).read())
        convf = dc.feat.ConvMolFeaturizer()
        exception = {'BOVQUY', 'CEJPAK', 'GAWTON', 'GIPTAA', 'IDIGUY', 'LADBIB01',
                     'PIGXUY', 'SIFBIT', 'SOJZEW', 'TOFPOW', 'QOVZIK', 'RIJNEF',
                     'SIBFAK', 'SIBFEO', 'TOKGIJ', 'TOKGOP', 'TUQTEE', 'BEDZUF'}

        self.mol_obj_dic = {}
        self.text_indices = {}

        for idx, line in enumerate(table):
            tag = line[-1]
            if tag in exception:
                continue

            c1 = Chem.MolFromMolBlock(mol_block[line[0]], removeHs=removeHs)
            c2 = Chem.MolFromMolBlock(mol_block[line[1]], removeHs=removeHs)

            if c1 is None or c2 is None:
                continue

            y = int(line[2])

            self.mol_obj_dic[tag] = [convf._featurize(c1), convf._featurize(c2), y]
            if self.text_features is not None and idx < len(self.text_features):
                self.text_feature_map[tag] = idx

    def dataset_generator(self, sample_list, data_aug=False):
        X, Y, Tags, Text_Features = [], [], [], []

        for sample in sample_list:
            if sample not in self.mol_obj_dic:
                continue

            x = [self.mol_obj_dic[sample][0], self.mol_obj_dic[sample][1]]
            y = self.mol_obj_dic[sample][-1]

            X.append(x)
            Y.append(y)
            Tags.append(sample)

            if sample in self.text_feature_map:
                text_feat = self.text_features[self.text_feature_map[sample]]
            else:
                text_feat = np.zeros(768, dtype=np.float32)
            Text_Features.append(text_feat)

            if data_aug:
                X.append([self.mol_obj_dic[sample][1], self.mol_obj_dic[sample][0]])
                Y.append(y)
                Tags.append(sample)
                Text_Features.append(text_feat)

        dataset = dc.data.NumpyDataset(X=np.array(X), y=np.array(Y))
        dataset.tags = Tags
        dataset.text_features = np.array(Text_Features)

        return dataset


class GraphConvModelText(GraphConvModel):

    def __init__(self, n_tasks, text_feature_dim=768, fusion_type='concat',
                 text_hidden_dim=256, **kwargs):
        self.text_feature_dim = text_feature_dim
        self.fusion_type = fusion_type
        self.text_hidden_dim = text_hidden_dim

        super().__init__(n_tasks, **kwargs)

    def build_graph(self):
        import tensorflow as tf

        atom_features = self.atom_features
        degree_slice = self.degree_slice
        membership = self.membership
        dropout_switch = self.dropout_switch

        output = super().build_graph()

        self.text_features = tf.compat.v1.placeholder(
            tf.float32, shape=[None, self.text_feature_dim], name='text_features'
        )

        with tf.compat.v1.variable_scope('Text_Processing'):
            text_proj = tf.compat.v1.layers.dense(self.text_features, self.text_hidden_dim)
            text_proj = tf.compat.v1.layers.batch_normalization(text_proj)
            text_proj = tf.nn.relu(text_proj)

        if self.fusion_type == 'concat':
            if len(output.shape) == 3:
                output = tf.reshape(output, [tf.shape(output)[0], -1])
            fused = tf.concat([output, text_proj], axis=1)
        elif self.fusion_type == 'add':
            if output.shape[-1] != text_proj.shape[-1]:
                output = tf.compat.v1.layers.dense(output, self.text_hidden_dim)
            fused = output + text_proj
        elif self.fusion_type == 'gate':
            if output.shape[-1] != text_proj.shape[-1]:
                output_proj = tf.compat.v1.layers.dense(output, self.text_hidden_dim)
            else:
                output_proj = output
            gate_input = tf.concat([output_proj, text_proj], axis=1)
            gate = tf.compat.v1.layers.dense(gate_input, self.text_hidden_dim)
            gate = tf.nn.sigmoid(gate)
            fused = gate * output_proj + (1 - gate) * text_proj
        else:
            fused = output

        return fused

    def default_generator(self, dataset, epochs=1, mode='fit',
                          deterministic=True, pad_batches=True):
        for epoch in range(epochs):
            for (X_b, y_b, w_b, ids_b) in dataset.iterbatches(
                    batch_size=self.batch_size,
                    deterministic=deterministic,
                    pad_batches=pad_batches):

                X_b = X_b.reshape(-1)

                if self.mode == 'classification':
                    y_b = dc.metrics.to_one_hot(y_b.flatten(), self.n_classes).reshape(
                        -1, self.n_tasks, self.n_classes)

                multiConvMol = dc.feat.mol_graphs.ConvMol.agglomerate_mols(X_b)
                n_samples = np.array(y_b.shape[0])

                if mode == 'predict':
                    dropout = np.array(0.0)
                else:
                    dropout = np.array(1.0)

                inputs = [
                    multiConvMol.get_atom_features(),
                    multiConvMol.deg_slice,
                    np.array(multiConvMol.membership),
                    n_samples,
                    dropout
                ]

                for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
                    inputs.append(multiConvMol.get_deg_adjacency_lists()[i])

                text_features_batch = []
                for id_b in ids_b:
                    if id_b in dataset.text_feature_map:
                        idx = dataset.text_feature_map[id_b]
                        text_features_batch.append(dataset.text_features[idx])
                    else:
                        text_features_batch.append(np.zeros(self.text_feature_dim, dtype=np.float32))

                inputs.append(np.array(text_features_batch))

                yield (inputs, [y_b], [w_b])


def black_box_function(args_dict, root_abs_path, train_set, valid_set):
    tf.compat.v1.reset_default_graph()

    graph_conv_layers_1 = args_dict['graph_conv_layers_1']
    graph_conv_layers_2 = args_dict['graph_conv_layers_2']
    graph_conv_layers_3 = args_dict['graph_conv_layers_3']
    graph_conv_act_fun = args_dict['graph_conv_act_fun']
    graph_gather_act_fun = args_dict['graph_gather_act_fun']
    merge = args_dict['merge']
    dense_layer_1 = args_dict['dense_layer_1']
    dense_layer_2 = args_dict['dense_layer_2']
    dense_layer_3 = args_dict['dense_layer_3']
    dense_act_fun = args_dict['dense_act_fun']
    dense_layer_dropout = args_dict['dense_layer_dropout']
    fusion_type = args_dict.get('fusion_type', 'concat')

    snapshot_path = os.path.join(root_abs_path, 'bayes_snapshot/')
    model_name = 'ten_fold_cross_validation-GCN-Text/'
    verify_dir_exists(os.path.join(snapshot_path, model_name))

    if os.listdir(os.path.join(snapshot_path, model_name)) == []:
        dataset_name = 'Step_0/'
    else:
        l_ = [int(i.split('_')[1]) for i in os.listdir(os.path.join(snapshot_path, model_name)) if 'Step_' in i]
        dataset_name = 'Step_{}/'.format(max(l_) + 1)

    path_to_save = os.path.join(snapshot_path, model_name, dataset_name, 'time_0')
    verify_dir_exists(path_to_save)

    model = GraphConvModelText(
        1,
        batch_size=128,
        graph_conv_layers=[graph_conv_layers_1, graph_conv_layers_2, graph_conv_layers_3],
        dense_layer_size=dense_layer_1,
        dropout=0.0,
        mode='classification',
        fusion_type=fusion_type
    )

    loss = model.fit_epoch(
        train_set, valid_set, epochs=100,
        model_dir=path_to_save
    )

    tf.compat.v1.reset_default_graph()
    print(f'\nLoss: {loss}')
    return loss


if __name__ == '__main__':
    multiprocessing.freeze_support()
    tf.compat.v1.disable_eager_execution()

    from hyperopt import fmin, tpe, Trials, hp

    root_abs_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_abs_path = os.path.join(root_abs_path, "data")
    text_feature_path = os.path.join(root_abs_path, "data/processed_features/text_features.npy")

    fold_10_path = os.path.join(data_abs_path, 'Fold_10.dir')
    fold_10 = eval(open(fold_10_path, 'r', encoding='utf-8').read())

    print("加载数据集和文本特征...")
    dataset = GetDeepChemCocrystalDatasetText(
        mol_blocks=os.path.join(data_abs_path, 'Mol_Blocks.dir'),
        cc_table=os.path.join(data_abs_path, 'CC_Table/CC_Table.tab'),
        text_feature_path=text_feature_path
    )

    Samples = fold_10['fold-0']['train'] + fold_10['fold-0']['valid']
    random.shuffle(Samples)

    num_sample = len(Samples)
    train_num = int(0.9 * num_sample)
    train_samples = Samples[:train_num]
    valid_samples = Samples[train_num:]

    train_set = dataset.dataset_generator(train_samples)
    valid_set = dataset.dataset_generator(valid_samples)

    args_dict = {
        'graph_conv_layers_1': hp.choice('graph_conv_layers_1', (64, 128, 256)),
        'graph_conv_layers_2': hp.choice('graph_conv_layers_2', (64, 128, 256, None)),
        'graph_conv_layers_3': hp.choice('graph_conv_layers_3', (64, 128, 256, None)),
        'graph_conv_act_fun': hp.choice('graph_conv_act_fun', (tf.nn.relu, tf.nn.elu, tf.nn.tanh)),
        'graph_gather_act_fun': hp.choice('graph_gather_act_fun', (tf.nn.relu, tf.nn.elu, tf.nn.tanh)),
        'merge': hp.choice('merge', ('add', 'concat')),
        'dense_layer_1': hp.choice('dense_layer_1', (64, 128, 256, 512, 1024)),
        'dense_layer_2': hp.choice('dense_layer_2', (64, 128, 256, 512, 1024, None)),
        'dense_layer_3': hp.choice('dense_layer_3', (64, 128, 256, 512, 1024, None)),
        'dense_act_fun': hp.choice('dense_act_fun', (tf.nn.relu, tf.nn.elu, tf.nn.tanh)),
        'dense_layer_dropout': hp.uniform('dense_layer_dropout', 0.0, 0.75),
        'fusion_type': hp.choice('fusion_type', ('concat', 'add', 'gate'))
    }

    trials = Trials()
    best = fmin(
        fn=lambda args: black_box_function(args, root_abs_path, train_set, valid_set),
        space=args_dict,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
        trials_save_file='trials_save_file-gcn_text'
    )
    print('\n最优参数:')
    print(best)