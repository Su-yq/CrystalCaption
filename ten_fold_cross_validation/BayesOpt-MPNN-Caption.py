import runpy
import sys
import os
import multiprocessing

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU, Dropout, BatchNorm1d
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import Data, DataLoader
import numpy as np
import random
import time
from scipy.sparse import coo_matrix

from crosscaption.Dataset_text import DatasetText


class NN(torch.nn.Module):

    def __init__(self, dim, edge_feat_num, act_fun, lin1_size, lin2_size):
        super(NN, self).__init__()
        self.dim = dim
        self.edge_feat_num = edge_feat_num
        self.act_fun = act_fun
        self.lin1_size = lin1_size
        self.lin2_size = lin2_size


def build_mpnn_with_text(
        dim,
        mp_step,
        act_fun,
        lin1_size,
        lin2_size,
        processing_steps,
        readout_act_func,
        readout_p_dropout,
        readout_dense_1,
        readout_dense_2,
        readout_dense_3,
        text_hidden_dim=256,
        fusion_type='concat'
):
    class Net(torch.nn.Module):
        def __init__(self, dim=64, edge_feat_num=4, node_feat_num=34, text_feat_dim=768):
            super(Net, self).__init__()
            self.lin0 = torch.nn.Linear(node_feat_num, dim)
            self.e_nn = NN(dim, edge_feat_num, act_fun, lin1_size, lin2_size)
            self.conv = NNConv(dim, dim, self.e_nn, aggr='mean', root_weight=False)
            self.gru = GRU(dim, dim)
            self.set2set = Set2Set(dim, processing_steps=processing_steps)

            self.text_proj = torch.nn.Sequential(
                Linear(text_feat_dim, text_hidden_dim),
                BatchNorm1d(text_hidden_dim),
                act_fun,
                Dropout(readout_p_dropout)
            )

            if fusion_type == 'concat':
                fusion_input_size = dim * 2 + text_hidden_dim
            else:
                fusion_input_size = dim * 2

            self.readout_lin1 = Linear(fusion_input_size, readout_dense_1)
            self.readout_bn1 = BatchNorm1d(readout_dense_1)

            if readout_dense_2 is not None:
                self.readout_lin2 = Linear(readout_dense_1, readout_dense_2)
                self.readout_bn2 = BatchNorm1d(readout_dense_2)

            if readout_dense_3 is not None:
                in_dim = readout_dense_2 if readout_dense_2 is not None else readout_dense_1
                self.readout_lin3 = Linear(in_dim, readout_dense_3)
                self.readout_bn3 = BatchNorm1d(readout_dense_3)

            out_dim = readout_dense_3 or readout_dense_2 or readout_dense_1
            self.readout_lin_out = Linear(out_dim, 2)

            self.fusion_type = fusion_type
            self.text_hidden_dim = text_hidden_dim

        def forward(self, data):
            out = F.relu(self.lin0(data.x))
            h = out.unsqueeze(0)

            for i in range(mp_step):
                m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
                out, h = self.gru(m.unsqueeze(0), h)
                out = out.squeeze(0)

            out = self.set2set(out, data.batch)

            text_features = data.text_feat
            if text_features.dim() == 1:
                text_features = text_features.unsqueeze(0)

            text_proj = self.text_proj(text_features)

            if self.fusion_type == 'concat':
                fused = torch.cat([out, text_proj], dim=1)
            elif self.fusion_type == 'add':
                if out.size(1) != text_proj.size(1):
                    out_proj = Linear(out.size(1), self.text_hidden_dim)(out)
                else:
                    out_proj = out
                fused = out_proj + text_proj
            elif self.fusion_type == 'gate':
                if out.size(1) != text_proj.size(1):
                    out_proj = Linear(out.size(1), self.text_hidden_dim)(out)
                else:
                    out_proj = out
                gate_input = torch.cat([out_proj, text_proj], dim=1)
                gate = Linear(gate_input.size(1), self.text_hidden_dim)(gate_input)
                gate = torch.sigmoid(gate)
                fused = gate * out_proj + (1 - gate) * text_proj
            else:
                fused = out

            out = self.readout_lin1(fused)
            out = self.readout_bn1(out)
            out = readout_act_func(out)
            out = F.dropout(out, p=readout_p_dropout)

            if readout_dense_2 is not None:
                out = self.readout_lin2(out)
                out = self.readout_bn2(out)
                out = readout_act_func(out)
                out = F.dropout(out, p=readout_p_dropout)

            if readout_dense_3 is not None:
                out = self.readout_lin3(out)
                out = self.readout_bn3(out)
                out = readout_act_func(out)
                out = F.dropout(out, p=readout_p_dropout)

            out = self.readout_lin_out(out)
            return F.log_softmax(out, dim=-1)

    return Net(dim=dim, edge_feat_num=4, node_feat_num=34, text_feat_dim=768)


def verify_dir_exists(dirname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def coo_format(A):
    coo_A = np.zeros([A.shape[0], A.shape[2]])
    for i in range(A.shape[1]):
        coo_A = coo_A + A[:, i, :]
    coo_A = coo_matrix(coo_A)
    edge_index = [coo_A.row, coo_A.col]
    edge_attr = []
    for j in range(len(edge_index[0])):
        edge_attr.append(A[edge_index[0][j], :, edge_index[1][j]])
    edge_attr = np.array(edge_attr)
    return edge_index, edge_attr


def make_dataset(data_abs_path, text_feature_dir=None):
    data1 = DatasetText(
        os.path.join(data_abs_path, 'CC_Table/CC_Table.tab'),
        mol_blocks_dir=os.path.join(data_abs_path, 'Mol_Blocks.dir'),
        text_feature_dir=text_feature_dir
    )
    data1.make_graph_dataset(Desc=0, A_type='OnlyCovalentBond', hbond=0,
                             pipi_stack=0, contact=0, make_dataframe=True)
    return data1


class GetInputDataText:
    def __init__(self, dataframe):
        self.graphs = {}
        df = dataframe
        for i in df:
            x = torch.tensor(df[i]['V'][:df[i]['graph_size']])
            edge_index, edge_attr = coo_format(df[i]['A'])
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            y = torch.tensor(np.array([df[i]['label']]), dtype=torch.long)

            text_feat = torch.tensor(df[i].get('text_features', np.zeros(768)),
                                     dtype=torch.float)

            data = Data(x=x, y=y, edge_attr=edge_attr, edge_index=edge_index,
                        tag=i, text_feat=text_feat)
            self.graphs[i] = data

    def split(self, train_samples=None, valid_samples=None, batch_size=128):
        train_list = [self.graphs[i] for i in train_samples if i in self.graphs]
        valid_list = [self.graphs[i] for i in valid_samples if i in self.graphs]
        return train_list, valid_list


def black_box_function(args_dict, root_abs_path, train_list, valid_list, text_feature_dir):
    batch_size = args_dict['batch_size']
    dim = args_dict['dim']
    mp_step = args_dict['mp_step']
    act_fun = args_dict['act_fun']
    lin1_size = args_dict['lin1_size']
    lin2_size = args_dict['lin2_size']
    processing_steps = args_dict['processing_steps']
    readout_act_func = args_dict['readout_act_func']
    readout_p_dropout = args_dict['readout_p_dropout']
    readout_dense_1 = args_dict['readout_dense_1']
    readout_dense_2 = args_dict['readout_dense_2']
    readout_dense_3 = args_dict['readout_dense_3']
    fusion_type = args_dict.get('fusion_type', 'concat')

    print(str(args_dict))

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_list, batch_size=batch_size, shuffle=False)

    snapshot_path = os.path.join(root_abs_path, 'bayes_snapshot/')
    model_name = 'ten_fold_cross_validation-MPNN-Text/'
    verify_dir_exists(os.path.join(snapshot_path, model_name))

    if os.listdir(os.path.join(snapshot_path, model_name)) == []:
        dataset_name = 'Step_0/'
    else:
        l_ = [int(i.split('_')[1]) for i in os.listdir(os.path.join(snapshot_path, model_name)) if 'Step_' in i]
        dataset_name = 'Step_{}/'.format(max(l_) + 1)

    MPNN = build_mpnn_with_text(
        dim, mp_step, act_fun, lin1_size, lin2_size, processing_steps,
        readout_act_func, readout_p_dropout, readout_dense_1,
        readout_dense_2, readout_dense_3, fusion_type=fusion_type
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MPNN.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.6, patience=100, min_lr=0.00001
    )

    history = {
        'Train Loss': [], 'Train Acc': [],
        'Valid Acc': [], 'Valid Loss': []
    }

    path = os.path.join(snapshot_path, model_name, dataset_name, 'time_0')
    verify_dir_exists(path)

    reports = {'valid acc': 0.0, 'valid loss': float('inf')}

    for epoch in range(1, 101):
        start_time = time.time()

        model.train()
        train_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, data.y)
            loss.backward()
            train_loss += loss.item() * data.num_graphs
            optimizer.step()

        train_loss /= len(train_loader.dataset)

        model.eval()
        valid_loss = 0
        correct = 0
        with torch.no_grad():
            for data in valid_loader:
                data = data.to(device)
                output = model(data)
                valid_loss += F.nll_loss(output, data.y).item() * data.num_graphs
                pred = output.max(dim=1)[1]
                correct += pred.eq(data.y).sum().item()

        valid_loss /= len(valid_loader.dataset)
        valid_acc = correct / len(valid_loader.dataset)

        scheduler.step(valid_loss)

        if valid_loss < reports['valid loss']:
            torch.save(model.state_dict(), os.path.join(path, 'ModelParams.pkl'))
            reports['valid loss'] = valid_loss
            reports['valid acc'] = valid_acc

        elapsed_time = time.time() - start_time
        print(f'Epoch:{epoch:03d} ==> Train Loss: {train_loss:.5f}, '
              f'Valid Acc: {valid_acc * 100:.2f}, Valid loss: {valid_loss:.5f}, '
              f'Elapsed Time:{elapsed_time:.2f} s')

    print(f'\n最终损失: {reports["valid loss"]}')
    return reports['valid loss']


if __name__ == '__main__':
    multiprocessing.freeze_support()
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

    data_list = GetInputDataText(temp_data.dataframe)
    train_list, valid_list = data_list.split(train_samples=train_samples,
                                             valid_samples=valid_samples)

    args_dict = {
        'batch_size': hp.choice('batch_size', (32,)),
        'dim': hp.choice('dim', (32, 64, 128)),
        'mp_step': hp.choice('mp_step', (2, 3, 4, 5)),
        'edge_feat_num': hp.choice('edge_feat_num', (4,)),
        'act_fun': hp.choice('act_fun', (F.relu,)),
        'lin1_size': hp.choice('lin1_size', (128, 512, 1024, 2048, None)),
        'lin2_size': hp.choice('lin2_size', (128, 512, 1024, 2048, None)),
        'processing_steps': hp.choice('processing_steps', (2, 3, 4, 5)),
        'readout_dense_1': hp.choice('readout_dense_1', (64, 128, 256, 512, 1024)),
        'readout_dense_2': hp.choice('readout_dense_2', (64, 128, 256, 512, 1024, None)),
        'readout_dense_3': hp.choice('readout_dense_3', (64, 128, 256, 512, 1024, None)),
        'readout_act_func': hp.choice('readout_act_func', (F.relu,)),
        'readout_p_dropout': hp.uniform('readout_p_dropout', 0.0, 0.75),
        'fusion_type': hp.choice('fusion_type', ('concat', 'add', 'gate'))
    }

    trials = Trials()
    best = fmin(
        fn=lambda args: black_box_function(args, root_abs_path, train_list,
                                           valid_list, text_feature_dir),
        space=args_dict,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
        trials_save_file='trials_save_file-mpnn_text'
    )
    print('\n最优参数:')
    print(best)