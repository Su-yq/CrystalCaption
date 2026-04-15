import sys
import os
import xlsxwriter
import pandas as pd

#python predict.py -table ./data/Test/Test_Table.tab -mol_dir ./data/Mol_Blocks.dir -fmt sdf -out predictions.xlsx -model_path ./model/CC_Models/model_0
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from openbabel import pybel

    sys.modules['pybel'] = pybel
    print("Successfully patched pybel from openbabel")
except ImportError:
    try:
        import pybel

        print("Direct pybel import successful")
    except ImportError:
        print("Creating pybel compatibility layer...")
        from rdkit import Chem


        class PybelCompat:
            def readstring(self, format_str, data):
                if format_str.lower() == 'sdf':
                    return Chem.MolFromMolBlock(data)
                elif format_str.lower() in ['mol', 'mol2']:
                    return Chem.MolFromMolBlock(data)
                else:
                    return Chem.MolFromSmiles(data)

            def readfile(self, format_str, filename):
                with open(filename, 'r') as f:
                    content = f.read()
                return self.readstring(format_str, content)


        pybel_compat = PybelCompat()

        class PybelModule:
            readstring = pybel_compat.readstring
            readfile = pybel_compat.readfile


        sys.modules['pybel'] = PybelModule()

import tensorflow as tf
import numpy as np
import time
from Featurize import *
import argparse
import glob
from multiprocessing import Pool
import numpy as np
import time
from rdkit import Chem
from rdkit.Chem import rdDepictor, Draw
from io import BytesIO
import os
import argparse


def GetNodeMask(graph_size, max_size=None):
    if max_size == None:
        max_size = np.max(graph_size)
    return np.array(
        [np.pad(np.ones([s, 1]), ((0, max_size - s), (0, 0)), 'constant', constant_values=(0)) for s in graph_size],
        dtype=np.float32)


class Dataset(object):

    def __init__(self, table_dir, mol_dir='./', mol_file_type='sdf'):
        try:
            self.table = eval(open(table_dir).read())
        except:
            self.table = [i.strip().split('\t') for i in open(table_dir).read().strip().split('\n')]
        self.mol_dir = mol_dir
        self.mol_file_type = '.' + mol_file_type

    def _task(self, items):
        array = {}
        tag = items[0].split('.')[0] + '&' + items[1].split('.')[0]
        array.setdefault('tags', tag)
        file1 = self.mol_dir + '/' + items[0].split('.')[0] + self.mol_file_type
        file2 = self.mol_dir + '/' + items[1].split('.')[0] + self.mol_file_type

        print(f"Processing: {tag}")
        print(f"  File1: {file1}")
        print(f"  File2: {file2}")
        print(f"  Files exist: {os.path.exists(file1)}, {os.path.exists(file2)}")

        try:
            c1 = Coformer(file1)
            c2 = Coformer(file2)
            cc = Cocrystal(c1, c2)
            label = int(items[2])
            array.setdefault('labels', label)
            subgraph_size = np.array([c1.atom_number, c2.atom_number])
            array.setdefault('subgraph_size', subgraph_size)
            if self.Desc:
                desc = cc.descriptors()
                array.setdefault('global_state', desc)
            if self.A_type:
                A = cc.CCGraphTensor(t_type=self.A_type, hbond=self.hbond, pipi_stack=self.pipi_stack,
                                     contact=self.contact)
                V = cc.VertexMatrix.feature_matrix()
                array.setdefault('A', A)
                array.setdefault('V', V)
            if self.fp_type:
                fp = cc.Fingerprints(fp_type=self.fp_type, nBits=self.nBits)
                array.setdefault('fingerprints', fp)
            return array
        except Exception as e:
            print(f'Bad input sample: {tag}, Error: {str(e)}')
            return None

    def _PreprocessData(self, max_graph_size=None):
        graph_size = np.array([a.shape[0] for a in self.A]).astype(np.int32)
        if max_graph_size:
            largest_graph = max_graph_size
        else:
            largest_graph = max(graph_size)
        graph_vertices = []
        graph_adjacency = []
        for i in range(len(self.V)):
            graph_vertices.append(np.pad(self.V[i].astype(np.float32),
                                         ((0, largest_graph - self.V[i].shape[0]), (0, 0)),
                                         'constant', constant_values=(0)))
            new_A = self.A[i]
            graph_adjacency.append(np.pad(new_A.astype(np.float32),
                                          ((0, largest_graph - new_A.shape[0]), (0, 0),
                                           (0, largest_graph - new_A.shape[0])),
                                          'constant', constant_values=(0)))
        self.V = np.stack(graph_vertices, axis=0)
        self.A = np.stack(graph_adjacency, axis=0)
        self.labels = self.labels.astype(np.int32)
        if 'global_state' in self.__dict__:
            self.global_state = self.global_state.astype(np.float32)
        self.masks = GetNodeMask(graph_size, max_size=largest_graph)
        self.graph_size = graph_size
        self.subgraph_size = self.subgraph_size.astype(np.int32)

    def make_graph_dataset(self, Desc=0, A_type='OnlyCovalentBond', hbond=0, pipi_stack=0, contact=0,
                           processes=15, max_graph_size=160):  # make_dataframe=True
        self.Desc = Desc
        self.A_type = A_type
        self.hbond = hbond
        self.pipi_stack = pipi_stack
        self.contact = contact
        self.fp_type = None
        start = time.time()
        pool = Pool(processes=processes)
        D = pool.map(self._task, [items for items in self.table])
        pool.close()
        pool.join()
        self.data_attr_names = D[0].keys()
        attrs = self.__dict__
        for k in self.data_attr_names:
            attrs.setdefault(k, [])
        for i in [j for j in D if j != None]:
            for j in i:
                attrs[j].append(i[j])
        for l in attrs:
            if l in self.data_attr_names:
                attrs[l] = np.array(attrs[l])
        del D
        self._PreprocessData(max_graph_size=max_graph_size)
        end = time.time()
        t = round(end - start, 2)
        print('Elapsed Time: ' + str(t) + ' s')

    def save(self, name='DataForInfer.npz'):
        if 'global_state' in self.__dict__:
            np.savez(name, V=self.V, A=self.A, labels=self.labels, masks=self.masks, graph_size=self.graph_size,
                     tags=self.tags, global_state=self.global_state, subgraph_size=self.subgraph_size)
        else:
            np.savez(name, V=self.V, A=self.A, labels=self.labels, masks=self.masks, graph_size=self.graph_size,
                     tags=self.tags, subgraph_size=self.subgraph_size)
        del self.V, self.A, self.labels, self.masks, self.graph_size, self.tags, self.subgraph_size
        if 'global_state' in self.__dict__:
            del self.global_state

class DataLoader(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file, allow_pickle=1)
        for key in data:
            self.__dict__[key] = data[key]


def get_inputs(model_path):
    saver = tf.train.import_meta_graph(model_path)
    graph = tf.get_default_graph()
    placeholders = {}
    V = graph.get_tensor_by_name('V_input:0')
    placeholders.setdefault(V.name.split('_input')[0], V)
    A = graph.get_tensor_by_name('AdjMat_input:0')
    placeholders.setdefault('A', A)
    labels = graph.get_tensor_by_name('labels_input:0')
    placeholders.setdefault(labels.name.split('_input')[0], labels)
    masks = graph.get_tensor_by_name('masks_input:0')
    placeholders.setdefault(masks.name.split('_input')[0], masks)
    try:
        graph_size = graph.get_tensor_by_name('graph_size_input:0')
        placeholders.setdefault(graph_size.name.split('_input')[0], graph_size)
    except:
        print("The name 'graph_size_input:0' refers to a Tensor which does not exist.")
    try:
        tags = graph.get_tensor_by_name('tags_input:0')
        placeholders.setdefault(tags.name.split('_input')[0], tags)
    except:
        print("The name 'tags_input:0' refers to a Tensor which does not exist.")
    try:
        global_state = graph.get_tensor_by_name('global_state_input:0')
        placeholders.setdefault(global_state.name.split('_input')[0], global_state)
    except:
        print("The name 'global_state_input:0' refers to a Tensor which does not exist.")
    try:
        subgraph_size = graph.get_tensor_by_name('subgraph_size_input:0')
        placeholders.setdefault(subgraph_size.name.split('_input')[0], subgraph_size)
    except:
        print("The name 'subgraph_size_input:0' refers to a Tensor which does not exist.")
    return saver, graph, placeholders


def get_feed_dict(data, placeholders):
    feed_dict = {}
    for key in ['V', 'A', 'labels', 'tags', 'global_state', 'masks', 'graph_size', 'subgraph_size']:
        feed_dict.setdefault(placeholders[key], data.__dict__[key])
    return feed_dict


class Inference(object):
    def __init__(self, table_path, meta_file, mol_dir='./', mol_file_type='sdf'):
        data = Dataset(table_path, mol_dir=mol_dir, mol_file_type=mol_file_type)
        data.make_graph_dataset(Desc=1, A_type='OnlyCovalentBond', hbond=0, pipi_stack=0, contact=0,
                                processes=8, max_graph_size=160)
        # data_buffer = BytesIO()
        data.save(name='data_buffer.npz')
        # print(data_buffer.getvalue())
        data = DataLoader('data_buffer.npz')
        os.remove('data_buffer.npz')
        self.saver, self.graph, self.placeholders = get_inputs(meta_file)
        self.feed_dict = get_feed_dict(data, self.placeholders)
        is_training = self.graph.get_tensor_by_name('is_training:0')
        self.feed_dict[is_training] = 0

    def predict(self, path):
        model = tf.train.latest_checkpoint(path)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            out = self.graph.get_tensor_by_name('final/add:0')
            tags = self.graph.get_tensor_by_name('tags_input:0')
            self.saver.restore(sess, model)
            pred, tags = sess.run([out, tags], feed_dict=self.feed_dict)
        sess.close()
        return pred, tags


def argmax(pred):
    return pred.index(max(pred))


def Bagging(scores):
    pred_labels = [[np.argmax(i) for i in score] for score in scores]
    all_pred_labels = np.array(pred_labels).sum(axis=0)
    bagging = all_pred_labels > 5
    # bagging_labels = np.array(['Yes' if i else 'No' for i in bagging ])
    sum_scores = np.array(scores).sum(axis=0)
    return sum_scores[:, 1]  # , bagging_labels


def GetCoformerSmiles(cc_table, mol_dir='./coformers'):
    table = [line.strip().split('\t') for line in open(cc_table).readlines()]
    info = []
    for items in table:
        m1 = Coformer(mol_dir + '/' + items[0], removeh=True).rdkit_mol
        m2 = Coformer(mol_dir + '/' + items[1], removeh=True).rdkit_mol
        smi1 = Chem.MolToSmiles(m1)
        smi2 = Chem.MolToSmiles(m2)
        info.append([smi1, smi2])
    return info


def main(table, mol_dir, fmt='sdf', xlsx_name='Result.xlsx', model_path='./snapshot/CCGNet_block/CC_Dataset/'):

    paths = glob.glob(model_path + '/*/')
    meta_file = tf.train.latest_checkpoint(paths[0]) + '.meta'
    infer = Inference(table, meta_file, mol_dir=mol_dir, mol_file_type=fmt)

    score_pool = []
    for p in paths:
        score, tags = infer.predict(p)
        score_pool.append(score)
    sum_scores = Bagging(score_pool)
    Smiles = np.array(GetCoformerSmiles(table, mol_dir=mol_dir))
    sorted_args = np.argsort(sum_scores)[::-1]
    sum_scores = sum_scores[sorted_args]
    tags = tags[sorted_args]
    Smiles = Smiles[sorted_args]


    header = ['Tag', 'Score']
    item_style = {'align': 'center', 'valign': 'vcenter', 'top': 2, 'left': 2,
                  'right': 2, 'bottom': 2, 'text_wrap': 1}
    header_style = {'bold': 1, 'valign': 'vcenter', 'align': 'center', 'top': 2,
                    'left': 2, 'right': 2, 'bottom': 2}
    number_style = {'num_format': '0.00', 'align': 'center', 'valign': 'vcenter', 'top': 2, 'left': 2,
                    'right': 2, 'bottom': 2}

    workbook = xlsxwriter.Workbook(xlsx_name)
    ItemStyle = workbook.add_format(item_style)
    HeaderStyle = workbook.add_format(header_style)
    NumberStyle = workbook.add_format(number_style)

    worksheet = workbook.add_worksheet()
    worksheet.set_column('A:A', 30)
    worksheet.set_column('B:B', 15)
    for ix_, i in enumerate(header):
        worksheet.write(0, ix_, i, HeaderStyle)

    for ix, i in enumerate(tags):
        smi1, smi2 = Smiles[ix]
        score = sum_scores[ix]
        worksheet.set_row(ix + 1, 30)
        worksheet.write(ix + 1, 0, i, ItemStyle)
        worksheet.write_number(ix + 1, 1, score, NumberStyle)
    workbook.close()

def parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-table', type=str, help='The table of coformer pairs')
    parser.add_argument('-mol_dir', type=str, default='./', help='path of molecular file, default is "./"')
    parser.add_argument('-out', type=str, default='Result.xlsx',
                        help='name of predictive file, default is "Result.xlsx"')
    parser.add_argument('-fmt', type=str, default='sdf',
                        help='format of molecular file，which support sdf, mol, mol2. default is sdf',
                        choices=['sdf', 'mol', 'mol2'])
    parser.add_argument('-model_path', type=str, default='./model/CC_Models/',
                        help='The path where the models are stored, default is ./snapshot/CCGNet_block/CC_Dataset/')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parameter()
    main(args.table, args.mol_dir, fmt=args.fmt, model_path=args.model_path, xlsx_name=args.out)
