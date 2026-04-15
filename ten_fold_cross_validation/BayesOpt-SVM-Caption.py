import runpy
import sys
import os
import multiprocessing

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from crosscaption.Dataset_text import DatasetText
import numpy as np
from hyperopt import fmin, tpe, Trials, hp, STATUS_OK
import random


def make_dataset(data_abs_path, text_feature_dir=None):
    data1 = DatasetText(
        os.path.join(data_abs_path, 'CC_Table/CC_Table.tab'),
        mol_blocks_dir=os.path.join(data_abs_path, 'Mol_Blocks.dir'),
        text_feature_dir=text_feature_dir
    )
    data1.make_graph_dataset(Desc=1, A_type='OnlyCovalentBond', hbond=0,
                             pipi_stack=0, contact=0, make_dataframe=True)
    return data1


def extract_features_with_text(data, samples, text_features_dict):
    features = []
    labels = []
    text_feats = []

    for sample in samples:
        if sample in data.dataframe:
            desc = data.dataframe[sample]['global_state']
            features.append(desc)
            labels.append(data.dataframe[sample]['label'])

            if sample in text_features_dict:
                text_feats.append(text_features_dict[sample])
            else:
                text_feats.append(np.zeros(768, dtype=np.float32))

    features = np.array(features)
    text_feats = np.array(text_feats)

    combined_features = np.concatenate([features, text_feats], axis=1)

    return combined_features, np.array(labels)


def black_box_function(args_dict, x_train, y_train, x_valid, y_valid):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_valid_scaled = scaler.transform(x_valid)

    clf = SVC(**args_dict, probability=False)
    clf.fit(x_train_scaled, y_train)
    valid_pred = clf.predict(x_valid_scaled)
    valid_acc = accuracy_score(y_valid, valid_pred)
    print(f'准确率: {valid_acc:.4f}, 参数: {args_dict}')
    return {'loss': 1 - valid_acc, 'status': STATUS_OK}


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

    text_features_dict = {}
    if hasattr(temp_data, 'text_feature_map'):
        text_features = np.load(os.path.join(text_feature_dir, 'text_features.npy'))
        for tag, idx in temp_data.text_feature_map.items():
            text_features_dict[tag] = text_features[idx]

    Samples = fold_10['fold-0']['train'] + fold_10['fold-0']['valid']
    random.seed(10)
    random.shuffle(Samples)

    num_sample = len(Samples)
    train_num = int(0.9 * num_sample)
    train_samples = Samples[:train_num]
    valid_samples = Samples[train_num:]

    x_train, y_train = extract_features_with_text(temp_data, train_samples, text_features_dict)
    x_valid, y_valid = extract_features_with_text(temp_data, valid_samples, text_features_dict)

    print(f"训练集形状: {x_train.shape}, 验证集形状: {x_valid.shape}")

    space4svc = {
        'C': hp.uniform('C', 0.1, 20),
        'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
        'gamma': hp.choice('gamma', ['scale', 'auto', hp.uniform('gamma_custom', 0.001, 1)]),
        'degree': hp.choice('degree', range(1, 6)),
        'coef0': hp.uniform('coef0', 0, 10)
    }

    trials = Trials()
    best = fmin(
        fn=lambda args: black_box_function(args, x_train, y_train, x_valid, y_valid),
        space=space4svc,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
        trials_save_file='trials_save_file-svm_text'
    )
    print('\n最优参数:')
    print(best)

    np.save(os.path.join(root_abs_path, 'bayes_snapshot/ten_fold_cross_validation-SVM-Text/best_params.npy'), best)