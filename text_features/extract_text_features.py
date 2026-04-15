import pandas as pd
import numpy as np
import torch
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

CONFIG = {
    'text_csv_path': 'D:/课业/毕业设计/ccgnet-main_shiyan/ccgnet-main/data/AI/ECC_result.csv',
    'cc_table_path': 'D:/课业/毕业设计/ccgnet-main_shiyan/ccgnet-main/data/CC_Table/ECC_Table.tab',
    'output_dir': 'D:/课业/毕业设计/ccgnet-main_shiyan/ccgnet-main/data/processed_features/',
    'model_name': 'D:/software/scibert',
    'max_length': 320,
    'batch_size': 8,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

def create_output_dir():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    print(f"[信息] 输出目录: {CONFIG['output_dir']}")

def load_and_preprocess_text_data(csv_path):
    print(f"[步骤1] 正在加载文本数据: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
        print(f"[信息] 成功读取CSV文件，形状: {df.shape}")
    except Exception as e:
        print(f"[错误] 无法读取CSV文件: {e}")
        return None

    print(f"[信息] 原始数据形状: {df.shape}")
    print(f"[信息] 列名: {list(df.columns)}")

    initial_count = len(df)
    df = df[df['conclusion'] != 'Error'].copy()
    filtered_count = initial_count - len(df)
    print(f"[信息] 过滤掉 {filtered_count} 条错误数据（conclusion='Error'）")

    required_cols = ['col4', 'conclusion', 'key_characteristics', 'analysis']
    for col in required_cols:
        if col not in df.columns:
            print(f"[错误] 缺少必要列: {col}")
            return None

    for col in ['conclusion', 'key_characteristics', 'analysis']:
        df[col] = df[col].fillna('').astype(str).str.strip()

    print("[信息] 创建组合文本...")
    df['combined_text'] = (
        "结论: " + df['conclusion'] + ". " +
        "关键特征: " + df['key_characteristics'] + ". " +
        "详细分析: " + df['analysis'].str[:500]
    )

    df['col4_clean'] = df['col4'].astype(str).str.strip()

    print(f"[信息] 预处理后文本数据形状: {df.shape}")
    print(f"[信息] col4示例: {df['col4_clean'].iloc[:3].tolist()}")
    print(f"[信息] 组合文本示例: {df['combined_text'].iloc[0][:100]}...")

    return df

def load_cc_table(cc_table_path):
    print(f"\n[步骤2] 正在加载CC_Table: {cc_table_path}")

    try:
        df_cc = pd.read_csv(
            cc_table_path,
            sep='\t',
            header=None,
            names=['mol1', 'mol2', 'label', 'pair_id']
        )
        print(f"[信息] 成功读取CC_Table，形状: {df_cc.shape}")
    except Exception as e:
        print(f"[错误] 读取CC_Table失败: {e}")
        return None

    df_cc['pair_id_clean'] = df_cc['pair_id'].astype(str).str.strip()

    print(f"[信息] CC_Table示例 (前5行):")
    print(df_cc.head())
    print(f"[信息] pair_id示例: {df_cc['pair_id_clean'].iloc[:3].tolist()}")

    return df_cc

def align_text_with_pairs_simple(df_text, df_cc):
    print(f"\n[步骤3] 对齐文本数据与分子对 (简化版)")
    print(f"[信息] 文本数据行数: {len(df_text)}")
    print(f"[信息] CC_Table行数: {len(df_cc)}")

    df_cc = df_cc.copy()
    df_cc['combined_text'] = None
    df_cc['is_missing_text'] = True

    text_dict = {}
    for idx, row in df_text.iterrows():
        col4_value = str(row['col4_clean']).strip()
        if col4_value and col4_value != 'nan':
            text_dict[col4_value] = row['combined_text']

    print(f"[信息] 文本字典大小: {len(text_dict)}")
    print(f"[信息] 示例键: {list(text_dict.keys())[:5]}")

    match_count = 0
    for idx, row in df_cc.iterrows():
        pair_id = str(row['pair_id_clean']).strip()

        if pair_id in text_dict:
            df_cc.at[idx, 'combined_text'] = text_dict[pair_id]
            df_cc.at[idx, 'is_missing_text'] = False
            match_count += 1

    print(f"[信息] 匹配成功: {match_count}/{len(df_cc)} ({match_count/len(df_cc)*100:.1f}%)")

    default_text = "共晶形成可能性分析: 数据缺失。需要进一步实验验证。"
    df_cc['combined_text'] = df_cc['combined_text'].fillna(default_text)

    missing_count = df_cc['is_missing_text'].sum()
    print(f"[信息] 对齐完成，最终数据形状: {df_cc.shape}")
    print(f"[信息] 匹配统计:")
    print(f"  总分子对: {len(df_cc)}")
    print(f"  有文本特征的: {match_count}")
    print(f"  缺失文本的: {missing_count}")

    return df_cc

def clean_text_data(aligned_df):
    print(f"\n[数据清洗] 开始清洗文本数据...")

    print(f"  清洗前 combined_text 类型分布:")
    type_counts = aligned_df['combined_text'].apply(lambda x: type(x).__name__).value_counts()
    for type_name, count in type_counts.items():
        print(f"    {type_name}: {count}")

    original_len = len(aligned_df)
    aligned_df['combined_text'] = aligned_df['combined_text'].astype(str)
    print(f"  已将所有文本转换为字符串类型")

    default_text = "共晶形成可能性分析: 数据缺失。需要进一步实验验证。"

    invalid_patterns = ['None', 'nan', 'NaN', '<NA>', 'null', 'Null', 'NULL', 'NoneType', 'NaT']

    for pattern in invalid_patterns:
        exact_mask = aligned_df['combined_text'].str.lower() == pattern.lower()
        partial_mask = aligned_df['combined_text'].str.contains(pattern, case=False, na=False)

        combined_mask = exact_mask | partial_mask
        if combined_mask.sum() > 0:
            aligned_df.loc[combined_mask, 'combined_text'] = default_text
            print(f"  替换了 {combined_mask.sum()} 个包含 '{pattern}' 的文本")

    empty_mask = aligned_df['combined_text'].str.strip() == ''
    if empty_mask.sum() > 0:
        aligned_df.loc[empty_mask, 'combined_text'] = default_text
        print(f"  替换了 {empty_mask.sum()} 个空文本")

    print(f"  清洗后 combined_text 类型分布:")
    type_counts_after = aligned_df['combined_text'].apply(lambda x: type(x).__name__).value_counts()
    for type_name, count in type_counts_after.items():
        print(f"    {type_name}: {count}")

    print(f"\n  清洗后文本示例 (前3个):")
    for i in range(min(3, len(aligned_df))):
        text = aligned_df.iloc[i]['combined_text']
        print(f"    样本 {i}: {repr(text[:80])}...")

    print(f"  数据清洗完成，总样本数: {len(aligned_df)}")
    return aligned_df

class SciBERTFeatureExtractor:

    def __init__(self, model_name=CONFIG['model_name'], device=CONFIG['device']):
        print(f"\n[步骤4] 初始化SciBERT模型: {model_name}")
        print(f"[信息] 使用设备: {device}")

        self.device = device

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.model.eval()

            dummy_input = self.tokenizer("test", return_tensors="pt", truncation=True, padding=True, max_length=50).to(device)
            with torch.no_grad():
                dummy_output = self.model(**dummy_input)
            self.feature_dim = dummy_output.last_hidden_state.size(-1)

            print(f"[信息] SciBERT模型加载成功，特征维度: {self.feature_dim}")
        except Exception as e:
            print(f"[错误] 加载SciBERT模型失败: {e}")
            raise

    def extract_features(self, texts, batch_size=CONFIG['batch_size']):
        if not isinstance(texts, (list, pd.Series)):
            texts = list(texts)

        print(f"[信息] 开始提取文本特征，共 {len(texts)} 条文本")

        if len(texts) > 0:
            print(f"[调试] 第一批数据检查 (前{min(2, len(texts))}条):")
            for i in range(min(2, len(texts))):
                text = texts[i]
                print(f"  文本 {i}: 类型={type(text).__name__}, 长度={len(str(text))}, 预览={repr(str(text)[:80])}")

        all_features = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_size_actual = len(batch_texts)

            try:
                if i == 0:
                    print(f"[调试] 第一批 batch_texts 详细信息:")
                    for j, text in enumerate(batch_texts):
                        print(f"  索引 {j}: 类型={type(text).__name__}, 值={repr(str(text)[:50])}")

                batch_texts_clean = [str(text) for text in batch_texts]

                encoded = self.tokenizer(
                    batch_texts_clean,
                    truncation=True,
                    padding=True,
                    max_length=CONFIG['max_length'],
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**encoded)
                    features = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                all_features.append(features)

            except Exception as e:
                print(f"\n[错误] 第 {i}-{i+batch_size_actual} 批处理失败!")
                print(f"  错误类型: {type(e).__name__}")
                print(f"  错误信息: {e}")
                print(f"  批处理大小: {batch_size_actual}")

                print(f"  失败批次内容示例 (前2条):")
                for j in range(min(2, len(batch_texts))):
                    print(f"    文本 {j}: {repr(str(batch_texts[j])[:100])}")

                raise ValueError(f"特征提取失败: {e}")

            processed = min(i + batch_size, len(texts))
            if i % (batch_size * 10) == 0 or processed == len(texts):
                print(f"  进度: {processed}/{len(texts)} ({processed/len(texts)*100:.1f}%)")

        if all_features:
            features_array = np.vstack(all_features)
            if np.allclose(features_array, 0):
                print(f"\n[警告] 特征矩阵似乎全为零!")
                print(f"  特征均值: {np.mean(features_array)}")
                print(f"  特征标准差: {np.std(features_array)}")
        else:
            features_array = np.array([])

        print(f"[信息] 特征提取完成，特征矩阵形状: {features_array.shape}")
        return features_array

def save_results(aligned_df, features, output_dir):
    print(f"\n[步骤5] 保存结果到 {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    feature_file = os.path.join(output_dir, 'text_features_ECC.npy')
    np.save(feature_file, features)
    print(f"[信息] 特征矩阵保存至: {feature_file}")
    print(f"[信息] 特征矩阵形状: {features.shape}")

    if features.size > 0:
        print(f"[验证] 特征矩阵统计:")
        print(f"  特征均值: {np.mean(features):.6f}")
        print(f"  特征标准差: {np.std(features):.6f}")
        print(f"  特征绝对值均值: {np.mean(np.abs(features)):.6f}")
        zero_count = np.all(features == 0, axis=1).sum()
        print(f"  全零样本数: {zero_count}/{len(features)} ({(zero_count/len(features))*100:.1f}%)")

    metadata_file = os.path.join(output_dir, 'feature_metadata_ECC.csv')

    metadata_df = aligned_df.copy()
    metadata_df['feature_index'] = range(len(metadata_df))

    metadata_df.to_csv(metadata_file, index=False)
    print(f"[信息] 元数据保存至: {metadata_file}")

    mapping_df = metadata_df[['pair_id', 'feature_index']].copy()
    mapping_file = os.path.join(output_dir, 'feature_mapping_ECC.csv')
    mapping_df.to_csv(mapping_file, index=False)
    print(f"[信息] 特征映射保存至: {mapping_file}")

    stats = {
        'total_samples': len(aligned_df),
        'feature_dim': features.shape[1],
        'extraction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'matched_count': int((~aligned_df['is_missing_text']).sum()),
        'missing_count': int(aligned_df['is_missing_text'].sum()),
        'match_rate': f"{(~aligned_df['is_missing_text']).sum()/len(aligned_df)*100:.1f}%",
        'feature_mean': float(np.mean(features)) if features.size > 0 else 0,
        'feature_std': float(np.std(features)) if features.size > 0 else 0,
        'zero_samples': int(np.all(features == 0, axis=1).sum()) if features.size > 0 else 0
    }

    import json
    stats_file = os.path.join(output_dir, 'extraction_stats_ECC.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[信息] 统计信息保存至: {stats_file}")

    return {
        'feature_file': feature_file,
        'metadata_file': metadata_file,
        'mapping_file': mapping_file,
        'stats_file': stats_file
    }

def main():
    print("=" * 60)
    print("SciBERT文本特征提取流程 (修复版)")
    print("=" * 60)

    try:
        create_output_dir()

        df_text = load_and_preprocess_text_data(CONFIG['text_csv_path'])
        if df_text is None:
            return

        df_cc = load_cc_table(CONFIG['cc_table_path'])
        if df_cc is None:
            return

        aligned_df = align_text_with_pairs_simple(df_text, df_cc)

        aligned_df = clean_text_data(aligned_df)

        extractor = SciBERTFeatureExtractor()
        features = extractor.extract_features(aligned_df['combined_text'])

        saved_files = save_results(aligned_df, features, CONFIG['output_dir'])

        print("\n" + "=" * 60)
        print("流程完成!")
        print("=" * 60)

        print(f"\n[最终统计]")
        print(f"  总分子对: {len(aligned_df)}")
        print(f"  匹配的文本数量: {(~aligned_df['is_missing_text']).sum()}")
        print(f"  匹配率: {(~aligned_df['is_missing_text']).sum()/len(aligned_df)*100:.1f}%")
        print(f"  特征维度: {features.shape[1]}")
        print(f"  特征均值: {np.mean(features):.6f}")
        print(f"  特征标准差: {np.std(features):.6f}")
        print(f"  输出目录: {CONFIG['output_dir']}")

        return aligned_df, features, saved_files

    except Exception as e:
        print(f"\n[错误] 流程失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    aligned_df, features, saved_files = main()

    if aligned_df is not None:
        print("\n" + "=" * 60)
        print("下一步：")
        print("1. 检查输出目录中的文件")
        print("2. 根据 usage_example.py 修改 Dataset 类")
        print("3. 开始训练融合文本特征的新模型")
        print("=" * 60)