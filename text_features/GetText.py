import requests
import json
import time
import logging
import sys
import os
from datetime import datetime
import hashlib
import csv
from typing import Dict, List, Tuple
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('resualt/eutectic_analysis_ECC.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class QwenEutecticAnalyzer:
    def __init__(self, api_key: str, model: str = "qwen-turbo"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        self.cache_dir = "cache"
        self.cache_file = os.path.join(self.cache_dir, "eutectic_cache.json")
        self.cache = {}
        self.load_cache()

        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "cached": 0,
            "rate_limited": 0,
            "bad_request": 0,
            "unknown_molecules": 0,
            "cache_skipped": 0
        }

        self.input_file = None

    def load_cache(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                logger.info(f"已加载缓存，包含 {len(self.cache)} 条记录")

                error_count = 0
                for key, result in self.cache.items():
                    if result.get("conclusion") == "Error":
                        error_count += 1
                if error_count > 0:
                    logger.info(f"缓存中有 {error_count} 条错误记录")

            except Exception as e:
                logger.warning(f"加载缓存失败: {e}")
                self.cache = {}

    def save_cache(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            logger.info("缓存已保存")
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")

    def get_cache_key(self, mol1: str, mol2: str) -> str:
        sorted_pair = tuple(sorted([mol1, mol2]))
        key_str = f"{sorted_pair[0]}|{sorted_pair[1]}"
        return hashlib.md5(key_str.encode('utf-8')).hexdigest()

    def call_qwen_api(self, prompt: str, max_retries: int = 3) -> Dict:
        if len(prompt) > 3000:
            logger.warning(f"提示词过长 ({len(prompt)} 字符)，进行截断")
            prompt = prompt[:3000]

        payload = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个专业的化学家，专门研究共晶体系和相图分析。请提供准确、专业的分析。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            "parameters": {
                "temperature": 0.2,
                "max_tokens": 500,
                "top_p": 0.8,
                "repetition_penalty": 1.05
            }
        }

        for attempt in range(max_retries):
            try:
                logger.debug(f"发送API请求 (尝试 {attempt + 1}/{max_retries})")
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )

                logger.debug(f"API响应状态码: {response.status_code}")

                if response.status_code == 200:
                    result = response.json()

                    content = self.extract_content_from_response(result)

                    if content:
                        logger.debug(f"API调用成功，返回内容长度: {len(content)}")
                        return {"success": True, "content": content}
                    else:
                        logger.warning(f"API响应成功但无法提取内容")
                        return {"success": False, "error": "无法提取API响应内容"}

                elif response.status_code == 429:
                    self.stats["rate_limited"] += 1
                    wait_time = (2 ** attempt) * 10
                    logger.warning(f"速率限制，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue

                elif response.status_code == 401:
                    logger.error("API密钥无效")
                    return {"success": False, "error": "API密钥无效"}

                elif response.status_code == 403:
                    logger.error("API访问被拒绝")
                    return {"success": False, "error": "API访问被拒绝 - 额度可能已用尽"}

                elif response.status_code == 400:
                    self.stats["bad_request"] += 1
                    logger.error(f"API 400错误")
                    return {"success": False, "error": "API请求参数错误"}

                else:
                    logger.error(f"API调用失败，状态码: {response.status_code}")
                    return {"success": False, "error": f"API错误: {response.status_code}"}

            except requests.exceptions.Timeout:
                logger.warning(f"请求超时，第 {attempt + 1} 次重试...")
                if attempt < max_retries - 1:
                    time.sleep((2 ** attempt) * 5)
                    continue
                else:
                    return {"success": False, "error": "请求超时"}

            except requests.exceptions.RequestException as e:
                logger.error(f"请求异常: {e}")
                if attempt < max_retries - 1:
                    time.sleep((2 ** attempt) * 3)
                    continue
                else:
                    return {"success": False, "error": f"请求异常: {str(e)}"}

        return {"success": False, "error": "API调用失败，达到最大重试次数"}

    def extract_content_from_response(self, response_data: Dict) -> str:
        try:
            if 'output' in response_data and 'text' in response_data['output']:
                return response_data['output']['text'].strip()

            if 'output' in response_data and 'choices' in response_data['output']:
                choices = response_data['output']['choices']
                if isinstance(choices, list) and len(choices) > 0:
                    choice = choices[0]
                    if isinstance(choice, dict):
                        if 'message' in choice and 'content' in choice['message']:
                            return choice['message']['content'].strip()
                        elif 'content' in choice:
                            return choice['content'].strip()

            if 'output' in response_data and 'message' in response_data['output']:
                message = response_data['output']['message']
                if 'content' in message:
                    return message['content'].strip()

            if 'output' in response_data:
                output_str = str(response_data['output'])
                if output_str and len(output_str) > 10:
                    return output_str.strip()

            return ""

        except Exception as e:
            logger.error(f"提取响应内容时出错: {e}")
            return ""

    def analyze_eutectic_pair(self, mol1: str, mol2: str, use_cache: bool = True,
                              skip_error_cache: bool = True) -> Dict:
        cache_key = self.get_cache_key(mol1, mol2)

        if use_cache and cache_key in self.cache:
            cached_result = self.cache[cache_key]

            if skip_error_cache and cached_result.get("conclusion") == "Error":
                self.stats["cache_skipped"] += 1
                logger.info(f"跳过缓存中的错误结果: {mol1} + {mol2}")
            else:
                self.stats["cached"] += 1
                logger.debug(f"使用缓存: {mol1} + {mol2}")
                return cached_result

        logger.info(f"分析: {mol1} + {mol2}")

        prompt = f"""Please analyze the following two molecules: {mol1} and {mol2}, 
                        determine whether they can form a eutectic, and describe the key characteristics of such a eutectic in detail. 
                        The output must include a clear "Yes/No" conclusion and a structured feature description in English only.
                        Please provide your analysis in the following format:Conclusion: [Yes/No]Likelihood: [High/Medium/Low]Key Characteristics:- [Characteristic 1]- [Characteristic 2]- [Characteristic 3]
                        Detailed Analysis:[Detailed explanation]Potential Applications:[Applications if applicable]"""

        api_result = self.call_qwen_api(prompt)

        if not api_result["success"]:
            result = {
                "molecule1": mol1,
                "molecule2": mol2,
                "raw_analysis": f"API错误: {api_result.get('error', '未知错误')}",
                "timestamp": datetime.now().isoformat(),
                "conclusion": "Error",
                "likelihood": "Error",
                "key_characteristics": "",
                "cache_key": cache_key
            }

            if self.is_abbreviation(mol1) or self.is_abbreviation(mol2):
                self.stats["unknown_molecules"] += 1

        else:
            analysis_text = api_result["content"]

            result = {
                "molecule1": mol1,
                "molecule2": mol2,
                "raw_analysis": analysis_text,
                "timestamp": datetime.now().isoformat(),
                "conclusion": self.extract_conclusion(analysis_text),
                "likelihood": self.extract_likelihood(analysis_text),
                "key_characteristics": self.extract_key_characteristics(analysis_text),
                "cache_key": cache_key
            }

        if use_cache:
            self.cache[cache_key] = result

        return result

    def is_abbreviation(self, molecule: str) -> bool:
        if len(molecule) <= 5 and molecule.isupper():
            return True

        abbreviation_patterns = [r'^[A-Z]{3,5}$', r'^[A-Z]+[0-9]+$']
        for pattern in abbreviation_patterns:
            if re.match(pattern, molecule):
                return True

        return False

    def extract_conclusion(self, analysis_text: str) -> str:
        if not analysis_text:
            return "Error"

        analysis_text_lower = analysis_text.lower()

        if "error" in analysis_text_lower or "失败" in analysis_text_lower or "无法" in analysis_text_lower:
            return "Error"

        conclusion_patterns = [
            r"结论\s*[:：]\s*(是|否|不确定|unknown)",
            r"conclusion\s*[:：]\s*(yes|no|unknown|不确定)",
            r"^结论\s*(是|否|不确定)",
            r"^conclusion\s*(yes|no|unknown)",
        ]

        for pattern in conclusion_patterns:
            matches = re.findall(pattern, analysis_text_lower, re.IGNORECASE)
            if matches:
                conclusion = matches[0].lower()
                if conclusion in ["是", "yes"]:
                    return "Yes"
                elif conclusion in ["否", "no"]:
                    return "No"
                elif conclusion in ["不确定", "unknown"]:
                    return "Unknown"

        if "可以形成" in analysis_text or "能形成" in analysis_text or "是共晶" in analysis_text:
            return "Yes"
        elif "不能形成" in analysis_text or "无法形成" in analysis_text or "非共晶" in analysis_text:
            return "No"
        elif "不确定" in analysis_text or "信息不足" in analysis_text or "不明确" in analysis_text:
            return "Unknown"

        return "Unknown"

    def extract_likelihood(self, analysis_text: str) -> str:
        if not analysis_text:
            return "Error"

        conclusion = self.extract_conclusion(analysis_text)
        if conclusion == "Error":
            return "Error"

        analysis_text_lower = analysis_text.lower()

        if "高" in analysis_text_lower and ("可能" in analysis_text_lower or "概率" in analysis_text_lower):
            return "High"
        elif "中" in analysis_text_lower and ("可能" in analysis_text_lower or "概率" in analysis_text_lower):
            return "Medium"
        elif "低" in analysis_text_lower and ("可能" in analysis_text_lower or "概率" in analysis_text_lower):
            return "Low"

        if conclusion == "Yes":
            return "Medium"
        elif conclusion == "No":
            return "Low"
        elif conclusion == "Unknown":
            return "Unknown"

        return "Unknown"

    def extract_key_characteristics(self, analysis_text: str) -> str:
        if not analysis_text:
            return ""

        conclusion = self.extract_conclusion(analysis_text)
        if conclusion == "Error":
            return ""

        lines = analysis_text.split('\n')
        reasons = []

        for line in lines:
            line_lower = line.lower()
            if "结论" in line_lower or "conclusion" in line_lower:
                continue

            if "理由" in line_lower or "reason" in line_lower or "因为" in line or "由于" in line:
                reason_match = re.search(r"[:：]\s*(.+)", line)
                if reason_match:
                    reasons.append(reason_match.group(1).strip())
                else:
                    reasons.append(line.strip())
            elif line.strip() and len(line.strip()) > 10:
                reasons.append(line.strip())

        if reasons:
            return "; ".join(reasons[:2])

        return ""

    def parse_input_file(self, input_file: str) -> List[Tuple[str, str, str, str]]:
        data = []

        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split('\t')
                    if len(parts) >= 4:
                        mol1 = parts[0].strip()
                        mol2 = parts[1].strip()
                        col3 = parts[2].strip()
                        col4 = parts[3].strip()

                        data.append((mol1, mol2, col3, col4))
                    else:
                        logger.warning(f"第 {line_num} 行格式错误: {line}")

        except Exception as e:
            logger.error(f"解析文件失败: {e}")
            raise

        logger.info(f"成功解析 {len(data)} 行数据")
        return data

    def analyze_file(self, input_file: str, output_file: str,
                     batch_size: int = 2, delay: float = 15.0,
                     start_from: int = 0, max_rows: int = None,
                     use_cache: bool = True, skip_error_cache: bool = True):
        self.input_file = input_file

        data = self.parse_input_file(input_file)

        if not data:
            logger.error("没有数据需要处理")
            return []

        if max_rows is not None:
            end_at = min(start_from + max_rows, len(data))
        else:
            end_at = len(data)

        actual_data = data[start_from:end_at]

        logger.info(f"从第 {start_from} 行开始，处理到第 {end_at - 1} 行，共 {len(actual_data)} 行")

        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        temp_output_file = output_file + ".tmp"

        existing_results = {}
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            index = int(row['index'])
                            existing_results[index] = row
                        except (KeyError, ValueError):
                            continue
                logger.info(f"已加载 {len(existing_results)} 条已有结果")
            except Exception as e:
                logger.warning(f"加载已有结果失败: {e}")

        temp_results = {}
        if os.path.exists(temp_output_file):
            try:
                with open(temp_output_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            index = int(row['index'])
                            temp_results[index] = row
                        except (KeyError, ValueError):
                            continue
                logger.info(f"从临时文件加载了 {len(temp_results)} 条中间结果")
            except Exception as e:
                logger.warning(f"加载临时文件失败: {e}")

        all_results = {**existing_results, **temp_results}

        batch_count = (len(actual_data) + batch_size - 1) // batch_size

        for batch_idx in range(batch_count):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(actual_data))
            batch_data = actual_data[start_idx:end_idx]

            logger.info(
                f"处理批次 {batch_idx + 1}/{batch_count}, 行 {start_idx + start_from} 到 {end_idx - 1 + start_from}")

            batch_results = []
            for idx, (mol1, mol2, col3, col4) in enumerate(batch_data):
                global_idx = start_from + start_idx + idx

                if global_idx in all_results:
                    existing_result = all_results[global_idx]
                    if existing_result.get('conclusion') not in ['Error', 'Unknown', 'Pending']:
                        logger.info(f"行 {global_idx}: 已有成功结果，跳过")
                        continue
                    elif skip_error_cache and existing_result.get('conclusion') == 'Error':
                        logger.info(f"行 {global_idx}: 已有错误结果，重新分析")

                self.stats["total_processed"] += 1

                try:
                    analysis_result = self.analyze_eutectic_pair(mol1, mol2, use_cache, skip_error_cache)

                    result_row = {
                        "global_index": global_idx,
                        "molecule1": mol1,
                        "molecule2": mol2,
                        "col3": col3,
                        "col4": col4,
                        "conclusion": analysis_result.get("conclusion", "Unknown"),
                        "likelihood": analysis_result.get("likelihood", "Unknown"),
                        "key_characteristics": analysis_result.get("key_characteristics", ""),
                        "analysis": analysis_result.get("raw_analysis", ""),
                        "timestamp": analysis_result.get("timestamp", ""),
                        "cache_key": analysis_result.get("cache_key", "")
                    }

                    batch_results.append(result_row)
                    self.stats["successful"] += 1

                    logger.info(
                        f"行 {global_idx}: {mol1} + {mol2} -> 结论: {result_row['conclusion']}")

                    time.sleep(2.0)

                except Exception as e:
                    logger.error(f"处理行 {global_idx} 时出错: {e}")
                    self.stats["failed"] += 1

                    error_row = {
                        "global_index": global_idx,
                        "molecule1": mol1,
                        "molecule2": mol2,
                        "col3": col3,
                        "col4": col4,
                        "conclusion": "Error",
                        "likelihood": "Error",
                        "key_characteristics": "",
                        "analysis": f"处理错误: {str(e)}",
                        "timestamp": datetime.now().isoformat(),
                        "cache_key": ""
                    }
                    batch_results.append(error_row)

                    time.sleep(5.0)

            for result in batch_results:
                idx = result.get('global_index', 0)
                all_results[idx] = result

            self._save_temp_results(all_results, temp_output_file, start_from, len(data))

            if batch_idx < batch_count - 1:
                logger.info(f"批次间延迟 {delay} 秒...")
                time.sleep(delay)

        self._save_final_results(all_results, output_file, start_from, len(data))

        if os.path.exists(temp_output_file):
            try:
                os.remove(temp_output_file)
                logger.info("临时文件已删除")
            except Exception as e:
                logger.warning(f"删除临时文件失败: {e}")

        self.save_cache()

        self.print_statistics()

        return list(all_results.values())

    def _save_temp_results(self, results: Dict[int, Dict], temp_file: str, start_from: int, total_rows: int):
        try:
            with open(temp_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    "index", "molecule1", "molecule2", "col3", "col4",
                    "conclusion", "likelihood", "key_characteristics",
                    "analysis_timestamp", "cache_key", "analysis"
                ]

                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for idx in sorted(results.keys()):
                    if idx >= start_from:
                        result = results[idx]
                        analysis_text = str(result.get("analysis", ""))

                        row = {
                            "index": idx,
                            "molecule1": result.get("molecule1", ""),
                            "molecule2": result.get("molecule2", ""),
                            "col3": result.get("col3", ""),
                            "col4": result.get("col4", ""),
                            "conclusion": result.get("conclusion", "Unknown"),
                            "likelihood": result.get("likelihood", "Unknown"),
                            "key_characteristics": result.get("key_characteristics", ""),
                            "analysis_timestamp": result.get("timestamp", ""),
                            "cache_key": result.get("cache_key", ""),
                            "analysis": analysis_text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                        }
                        writer.writerow(row)

            logger.info(f"临时结果已保存到: {temp_file}")

        except Exception as e:
            logger.error(f"保存临时结果失败: {e}")

    def _save_final_results(self, results: Dict[int, Dict], output_file: str, start_from: int, total_rows: int):
        try:
            data = []
            if self.input_file:
                try:
                    with open(self.input_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                parts = line.split('\t')
                                if len(parts) >= 4:
                                    data.append(parts[:4])
                except Exception as e:
                    logger.warning(f"重新读取输入文件失败: {e}")

            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    "index", "molecule1", "molecule2", "col3", "col4",
                    "conclusion", "likelihood", "key_characteristics",
                    "analysis_timestamp", "cache_key", "analysis"
                ]

                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for idx in range(total_rows):
                    if idx in results:
                        result = results[idx]
                    elif data and idx < len(data):
                        mol1, mol2, col3, col4 = data[idx]
                        result = {
                            "molecule1": mol1,
                            "molecule2": mol2,
                            "col3": col3,
                            "col4": col4,
                            "conclusion": "Pending" if idx >= start_from else "Not Processed",
                            "likelihood": "Pending" if idx >= start_from else "Not Processed",
                            "key_characteristics": "",
                            "analysis": "Not processed yet" if idx >= start_from else "Out of processing range",
                            "timestamp": "",
                            "cache_key": ""
                        }
                    else:
                        result = {
                            "molecule1": "",
                            "molecule2": "",
                            "col3": "",
                            "col4": "",
                            "conclusion": "Error",
                            "likelihood": "Error",
                            "key_characteristics": "",
                            "analysis": "Data not found",
                            "timestamp": "",
                            "cache_key": ""
                        }

                    analysis_text = str(result.get("analysis", ""))

                    row = {
                        "index": idx,
                        "molecule1": result.get("molecule1", ""),
                        "molecule2": result.get("molecule2", ""),
                        "col3": result.get("col3", ""),
                        "col4": result.get("col4", ""),
                        "conclusion": result.get("conclusion", "Unknown"),
                        "likelihood": result.get("likelihood", "Unknown"),
                        "key_characteristics": result.get("key_characteristics", ""),
                        "analysis_timestamp": result.get("timestamp", ""),
                        "cache_key": result.get("cache_key", ""),
                        "analysis": analysis_text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                    }
                    writer.writerow(row)

            logger.info(f"最终结果已保存到: {output_file} (共 {total_rows} 行)")

        except Exception as e:
            logger.error(f"保存最终结果失败: {e}")

    def print_statistics(self):
        logger.info("=" * 60)
        logger.info("分析统计信息:")
        logger.info(f"总共处理: {self.stats['total_processed']}")
        logger.info(f"成功: {self.stats['successful']}")
        logger.info(f"失败: {self.stats['failed']}")
        logger.info(f"使用缓存: {self.stats['cached']}")
        logger.info(f"跳过错误缓存: {self.stats['cache_skipped']}")
        logger.info(f"速率限制次数: {self.stats['rate_limited']}")
        logger.info(f"400错误次数: {self.stats['bad_request']}")
        logger.info(f"未知分子缩写: {self.stats['unknown_molecules']}")

        if self.stats['total_processed'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_processed']) * 100
            logger.info(f"成功率: {success_rate:.2f}%")

        logger.info("=" * 60)


def main():
    print("=" * 60)
    print("千问API共晶体系批量分析工具")
    print("=" * 60)

    API_KEY = "sk-324133baaa3645adb409411db3ff0c4a"

    INPUT_FILE = "D:/课业/毕业设计/ccgnet-main_shiyan/ccgnet-main/data/CC_Table/ECC_Table_converted.tab"
    OUTPUT_FILE = "D:/课业/毕业设计/ccgnet-main_shiyan/ccgnet-main/data/AI/ECC_result.csv"

    if not os.path.exists(INPUT_FILE):
        print(f"错误: 输入文件不存在: {INPUT_FILE}")
        sys.exit(1)

    print(f"输入文件: {INPUT_FILE}")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"API密钥: {API_KEY[:8]}...")
    print(f"使用模型: qwen-turbo")
    print("-" * 60)

    print("配置选项:")
    print("1. 从指定行开始处理")
    print("2. 从上次中断处继续 (如果有结果文件)")
    print("3. 从头开始重新处理")
    print("4. 测试模式 (处理前10行)")
    print("5. 自定义范围和行数")

    choice = input("请选择 (1-5, 默认1): ").strip()

    if choice == "2":
        if os.path.exists(OUTPUT_FILE):
            try:
                with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    last_processed = 0
                    for i in range(len(lines) - 1, 0, -1):
                        if i > 0:
                            parts = lines[i].strip().split(',')
                            if len(parts) > 5:
                                conclusion = parts[5] if len(parts) > 5 else ""
                                if conclusion not in ["Pending", "Not Processed", "Error"]:
                                    try:
                                        last_processed = int(parts[0]) + 1
                                        break
                                    except:
                                        continue
                    START_FROM = last_processed
                    print(f"从上次中断处继续: 第 {START_FROM} 行")
            except:
                START_FROM = 0
                print("无法读取已有结果，从头开始")
        else:
            START_FROM = 0
            print("没有找到已有结果，从头开始")

    elif choice == "3":
        START_FROM = 0
        print("从头开始处理所有行")

    elif choice == "4":
        START_FROM = 0
        MAX_ROWS = 10
        print(f"测试模式：只处理前{MAX_ROWS}行")

    elif choice == "5":
        try:
            start = int(input("请输入开始行号 (0-based): "))
            START_FROM = start
            max_rows_input = input("请输入最大处理行数 (直接回车表示无限制): ").strip()
            if max_rows_input:
                MAX_ROWS = int(max_rows_input)
                print(f"从第 {START_FROM} 行开始，最多处理 {MAX_ROWS} 行")
            else:
                MAX_ROWS = None
                print(f"从第 {START_FROM} 行开始，处理到文件结束")
        except ValueError:
            print("输入格式错误，使用默认值")
            START_FROM = 0
            MAX_ROWS = None

    else:
        try:
            start = int(input("请输入开始行号 (0-based): "))
            START_FROM = start
            print(f"从第 {START_FROM} 行开始处理")
        except ValueError:
            print("输入格式错误，使用默认值0")
            START_FROM = 0

    BATCH_SIZE = 2
    DELAY = 15.0

    print("-" * 60)
    print("配置确认:")
    print(f"- 起始行: {START_FROM}")
    if 'MAX_ROWS' in locals() and MAX_ROWS is not None:
        print(f"- 最大行数: {MAX_ROWS}")
    else:
        print(f"- 最大行数: 无限制")
    print(f"- 批次大小: {BATCH_SIZE}")
    print(f"- 批次延迟: {DELAY}秒")

    print("\n缓存处理选项:")
    print("检测到缓存中有大量错误结果，是否跳过缓存中的错误结果？")
    print("建议选择 'y' 以重新分析所有错误结果")
    skip_error = input("跳过错误缓存? (y/N, 默认y): ").strip().lower()
    SKIP_ERROR_CACHE = skip_error != 'n'

    confirm = input("\n确认开始分析? (y/N): ").strip().lower()
    if confirm != 'y':
        print("取消操作")
        sys.exit(0)

    analyzer = QwenEutecticAnalyzer(api_key=API_KEY, model="qwen-turbo")

    try:
        print("\n开始分析...")
        start_time = datetime.now()

        if 'MAX_ROWS' in locals():
            max_rows_param = MAX_ROWS
        else:
            max_rows_param = None

        results = analyzer.analyze_file(
            input_file=INPUT_FILE,
            output_file=OUTPUT_FILE,
            batch_size=BATCH_SIZE,
            delay=DELAY,
            start_from=START_FROM,
            max_rows=max_rows_param,
            use_cache=True,
            skip_error_cache=SKIP_ERROR_CACHE
        )

        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "=" * 60)
        print("分析完成!")
        print(f"总耗时: {duration}")
        print(f"结果已保存到: {OUTPUT_FILE}")
        print("=" * 60)

    except Exception as e:
        print(f"分析过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()