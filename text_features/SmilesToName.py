import requests
import time
import urllib.parse

ENERGETIC_COMPOUNDS_MAP = {
    'CL-20': '159266',
    'HMX': '31294',
    'RDX': '92112',
    'TNT': '8376',
    'BTF': '123789',
    'PETN': '6518',
    'TATB': '9926567',
    'NTO': '62586',
    'FOX-7': '3035345',
    'LLM-105': '11772658',
    'DAAF': '11763112',
    'HNS': '30835',
    'NQ': '6226',
    'TNGU': '11748040',
    'DNTF': '11740059',
    'PA': '6683',
    'BTO': '3035109',
    'ATZ': '10467',
    'TPPO': '7586',
    'GTA': '5986',
    'DO': '3129',
    'BL': '7302',
    'HMPA': '7912',
    'NMP': '31359',
    'BQ': '4650',
    'NAQ': '4650',
    'CPL': '10467',
    'DNP': '14932',
    'MNO': '701',
    'DMI': '66635',
    'FA': '284',
    'PDA': '1049',
    '1-BN': '111173',
    'DMB': '11450',
    'DMDBT': '138390',
    'TT': '11434',
    'CIM': '5942',
    'CE': '23974',
    '3,5-DNP': '14932',
    '3-AT': '10467',
    '4-AT': '126954',
    'ANAT': '450419',
    'AT': '9257',
    'Ant': '8418',
    'Benzene-1,2-diamine': '7243',
    'DMF': '6228',
    'DNB': '7452',
    'DNT': '38413',
    'EDNA': '61250',
    'NN': '9321',
    'Nap': '931',
    'Nc1ccccc1N': '7812',
    'PDCA': '1017',
    'Per': '9143',
    'Phenothiazine': '7108',
    'Py': '1049',
    'Pyr': '8027',
    'TNA': '6995',
    'TNAZ': '123124',
    'TNB': '7433',
    'phenanthrene': '995',
    'xylene': '7237',
}

def is_likely_smiles(identifier):
    if any(c in identifier for c in '()[]=+#@'):
        return True
    if not identifier.isdigit() and any(c.isdigit() for c in identifier):
        return True
    if any(c.islower() for c in identifier):
        return True
    return False


def get_compound_name(identifier):
    if identifier in ENERGETIC_COMPOUNDS_MAP:
        cid = ENERGETIC_COMPOUNDS_MAP[identifier]
        if cid:
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IUPACName/JSON"
        else:
            encoded_name = urllib.parse.quote(identifier, safe='')
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/property/IUPACName/JSON"
    elif is_likely_smiles(identifier):
        encoded_id = urllib.parse.quote(identifier, safe='')
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded_id}/property/IUPACName/JSON"
    else:
        if identifier.isdigit():
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{identifier}/property/IUPACName/JSON"
        else:
            encoded_id = urllib.parse.quote(identifier, safe='')
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded_id}/property/IUPACName/JSON"

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            name = data['PropertyTable']['Properties'][0]['IUPACName']
            return name
        else:
            if not identifier.isdigit() and identifier not in ENERGETIC_COMPOUNDS_MAP:
                encoded_name = urllib.parse.quote(identifier, safe='')
                name_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/property/IUPACName/JSON"

                response = requests.get(name_url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    name = data['PropertyTable']['Properties'][0]['IUPACName']
                    return name

            print(f"警告: 无法获取 {identifier} 的名称 (状态码: {response.status_code})")
            return identifier
    except Exception as e:
        print(f"错误: 处理 {identifier} 时出错: {e}")
        return identifier


def batch_find_cids(unknown_compounds):
    import urllib.parse

    print("开始批量查找未知化合物的CID...")
    results = {}

    for compound in unknown_compounds:
        if compound in ENERGETIC_COMPOUNDS_MAP and ENERGETIC_COMPOUNDS_MAP[compound] is not None:
            continue

        print(f"查找 {compound}...")

        encoded_name = urllib.parse.quote(compound, safe='')
        search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/cids/JSON"

        try:
            response = requests.get(search_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'IdentifierList' in data and 'CID' in data['IdentifierList']:
                    cids = data['IdentifierList']['CID']
                    if cids:
                        results[compound] = cids[0]
                        print(f"  {compound}: CID = {cids[0]}")
                    else:
                        results[compound] = None
                        print(f"  {compound}: 未找到CID")
                else:
                    results[compound] = None
                    print(f"  {compound}: 未找到CID")
            else:
                results[compound] = None
                print(f"  {compound}: 查询失败 (状态码: {response.status_code})")
        except Exception as e:
            results[compound] = None
            print(f"  {compound}: 查询错误 - {e}")

        time.sleep(0.2)

    print("\n查找结果:")
    for compound, cid in results.items():
        if cid:
            print(f"    '{compound}': '{cid}',")
        else:
            print(f"    '{compound}': None,")

    return results
def simple_convert(input_file, output_file, error_log_file="errors.log"):
    error_log = []

    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) < 4:
                print(f"警告: 第 {line_num} 行格式错误，跳过")
                error_log.append(f"第 {line_num} 行: 格式错误 - {line}")
                continue

            name1 = get_compound_name(parts[0])
            name2 = get_compound_name(parts[1])

            if parts[0] in ENERGETIC_COMPOUNDS_MAP and ENERGETIC_COMPOUNDS_MAP[parts[0]] is None:
                error_log.append(f"第 {line_num} 行: {parts[0]} 需要人工确认CID")
            if parts[1] in ENERGETIC_COMPOUNDS_MAP and ENERGETIC_COMPOUNDS_MAP[parts[1]] is None:
                error_log.append(f"第 {line_num} 行: {parts[1]} 需要人工确认CID")

            f_out.write(f"{name1}\t{name2}\t{parts[2]}\t{parts[3]}\n")

            print(f"处理第 {line_num} 行: {parts[0][:20]}... -> {name1[:30]}..., {parts[1][:20]}... -> {name2[:30]}...")

            time.sleep(0.1)

    if error_log:
        with open(error_log_file, 'w') as f_err:
            f_err.write("需要人工确认的化合物:\n")
            for error in error_log:
                f_err.write(error + "\n")
        print(f"\n已生成错误日志到 {error_log_file}")


def find_unknown_compounds(input_file):
    unknown_compounds = set()

    with open(input_file, 'r') as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) >= 2:
                for i in range(2):
                    compound = parts[i]
                    if (not compound.isdigit() and
                            compound not in ENERGETIC_COMPOUNDS_MAP and
                            not any(c in compound for c in '()[]=+#@')):
                        unknown_compounds.add(compound)

    return sorted(list(unknown_compounds))


if __name__ == "__main__":
    input_file = "D:/课业/毕业设计/ccgnet-main_shiyan/ccgnet-main/data/CC_Table/ECC_Table.tab"
    output_file = "D:/课业/毕业设计/ccgnet-main_shiyan/ccgnet-main/data/CC_Table/ECC_Table_converted.tab"

    print("正在分析输入文件中的化合物...")
    unknown = find_unknown_compounds(input_file)

    if unknown:
        print(f"发现 {len(unknown)} 个未知化合物:")
        for i, comp in enumerate(unknown[:20], 1):
            print(f"  {i}. {comp}")
        if len(unknown) > 20:
            print(f"  ... 还有 {len(unknown) - 20} 个")

        answer = input("\n是否要批量查找这些化合物的CID? (y/n): ")
        if answer.lower() == 'y':
            batch_find_cids(unknown)

    print("\n开始转换...")
    simple_convert(input_file, output_file)

    print(f"\n转换完成！结果保存在: {output_file}")