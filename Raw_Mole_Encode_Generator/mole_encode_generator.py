import os
import re
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from openbabel import openbabel

def convert_to_smiles(file_path, file_format):
    try:
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats(file_format, "smi")
        
        mol = openbabel.OBMol()
        obConversion.ReadFile(mol, file_path)  # 读取指定格式的文件
        
        smiles = obConversion.WriteString(mol).strip()
        smiles = str.split(smiles, '\t')[0]
        return smiles
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    descriptor_values = [func(mol) for name, func in Descriptors._descList]
    return np.array(descriptor_values)

def get_valid_filename(filename):
    if '-' in filename:
        valid_name = filename.split('-')[0]
    else:
        valid_name = os.path.splitext(filename)[0]  # 去掉后缀名
    return valid_name

def natural_sort_key(s):
    '''按自然方法排序字符串list'''
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def process_smiles_and_cml(input_folder, output_file):
    smiles_list = []
    source_files = []
    descriptors_list = []

    # 获取并排序输入文件夹中的所有文件
    all_files = sorted(os.listdir(input_folder), key=natural_sort_key)

    for file in all_files:
        file_path = os.path.join(input_folder, file)
        
        if os.path.isfile(file_path):  # 只处理文件，不处理目录
            # 处理SMILES格式的txt文件
            if file.endswith('.txt'):
                with open(file_path, 'r') as f:
                    txt_smiles = f.read().strip().splitlines()
                    if txt_smiles:
                        for smiles in txt_smiles:
                            smiles_list.append(smiles)
                            source_files.append(get_valid_filename(file))
                    else:
                        print(f"TXT file {file_path} is empty, adding an empty line to the output.")
                        smiles_list.append("")  # 为空的txt文件添加一个空行
                        source_files.append(get_valid_filename(file))

            # 处理CML文件
            elif file.endswith('.cml'):
                smiles = convert_to_smiles(file_path, "cml")
                if smiles:
                    smiles_list.append(smiles)
                    source_files.append(get_valid_filename(file))
            
            # 处理MDL MOL文件
            elif file.endswith('.mol'):
                smiles = convert_to_smiles(file_path, "mol")
                if smiles:
                    smiles_list.append(smiles)
                    source_files.append(get_valid_filename(file))

    # 打开输出文件并写入文件名、SMILES和描述符
    with open(output_file, 'w') as f:
        for i, smiles in enumerate(smiles_list):
            source_file = source_files[i]
            print(smiles)
            if smiles:  # 非空行，计算描述符
                descriptors = compute_descriptors(smiles)
                if descriptors is not None:
                    descriptor_str = "\t".join(map(str, descriptors))
                else:
                    # 如果SMILES无法转换为分子，生成全0数组
                    descriptor_length = len(Descriptors._descList)
                    descriptors = np.zeros(descriptor_length)
                    descriptor_str = "\t".join(map(str, descriptors))
            else:  # 空行，占位符
                descriptor_length = len(Descriptors._descList)
                descriptors = np.zeros(descriptor_length)
                descriptor_str = "\t".join(map(str, descriptors))
            
            # 写入文件：文件名、SMILES和描述符，用制表符分隔
            f.write(f"{source_file}\t{smiles}\t{descriptor_str}\n")

    print(f"Processed {len(smiles_list)} SMILES expressions. Output written to {output_file}.")

# def main():

#     # 硬编码输入文件夹路径和输出文件路径
#     input_folder = '/mnt/e/Files/Code/cross_plateform/C_C++/code_new/Data/mole_sets/mole_input'
#     output_file = '/mnt/e/Files/Code/cross_plateform/C_C++/code_new/Data/mole_sets/out/descriptor.txt'

#     process_smiles_and_cml(input_folder, output_file)


# if __name__ == "__main__":
#     main()
