import os
from rdkit import Chem
from rdkit.Chem import Descriptors

def calculate_descriptors(mol):
    """计算分子的描述符。如果mol为None，则返回全零的描述符。"""
    descriptor_names = [desc_name for desc_name, _ in Descriptors.descList]
    if mol is None:
        descriptor_values = [0.0] * len(descriptor_names)
    else:
        descriptor_values = [desc_func(mol) for _, desc_func in Descriptors.descList]
    return descriptor_names, descriptor_values

def process_cml_file(cml_file, output_txt_file):
    """处理CML文件，计算描述符并保存为TXT文件"""
    mol = Chem.MolFromMolFile(cml_file)
    
    if mol is None:
        print(f"无法解析 CML 文件: {cml_file}")

    descriptor_names, descriptor_values = calculate_descriptors(mol)

    with open(output_txt_file, 'w') as txt_file:
        for name, value in zip(descriptor_names, descriptor_values):
            txt_file.write(f"{name}: {value}\n")

    print(f"描述符已成功计算并保存到: {output_txt_file}")

def process_smiles_file(smiles_file, output_txt_file):
    """处理包含SMILES字符串的TXT文件，计算描述符并保存为TXT文件"""
    with open(smiles_file, 'r') as file:
        smiles = file.read().strip()
    
    mol = Chem.MolFromSmiles(smiles) if smiles else None

    if mol is None and smiles:
        print(f"无法解析 SMILES 字符串: {smiles_file}")

    descriptor_names, descriptor_values = calculate_descriptors(mol)

    with open(output_txt_file, 'w') as txt_file:
        for name, value in zip(descriptor_names, descriptor_values):
            txt_file.write(f"{name}: {value}\n")

    print(f"描述符已成功计算并保存到: {output_txt_file}")

def process_folder(input_folder, output_folder):
    """处理文件夹中的所有CML文件和包含SMILES的TXT文件，并将结果保存到输出文件夹"""
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        output_txt_file = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_descriptors.txt")

        if filename.endswith(".cml"):
            process_cml_file(file_path, output_txt_file)
        elif filename.endswith(".txt"):
            process_smiles_file(file_path, output_txt_file)
        else:
            print(f"跳过不支持的文件: {filename}")

def main():
    """主函数，将分子结构转为分子指纹（描述形式）"""
    input_folder = '/path/to/your/input_folder'
    output_folder = '/path/to/your/output_folder'
    process_folder(input_folder, output_folder)

if __name__ == "__main__":
    main()
