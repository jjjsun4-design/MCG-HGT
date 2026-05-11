# =============================================================================
#           MolMCL 分子嵌入生成脚本 (最终完整版 v4)
#
# 目的: 从CSV或TXT文件中读取SMILES，使用预训练的MolMCL模型批量生成嵌入向量，
#       并将结果以高效的 NumPy `.npz` 格式保存。
# =============================================================================

import torch
from rdkit import Chem
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
import torch_geometric.transforms as T
import pandas as pd
import numpy as np
import warnings
import sys
import os

# --- 忽略来自RDKit的非关键警告 ---
from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')
warnings.filterwarnings("ignore", category=UserWarning)

# --- 从 MolMCL 项目中导入必要的模块 ---
try:
    from molmcl.finetune.model import GNNPredictor
    from molmcl.utils.data import mol_to_graph_data_obj_super_rich
except ImportError:
    print("错误: 无法导入 'molmcl' 模块。")
    print("请确保此脚本位于MolMCL项目的根目录下，或者该项目已经正确安装。")
    sys.exit(1) # !! 修正点: 补全了括号 !!

def generate_embeddings(smiles_list, checkpoint_path, device='cpu', node_ids=None):  # ★ 新增: node_ids
    """
    接收一个SMILES列表，使用预训练的MolMCL模型为其生成嵌入向量。
    """
    
    # ==================== 步骤一: 构建模型骨架并加载预训练权重 ====================
    print("--> 步骤一: 正在加载预训练模型...")
    model_config = {
        'num_layer': 5, 'emb_dim': 300, 'heads': 6, 'use_prompt': True,
        'backbone': 'gps', 'dropout_ratio': 0, 'temperature': 1.0,
        'normalize': False, 'layernorm': False
    }
    model = GNNPredictor(
        num_layer=model_config['num_layer'], emb_dim=model_config['emb_dim'], num_tasks=1,
        atom_feat_dim=170, bond_feat_dim=14, use_prompt=model_config['use_prompt'],
        drop_ratio=model_config['dropout_ratio'], temperature=model_config['temperature'],
        model_head=model_config['heads'], layer_norm_out=model_config['layernorm'],
        backbone=model_config['backbone']
    )
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except FileNotFoundError:
        print(f"错误: 预训练模型检查点文件未找到 -> {checkpoint_path}")
        return None, None

    model.load_state_dict(checkpoint['wrapper'], strict=False)
    model.to(device)
    model.eval()
    print(f"模型已成功加载到 {device} 设备并设置为评估模式！")

    # ==================== 步骤二: 准备输入分子批次 ====================
    print("\n--> 步骤二: 正在处理输入的SMILES...")
    data_list, valid_smiles_input = [], []
    valid_node_ids = []  # ★ 新增: 记录与有效 SMILES 对齐的 Node ID
    pe_transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
    
    # ★ 新增: 同步遍历 node_ids（若提供）
    if node_ids is None:
        iterable = enumerate(smiles_list)
    else:
        iterable = enumerate(zip(smiles_list, node_ids))

    for idx, item in iterable:
        if node_ids is None:
            smiles = item
            node_id = None
        else:
            smiles, node_id = item

        mol = Chem.MolFromSmiles(smiles)  # RDKit 解析 SMILES（官方 API）:contentReference[oaicite:1]{index=1}
        if mol:
            valid_smiles_input.append(smiles)
            if node_id is not None:
                valid_node_ids.append(str(node_id))  # 与有效 SMILES 一一对应
            data = mol_to_graph_data_obj_super_rich(mol)
            data_with_pe = pe_transform(data)
            data_list.append(data_with_pe)
        else:
            print(f"警告: SMILES '{smiles}' 无效，已跳过。")
    
    if not data_list:
        print("错误：文件中所有SMILES均无效或无法处理。")
        return None, None

    batch = Batch.from_data_list(data_list).to(device)
    print(f"已成功处理 {len(data_list)} 个有效SMILES并构建批次。")

    # ==================== 步骤三: 执行推理并提取嵌入 ====================
    print("\n--> 步骤三: 正在批量生成嵌入向量...")
    with torch.no_grad():
        if model.backbone == 'gps':
            _, node_reps = model.gnn(batch.x, batch.pe, batch.edge_index, batch.edge_attr, batch.batch)
        else:
            _, node_reps = model.gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
        # PyG 将变长图对齐为 [B, Nmax, D] 并返回 mask（官方文档）:contentReference[oaicite:2]{index=2}
        batch_x, batch_mask = to_dense_batch(node_reps, batch.batch)
        
        channel_reps = []
        for i in range(len(model.prompt_token)):
            h_g, _, _ = model.aggrs[i](batch_x, batch_mask)
            channel_reps.append(h_g)
        
        channel_embeddings = torch.stack(channel_reps, dim=1)
        prompt_weights = model.get_prompt_weight(act='softmax').squeeze()
        composite_embeddings = torch.einsum('bcd, c -> bd', channel_embeddings, prompt_weights)
    print("嵌入生成完毕！")
    
    # 将Tensor转换为Numpy数组以便返回
    composite_embeddings_np = composite_embeddings.cpu().numpy()
    channel_embeddings_np = channel_embeddings.cpu().numpy()

    # ★ 保留你原来的 .pt 导出行为（注意：依赖外层定义的 output_file）
    try:
        global output_file  # 使用 main 中的同名变量
        if 'output_file' in globals() and output_file:
            torch.save(composite_embeddings, output_file)
    except Exception as e:
        print(f"提示：保存 .pt 失败（已忽略，不影响 .npz）：{e}")

    # ★ 新增: 在返回的字典中加入 node_id
    all_embeddings = {
        'smiles': np.array(valid_smiles_input, dtype=object),
        'composite_embedding': composite_embeddings_np,
        'mcd_embedding': channel_embeddings_np[:, 0, :],
        'scd_embedding': channel_embeddings_np[:, 1, :],
        'cp_embedding': channel_embeddings_np[:, 2, :],
        'node_id': np.array(valid_node_ids, dtype=object) if valid_node_ids else np.array([], dtype=object)  # 新增
    }
        
    return all_embeddings

def main():
    """
    主函数，负责处理文件IO和调用嵌入生成函数。
    """
    # ============================== 用户配置区 ===============================
    # 请在这里修改你的文件路径和参数
    INPUT_FILE = "data/HIT/1237ingredients.xlsx" # !! 修改为你的输入文件路径 !!
    OUTPUT_FILE = "my_embeddings.npz"                          # !! 修改为你想要的输出文件名 !!
    global output_file
    output_file = "my_embeddings_ligand.pt"
    SMILES_COLUMN = 'Smiles'                                   # !! (仅用于CSV/XLSX) 包含SMILES的列名 !!
    NODE_ID_COLUMN = 'Node ID'                                  # ★ 新增: Node ID 列名
    CHECKPOINT_FILE = './checkpoint/zinc-gps_best.pt'
    # ========================================================================

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 从文件读取SMILES ---
    print(f"--- 开始处理文件: {INPUT_FILE} ---")
    try:
        if INPUT_FILE.lower().endswith(('.csv', '.txt')):
            df = pd.read_csv(INPUT_FILE)
        elif INPUT_FILE.lower().endswith(('.xlsx', '.xls')):
             df = pd.read_excel(INPUT_FILE)
        else:
            raise ValueError("不支持的文件格式，请输入.csv, .txt, .xlsx, 或 .xls 文件。")
        
        # ★ 新增: 用同一个 mask 对齐 SMILES 与 Node ID，保证一一对应
        mask = df[SMILES_COLUMN].notna() & (df[SMILES_COLUMN].astype(str).str.strip() != "")
        smiles_list = df.loc[mask, SMILES_COLUMN].astype(str).tolist()
        node_ids_list = df.loc[mask, NODE_ID_COLUMN].astype(str).tolist()
    except Exception as e:
        print(f"错误: 读取输入文件失败 - {e}")
        return

    # --- 调用核心函数生成嵌入 ---
    embedding_results = generate_embeddings(smiles_list, CHECKPOINT_FILE, device=DEVICE, node_ids=node_ids_list)

    # --- 保存结果 ---
    if embedding_results:
        print(f"\n--> 正在将嵌入保存到 '{OUTPUT_FILE}'...")
        # NumPy 官方建议用关键字参数命名各数组（避免 arr_0/arr_1）:contentReference[oaicite:3]{index=3}
        np.savez_compressed(OUTPUT_FILE, **embedding_results)
        print("保存成功！")
        
        # --- 打印结果预览 ---
        print("\n======================== 嵌入结果预览 ========================")
        composite_emb = embedding_results['composite_embedding']
        print(f"成功为 {len(embedding_results['smiles'])} 个分子生成了嵌入。")
        # ★ 新增: 打印 Node ID 对齐情况
        if 'node_id' in embedding_results and len(embedding_results['node_id']) == len(embedding_results['smiles']):
            print("Node ID 与 SMILES 数量一致，已对齐保存。")
        print(f"复合嵌入矩阵的形状: {composite_emb.shape}")
        print(f"第一个分子的嵌入预览 (前10维): {composite_emb[0, :10].round(4)}")

if __name__ == '__main__':
    main()
