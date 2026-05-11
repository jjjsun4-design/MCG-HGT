#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从 BindingDB/BioSNAP 风格的 train/val/test CSV
（列：DrugBank ID, Gene, Label, SMILES, Target Sequence）
构建论文使用的图数据 + 清洗后的带 index 的数据集。

相似度按本文实验设置定义：

1) 成分相似度（ingredient_similarity.txt）
   - 对每个成分的 SMILES 计算 ECFP4 (Morgan, radius=2, nBits=1024) 指纹
   - Tanimoto(Fi, Fj) = |Fi ∩ Fj| / |Fi ∪ Fj|
   - 只保留 Tanimoto >= 0.5 的成分对

2) 靶点相似度（target_similarity.txt）
   - 对任意两条蛋白序列 S_i, S_j，用 Smith-Waterman (SW) 算法计算局部比对得分
   - 自比对：SW(S_i, S_i)、SW(S_j, S_j)
   - 归一化相似度：
        SW_normalized(S_i, S_j) = SW(S_i, S_j) / sqrt( SW(S_i, S_i) * SW(S_j, S_j) )
   - 只保留 SW_normalized >= 0.3 的靶点对
   - 输出格式： target_id_i  target_id_j  SW_normalized

3) 交互边（edges.txt）
   - 所有 Label == 1 的样本，按 ingredient_id, target_id 输出一条边

4) 节点表（方便对照 ID）
   - ingredient_nodes.csv: ingredient_id, DrugBank ID, SMILES
   - target_nodes.csv:     target_id,    Gene,        Target Sequence

5) 清洗后的样本（带 index，对应 txt 里的 ID）
   - train_clean.csv / val_clean.csv / test_clean.csv
       列：ingredient_id, target_id, DrugBank ID, Gene, SMILES, Target Sequence, Label
   - train_idx.txt / val_idx.txt / test_idx.txt
       每行：ingredient_id target_id Label
"""

import os
import math
import pandas as pd

# ================= 超参数（直接对应文章设定） =================

# 成分 Tanimoto 相似度阈值（文中：similarity > 0.5）
ING_TANIMOTO_THRESHOLD = 0.5

# 蛋白 Smith-Waterman 归一化相似度阈值（文中：score > 0.3）
TARGET_SW_THRESHOLD = 0.3


# ================= 工具函数：读入并清洗 train/val/test =================

def load_split(path: str) -> pd.DataFrame:
    """
    读入某个 split（train/val/test），丢掉多余列，只保留核心列：
      DrugBank ID, Gene, Label, SMILES, Target Sequence
    """
    df = pd.read_csv(path)
    use_cols = ["DrugBank ID", "Gene", "Label", "SMILES", "Target Sequence"]

    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺少必要列: {missing}")

    df = df[use_cols].copy()
    # 丢掉缺失值
    df = df.dropna(subset=["DrugBank ID", "Gene", "SMILES", "Target Sequence", "Label"])
    # Label -> int(0/1)
    df["Label"] = df["Label"].astype(float).round().astype(int)
    return df


# ================= 建立统一的 ID 映射：DrugBankID/Gene → integer =================

def build_index_maps(all_df: pd.DataFrame):
    """
    从所有样本中建立 ingredient_id / target_id 映射。

    返回：
      drug2idx:   {DrugBank ID -> ingredient_id}
      target2idx: {Gene -> target_id}
      ingredient_df: [ingredient_id, DrugBank ID, SMILES]
      target_df:     [target_id, Gene, Target Sequence]
    """
    # 唯一成分 / 靶点
    unique_drugs = all_df[["DrugBank ID", "SMILES"]].drop_duplicates("DrugBank ID")
    unique_targets = all_df[["Gene", "Target Sequence"]].drop_duplicates("Gene")

    # 固定排序，保证编号稳定
    unique_drugs = unique_drugs.sort_values("DrugBank ID").reset_index(drop=True)
    unique_targets = unique_targets.sort_values("Gene").reset_index(drop=True)

    # 分配 integer ID
    unique_drugs["ingredient_id"] = range(len(unique_drugs))
    unique_targets["target_id"] = range(len(unique_targets))

    # 映射 dict
    drug2idx = dict(zip(unique_drugs["DrugBank ID"], unique_drugs["ingredient_id"]))
    target2idx = dict(zip(unique_targets["Gene"], unique_targets["target_id"]))

    # 节点表
    ingredient_df = unique_drugs[["ingredient_id", "DrugBank ID", "SMILES"]]
    target_df = unique_targets[["target_id", "Gene", "Target Sequence"]]

    return drug2idx, target2idx, ingredient_df, target_df


# ================= 1) edges.txt：正样本边 =================

def write_edges(all_df: pd.DataFrame,
                drug2idx,
                target2idx,
                out_path: str = "edges.txt"):
    """
    从所有样本中取 Label == 1 的记录，输出 edges.txt：
    每行：ingredient_id target_id
    """
    pos = all_df[all_df["Label"] == 1].copy()

    seen = set()
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in pos.iterrows():
            d_id = row["DrugBank ID"]
            t_id = row["Gene"]
            if d_id not in drug2idx or t_id not in target2idx:
                continue
            ing_id = drug2idx[d_id]
            tgt_id = target2idx[t_id]
            pair = (ing_id, tgt_id)
            if pair in seen:
                continue
            seen.add(pair)
            f.write(f"{ing_id} {tgt_id}\n")

    print(f"[edges] 写入 {len(seen)} 条正样本边到 {out_path}")


# ================= 2) ingredient_similarity.txt：RDKit Tanimoto =================

def compute_ingredient_similarity(ingredient_df: pd.DataFrame,
                                  out_path: str = "ingredient_similarity.txt",
                                  threshold: float = ING_TANIMOTO_THRESHOLD):
    """
    用 RDKit 的 MorganGenerator 计算成分之间的 ECFP4 Tanimoto 相似度。

    - 指纹：radius=2, fpSize=1024（ECFP4）
    - 只保留相似度 >= threshold 的成分对
    - 输出格式：ingredient_id_i ingredient_id_j similarity
    """
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    except ImportError:
        raise ImportError(
            "本函数需要 RDKit，请先安装：conda install -c conda-forge rdkit"
        )

    ingredient_df = ingredient_df.sort_values("ingredient_id").reset_index(drop=True)
    ids = ingredient_df["ingredient_id"].tolist()
    smiles_list = ingredient_df["SMILES"].tolist()
    n = len(ids)

    # 新 API：MorganGenerator（避免老版本的 DEPRECATION WARNING）
    gen = GetMorganGenerator(radius=2, fpSize=1024)

    fps = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"[warn] SMILES 解析失败，跳过该成分的相似度：id={ids[i]}, smiles={smi}")
            fps.append(None)
        else:
            fp = gen.GetFingerprint(mol)
            fps.append(fp)

    cnt = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ii in range(n):
            if fps[ii] is None:
                continue
            if ii % 500 == 0:
                print(f"[ingredient sim] 进度: {ii}/{n}")
            for jj in range(ii + 1, n):
                if fps[jj] is None:
                    continue
                sim = DataStructs.TanimotoSimilarity(fps[ii], fps[jj])
                if sim >= threshold:
                    f.write(f"{ids[ii]} {ids[jj]} {sim}\n")
                    cnt += 1

    print(f"[ingredient_similarity] 共写入 {cnt} 条成分相似边到 {out_path} (阈值={threshold})")


# ================= 3) target_similarity.txt：Smith-Waterman 归一化 =================

def compute_target_similarity_sw(target_df: pd.DataFrame,
                                 out_path: str = "target_similarity.txt",
                                 threshold: float = TARGET_SW_THRESHOLD):
    """
    使用 Smith-Waterman 局部比对 + 归一化 SW 分数计算靶点之间的序列相似度，
    对应本文实验设置中的定义：

      SW_normalized(S_i, S_j) = SW(S_i, S_j) / sqrt( SW(S_i, S_i) * SW(S_j, S_j) )

    然后只保留 SW_normalized >= threshold 的 (i, j) 对。

    输出格式（3 列）：
      target_id_i  target_id_j  SW_normalized

    其中 target_id_* 与 target_nodes.csv 里的 target_id 、
    以及 edges.txt / *_clean.csv 里的 target_id 完全一致。
    """
    try:
        from Bio import pairwise2
        from Bio.Align import substitution_matrices
    except ImportError:
        raise ImportError(
            "需要 Biopython 才能计算 Smith-Waterman 相似度：\n"
            "  conda install biopython  或  pip install biopython"
        )

    # 使用 BLOSUM62 替换矩阵（常规蛋白 SW 设置）
    matrix = substitution_matrices.load("BLOSUM62")
    GAP_OPEN = -10   # 缺口打开罚分
    GAP_EXT  = -1    # 缺口延伸罚分

    # 按 target_id 排序，保证编号和其它文件一致
    target_df_sorted = target_df.sort_values("target_id").reset_index(drop=True)

    ids = target_df_sorted["target_id"].tolist()
    raw_seqs = target_df_sorted["Target Sequence"].tolist()

    # 清洗序列：去空白、转大写
    seqs = []
    for s in raw_seqs:
        if isinstance(s, str):
            seqs.append("".join(s.split()).upper())
        else:
            seqs.append("")

    n = len(seqs)

    # 第一步：先算每条序列的自比对得分 SW(S_i, S_i)
    self_scores = [0.0] * n
    print("[target sim] 先计算自比对分数 SW(S_i, S_i) ...")
    for i, seq in enumerate(seqs):
        if not seq:
            continue
        score = pairwise2.align.localds(
            seq, seq, matrix, GAP_OPEN, GAP_EXT, score_only=True
        )
        self_scores[i] = score
        if i % 200 == 0:
            print(f"  自比对进度: {i}/{n}")

    # 第二步：两两计算 SW_normalized，并按阈值过滤
    cnt = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(n):
            si = seqs[i]
            sii = self_scores[i]
            if not si or sii <= 0:
                continue

            if i % 50 == 0:
                print(f"[target sim] 两两比对进度: {i}/{n}")

            for j in range(i + 1, n):
                sj  = seqs[j]
                sjj = self_scores[j]
                if not sj or sjj <= 0:
                    continue

                # Smith-Waterman 局部比对原始分数 SW(S_i, S_j)
                sw_ij = pairwise2.align.localds(
                    si, sj, matrix, GAP_OPEN, GAP_EXT, score_only=True
                )
                if sw_ij <= 0:
                    continue

                # 归一化到 (0,1] 区间（论文 Eq.(2)）
                sw_norm = sw_ij / math.sqrt(sii * sjj)
                if sw_norm >= threshold:
                    f.write(f"{ids[i]} {ids[j]} {sw_norm:.6f}\n")
                    cnt += 1

    print(f"[target_similarity] 共写入 {cnt} 条 SW 归一化相似边到 {out_path} (阈值={threshold})")


# ================= 4) 清洗 train/val/test，并加入 ingredient_id/target_id =================

def save_clean_split(df: pd.DataFrame,
                     split_name: str,
                     out_dir: str,
                     drug2idx,
                     target2idx):
    """
    在原始 split 上加上 ingredient_id / target_id，并输出：
      - {split}_clean.csv
      - {split}_idx.txt
    """
    tmp = df.copy()
    tmp["ingredient_id"] = tmp["DrugBank ID"].map(drug2idx)
    tmp["target_id"] = tmp["Gene"].map(target2idx)

    # 少数映射不到的直接丢掉
    tmp = tmp.dropna(subset=["ingredient_id", "target_id"])
    tmp["ingredient_id"] = tmp["ingredient_id"].astype(int)
    tmp["target_id"] = tmp["target_id"].astype(int)

    cols = ["ingredient_id", "target_id",
            "DrugBank ID", "Gene", "SMILES", "Target Sequence", "Label"]
    tmp = tmp[cols]

    csv_path = os.path.join(out_dir, f"{split_name}_clean.csv")
    txt_path = os.path.join(out_dir, f"{split_name}_idx.txt")

    tmp.to_csv(csv_path, index=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        for _, row in tmp.iterrows():
            f.write(f"{row['ingredient_id']} {row['target_id']} {int(row['Label'])}\n")

    print(f"[split] {split_name}: 保存 {len(tmp)} 条样本到 {csv_path} 和 {txt_path}")


# ================= 主函数 =================

def main(train_path="train.csv",
         val_path="val.csv",
         test_path="test.csv",
         out_dir="."):
    os.makedirs(out_dir, exist_ok=True)

    # 1. 读入现有的 train/val/test
    splits = {}
    if train_path and os.path.exists(train_path):
        print(f"[load] 读取 train: {train_path}")
        splits["train"] = load_split(train_path)
    if val_path and os.path.exists(val_path):
        print(f"[load] 读取 val:   {val_path}")
        splits["val"] = load_split(val_path)
    if test_path and os.path.exists(test_path):
        print(f"[load] 读取 test:  {test_path}")
        splits["test"] = load_split(test_path)

    if not splits:
        raise ValueError("没有找到任何一个 split（train/val/test），请检查路径。")

    # 2. 合并所有样本，用于建立统一 ID 映射
    all_df = pd.concat(splits.values(), ignore_index=True)
    print(f"[info] 合并后总样本数: {len(all_df)}")

    print("[step] 建立 ingredient / target ID 映射 ...")
    drug2idx, target2idx, ingredient_df, target_df = build_index_maps(all_df)
    print(f"  不同 drugs 数量:   {len(ingredient_df)}")
    print(f"  不同 targets 数量: {len(target_df)}")

    # 保存节点表（方便你对照 ID ↔ 原始信息）
    ing_nodes_path = os.path.join(out_dir, "ingredient_nodes.csv")
    tgt_nodes_path = os.path.join(out_dir, "target_nodes.csv")
    ingredient_df.to_csv(ing_nodes_path, index=False)
    target_df.to_csv(tgt_nodes_path, index=False)
    print(f"[nodes] 成分节点表保存到 {ing_nodes_path}")
    print(f"[nodes] 靶点节点表保存到 {tgt_nodes_path}")

    # 3. 写 edges.txt
    edges_path = os.path.join(out_dir, "edges.txt")
    print("[step] 写 edges.txt ...")
    write_edges(all_df, drug2idx, target2idx, edges_path)

    # 4. 写 ingredient_similarity.txt（RDKit Tanimoto）
    ing_sim_path = os.path.join(out_dir, "ingredient_similarity.txt")
    print("[step] 计算成分相似度 ingredient_similarity.txt (RDKit Tanimoto) ...")
    compute_ingredient_similarity(ingredient_df, ing_sim_path, ING_TANIMOTO_THRESHOLD)

    # 5. 写 target_similarity.txt（Smith-Waterman 归一化）
    tgt_sim_path = os.path.join(out_dir, "target_similarity.txt")
    print("[step] 计算靶点相似度 target_similarity.txt (Smith-Waterman 归一化) ...")
    compute_target_similarity_sw(target_df, tgt_sim_path, TARGET_SW_THRESHOLD)

    # 6. 为每个 split 输出清洗后的 csv + idx.txt
    print("[step] 为各个 split 输出带 ID 的清洗文件 ...")
    for name, df in splits.items():
        save_clean_split(df, name, out_dir, drug2idx, target2idx)

    print("全部完成 ✅，所有 ID 在 txt 和 csv 中都已经对齐。")


if __name__ == "__main__":
    # 默认假设当前目录下有 train.csv / test.csv
    # 如果没有 val.csv 会自动跳过
    main(
        train_path="train.csv",
        val_path="val.csv",   # 没有就让它不存在即可
        test_path="test.csv",
        out_dir="."
    )
