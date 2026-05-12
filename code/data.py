# -*- coding: utf-8 -*-
"""
utlis.py
- 读取数据/嵌入（含 ID 对齐；CSV/NPZ 支持）
- 构建异构图（相似边按无向加双边）
- 写入全尺寸节点特征（未知节点补零）
- InfoNCE 损失 / 常用工具
- 【新增】LLM 嵌入消融：
    * 通过 args.ablate_ligand_llm / args.ablate_target_llm 控制是否消融
    * 通过 args.llm_ablation_mode 控制模式：zero / random / shuffle
"""
from pathlib import Path
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import dgl

# =================== 随机种子 ===================

def set_seed(seed: int = 410):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    dgl.seed(seed)

# =================== 基础读取 ===================

def _read_edge_table_2cols(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"[,\s\t]+", header=None, engine="python", comment="#")
    if df.shape[1] < 2:
        raise ValueError(f"{path} 至少应包含两列 (source, target)，实际列数={df.shape[1]}")
    df = df.iloc[:, :2].copy()
    df.columns = ["source", "target"]
    df["source"] = df["source"].astype(int)
    df["target"] = df["target"].astype(int)
    return df

def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")

def _pick_tensor_like(obj) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for k in ("embeddings", "embed", "feat", "features", "x", "data"):
            if k in obj:
                return torch.as_tensor(obj[k])
        for v in obj.values():
            try:
                return torch.as_tensor(v)
            except Exception:
                continue
        raise ValueError("dict 中未找到可用的张量值")
    if isinstance(obj, (list, tuple)):
        return torch.as_tensor(obj[0])
    raise ValueError(f"不支持的对象类型: {type(obj)}")

def _load_embedding_any(path: str, npz_key: str | None = None) -> torch.Tensor:
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Embedding file not found: {path}")
    ext = Path(path).suffix.lower()
    if ext in [".pt", ".pth"]:
        obj = _safe_torch_load(path)
        t = _pick_tensor_like(obj)
        return t.float()
    if ext == ".npy":
        arr = np.load(path)
        return torch.from_numpy(arr).float()
    if ext == ".npz":
        z = np.load(path)
        files = list(z.files)
        if npz_key is not None:
            if npz_key not in files:
                raise KeyError(f"npz key '{npz_key}' not in {files}")
            arr = z[npz_key]
        else:
            if len(files) != 1:
                raise KeyError(f"{path} contains multiple arrays {files}, please set --target_npz_key")
            arr = z[files[0]]
        return torch.from_numpy(arr).float()
    if ext == ".csv":
        raise ValueError("CSV 请使用 _load_embedding_csv()")
    raise ValueError(f"Unsupported embedding file format: {ext}")

def _load_embedding_csv(path: str, id_key: str = "node_id") -> tuple[torch.Tensor, np.ndarray | None]:
    df = pd.read_csv(path)
    ids = None
    if id_key in df.columns:
        ids = df[id_key].values
        df = df.drop(columns=[id_key])
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        raise ValueError(f"{path} 中未找到数值特征列")
    emb = torch.from_numpy(df[num_cols].to_numpy()).float()
    if ids is not None:
        if df.shape[0] != len(ids):
            raise ValueError(f"{path}: 特征行数与 {id_key} 数不一致")
        ids = np.asarray(ids)
        if ids.dtype == object:
            ids = np.array([int(x) for x in ids], dtype=np.int64)
        else:
            ids = ids.astype(np.int64, copy=False)
    return emb, ids

def _load_npz_ids(path: str, id_key: str = "node_id") -> np.ndarray:
    z = np.load(path, allow_pickle=True)
    if id_key not in z.files:
        raise KeyError(f"'{id_key}' not found in {list(z.files)}")
    ids = z[id_key]
    if ids.dtype == object:
        ids = np.array([int(x) for x in ids], dtype=np.int64)
    else:
        ids = ids.astype(np.int64, copy=False)
    return ids


# =================== LLM 嵌入消融（新增） ===================

def _apply_llm_ablation(
    emb: torch.Tensor,
    ablate: bool,
    mode: str = "zero",
    name: str = "",
) -> torch.Tensor:
    """
    对整条 embedding 向量做 LLM ablation。
    这里默认假设传进来的 emb 就是“LLM 向量”（或你希望整块都视作 LLM 部分）。

    mode:
      - 'zero'    : 直接置零
      - 'random'  : 用同方差的高斯噪声替换
      - 'shuffle' : 打乱节点与向量的对应关系（保持分布不变，语义打乱）
    """
    if not ablate:
        return emb

    if emb.numel() == 0:
        return emb

    emb = emb.clone()
    mode = (mode or "zero").lower()

    if mode == "zero":
        emb.zero_()
    elif mode == "random":
        # 用原 embedding 的整体 std，尽量保持数值尺度一致
        std = float(emb.std().item())
        if std == 0.0:
            std = 1.0
        emb.normal_(mean=0.0, std=std)
    elif mode == "shuffle":
        # 打乱样本顺序，保持行内结构不变
        perm = torch.randperm(emb.size(0))
        emb = emb[perm]
    else:
        raise ValueError(f"Unknown llm_ablation_mode: {mode}")

    print(f"[LLM ablation] {name}: mode={mode}, shape={tuple(emb.shape)}")
    return emb


# =================== 数据入口 ===================

def process_data(args):
    base_path = Path(getattr(args, "data_dir", "data"))
    edges    = _read_edge_table_2cols(base_path / "edges.txt")
    is_edges = _read_edge_table_2cols(base_path / "ingredient_similarity.txt")
    ts_edges = _read_edge_table_2cols(base_path / "target_similarity.txt")

    ligand_path    = getattr(args, "ligand_embed", "my_embeddings_ligand.pt")
    target_path    = getattr(args, "target_embed", "my_embeddings.npz")
    ligand_id_key  = getattr(args, "ligand_id_key", "node_id")
    target_npz_key = getattr(args, "target_npz_key", None)
    target_id_key  = getattr(args, "target_id_key", "node_id")

    ext_ligand = Path(ligand_path).suffix.lower()
    ext_target = Path(target_path).suffix.lower()

    if ext_ligand == ".csv":
        emb_ing, ids_ing = _load_embedding_csv(ligand_path, ligand_id_key)
    else:
        emb_ing = _load_embedding_any(ligand_path, None)
        ids_ing = None

    if ext_target == ".csv":
        emb_tgt, ids_tgt = _load_embedding_csv(target_path, target_id_key)
    elif ext_target == ".npz":
        emb_tgt = _load_embedding_any(target_path, target_npz_key)
        ids_tgt = _load_npz_ids(target_path, target_id_key)
    else:
        emb_tgt = _load_embedding_any(target_path, target_npz_key)
        ids_tgt = None

    # ====== 在这里对 LLM 向量做消融（新增） ======
    ablate_ligand = bool(getattr(args, "ablate_ligand_llm", False))
    ablate_target = bool(getattr(args, "ablate_target_llm", False))
    llm_mode      = getattr(args, "llm_ablation_mode", "zero")

    emb_ing = _apply_llm_ablation(emb_ing, ablate_ligand, llm_mode, name="ligand")
    emb_tgt = _apply_llm_ablation(emb_tgt, ablate_target, llm_mode, name="target")

    initial_features = {
        "pretrained": {"ingredient": emb_ing, "target": emb_tgt},
        "ids": {"ingredient": None if ids_ing is None else torch.from_numpy(ids_ing),
                "target": None if ids_tgt is None else torch.from_numpy(ids_tgt)}
    }
    return edges, is_edges, ts_edges, initial_features

# =================== 构图（含 ID 对齐 + 补零） ===================

def build_graph(args, edges, is_edges, ts_edges, initial_features, device):
    os.environ['DGLBACKEND'] = 'pytorch'

    e = torch.from_numpy(edges[["source","target"]].values).long()
    graph_data = {
        ('ingredient', 'it', 'target'): (e[:, 0], e[:, 1]),
        ('target', 'ti', 'ingredient'): (e[:, 1], e[:, 0]),
    }
    if args.graph_struct in [1, 3]:
        is_e = torch.from_numpy(is_edges[["source","target"]].values).long()
        graph_data[('ingredient', 'is', 'ingredient')] = (
            torch.cat([is_e[:, 0], is_e[:, 1]]),
            torch.cat([is_e[:, 1], is_e[:, 0]]),
        )
    if args.graph_struct in [2, 3]:
        ts_e = torch.from_numpy(ts_edges[["source","target"]].values).long()
        graph_data[('target', 'ts', 'target')] = (
            torch.cat([ts_e[:, 0], ts_e[:, 1]]),
            torch.cat([ts_e[:, 1], ts_e[:, 0]]),
        )

    max_ing_from_edges = int(edges["source"].max()) if len(edges) else -1
    max_tgt_from_edges = int(edges["target"].max()) if len(edges) else -1
    if len(is_edges):
        max_ing_from_edges = max(max_ing_from_edges, int(is_edges[["source","target"]].values.max()))
    if len(ts_edges):
        max_tgt_from_edges = max(max_tgt_from_edges, int(ts_edges[["source","target"]].values.max()))

    feats = initial_features["pretrained"]
    emb_ing = feats['ingredient']
    emb_tgt = feats['target']
    N_ing, d_ing = emb_ing.shape[0], emb_ing.shape[1]
    N_tgt, d_tgt = emb_tgt.shape[0], emb_tgt.shape[1]

    ids_info = initial_features.get("ids", {})
    ing_ids  = ids_info.get("ingredient", None)
    tgt_ids  = ids_info.get("target", None)
    max_ing_from_ids = int(ing_ids.max().item()) if ing_ids is not None else -1
    max_tgt_from_ids = int(tgt_ids.max().item()) if tgt_ids is not None else -1

    num_nodes_target     = max(max_tgt_from_edges + 1, max_tgt_from_ids + 1, N_tgt)
    num_nodes_ingredient = max(max_ing_from_edges + 1, max_ing_from_ids + 1, N_ing)

    g = dgl.heterograph(
        graph_data,
        num_nodes_dict={'ingredient': num_nodes_ingredient, 'target': num_nodes_target}
    ).to(device)

    feat_ing_full = torch.zeros((num_nodes_ingredient, d_ing), dtype=torch.float32)
    if ing_ids is None:
        feat_ing_full[:N_ing] = emb_ing
    else:
        idx = ing_ids.long()
        if int(idx.max().item()) >= num_nodes_ingredient:
            raise ValueError(f"ingredient 的原始ID最大值 {int(idx.max().item())} 超过 num_nodes_ingredient={num_nodes_ingredient}")
        feat_ing_full[idx] = emb_ing
    g.nodes['ingredient'].data['features'] = feat_ing_full.to(device)

    feat_tgt_full = torch.zeros((num_nodes_target, d_tgt), dtype=torch.float32)
    if tgt_ids is None:
        feat_tgt_full[:N_tgt] = emb_tgt
    else:
        idx = tgt_ids.long()
        if int(idx.max().item()) >= num_nodes_target:
            raise ValueError(
                f"target 的原始ID最大值 {int(idx.max().item())} 超过 num_nodes_target={num_nodes_target}，"
                f"请检查 edges 与 node_id 的一致性。"
            )
        feat_tgt_full[idx] = emb_tgt
    g.nodes['target'].data['features'] = feat_tgt_full.to(device)

    rel_list = [
        ('ingredient', 'it', 'target'),
        ('target', 'ti', 'ingredient'),
        ('ingredient', 'is', 'ingredient'),
        ('target', 'ts', 'target'),
    ]
    return g, rel_list

# =================== 损失：InfoNCE ===================

def compute_loss(pos_score, neg_score, etype, tau: float = 0.07, top_m: int = 0):
    pos = pos_score[etype].reshape(-1) / tau
    B = pos.shape[0]
    neg_all = neg_score[etype].reshape(-1) / tau
    Nneg = neg_all.numel()
    if Nneg == 0:
        return F.softplus(-pos).mean()
    K = max(1, Nneg // B)
    neg = neg_all[:B * K].view(B, K)
    if isinstance(top_m, int) and 0 < top_m < K:
        neg, _ = torch.topk(neg, k=top_m, dim=1)
    logits = torch.cat([pos.unsqueeze(1), neg], dim=1)
    labels = torch.zeros(B, dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)

# =================== 其它工具 ===================

def cos_sim(a: torch.Tensor, b: torch.Tensor, dim: int = 1, eps: float = 1e-8):
    return F.cosine_similarity(a, b, dim=dim, eps=eps)

@torch.no_grad()
def remove_unseen_nodes(node_type: str, g: dgl.DGLHeteroGraph, unseen_ids_np):
    unseen_ids = torch.as_tensor(unseen_ids_np, device=g.device, dtype=torch.long)
    src, dst = g.edges(etype='it')
    if node_type == 'ingredient':
        mask = torch.isin(src, unseen_ids)
    elif node_type == 'target':
        mask = torch.isin(dst, unseen_ids)
    else:
        raise ValueError("node_type must be 'ingredient' or 'target'")
    remove_src = src[mask].clone()
    remove_dst = dst[mask].clone()
    g_sub = dgl.remove_nodes(g, unseen_ids, ntype=node_type)
    return g_sub, remove_src, remove_dst

@torch.no_grad()
def negative_sampling_batched(g, src_nodes: torch.Tensor, num_targets: int,
                              etype=('ingredient','it','target'),
                              m_candidates: int = 128):
    device = g.device
    B = src_nodes.shape[0]
    cand = torch.randint(0, num_targets, (B, m_candidates), device=device)
    u = src_nodes.view(-1, 1).expand(-1, m_candidates).reshape(-1)
    v = cand.reshape(-1)
    exist = g.has_edges_between(u, v, etype=etype).view(B, m_candidates)
    valid = ~exist
    has_valid = valid.any(dim=1)
    first_idx = valid.float().argmax(dim=1)
    chosen = cand[torch.arange(B, device=device), first_idx]
    fallback = torch.randint(0, num_targets, (B,), device=device)
    neg_dst = torch.where(has_valid, chosen, fallback)
    return src_nodes, neg_dst

@torch.no_grad()
def augment_similarity_graph(g: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
    """
    仅对同型相似关系（'is'、'ts'）：
      1) 在“补自环”之前，计算并写入 sim_deg_orig（不含自环）
      2) 为每个缺少自环的节点添加 1 条自环（避免重复添加）
    返回：新的 heterograph（DGL 会复制一份图结构）
    """
    device = g.device if hasattr(g, "device") else torch.device("cpu")

    def _write_sim_deg_orig(etype_name: str):
        cand = [ce for ce in g.canonical_etypes if ce[1] == etype_name and ce[0] == ce[2]]
        if not cand:
            return
        ntype = cand[0][0]
        _, dst = g.edges(etype=etype_name)
        num = g.num_nodes(ntype)
        deg = torch.bincount(dst.to(torch.long), minlength=num).to(device)
        g.nodes[ntype].data['sim_deg_orig'] = deg

    def _add_missing_self_loops(etype_name: str):
        cand = [ce for ce in g.canonical_etypes if ce[1] == etype_name and ce[0] == ce[2]]
        if not cand:
            return g
        ntype = cand[0][0]
        n = g.num_nodes(ntype)
        nodes = torch.arange(n, device=device)
        has = g.has_edges_between(nodes, nodes, etype=etype_name)
        add_nodes = nodes[~has]
        if add_nodes.numel() > 0:
            g2 = dgl.add_edges(g, add_nodes, add_nodes, etype=etype_name)
            return g2
        return g

    _write_sim_deg_orig('is')
    _write_sim_deg_orig('ts')
    g = _add_missing_self_loops('is')
    g = _add_missing_self_loops('ts')
    return g

