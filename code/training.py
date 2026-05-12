# -*- coding: utf-8 -*-
"""
train.py — HGT 链路预测训练脚本（含“同型相似边补自环”可控开关 + EMA）

要点：
  • CVS1 的评估支持 3 种口径（--cv1_eval）：
      - strict       : 移除验证监督边 + 其反向边（最严格）
      - keep_reverse : 只移除验证监督边本身，保留反向边（默认，轻微泄露）
      - legacy       : 不移除任何验证监督边（最松）
  • CVS2/3 的评估：仅保留训练监督边（无泄漏评估）
  • 训练期严格冷启动：--strict_unseen 或 --strict_cold_start（等价）
  • DataLoader 采样：exclude='reverse_types' + reverse_etypes，避免批内捷径
  • 可控图增强：
      - --augment_sim_loops  开启“is/ts 同型边补自环 + 写 sim_deg_orig”
      - --augment_cv1        在 CVS1 下也执行增强（默认 CVS1 不增强）
  • 新增 EMA：
      - --use_ema --ema_decay
  • 【配合 utlis.py 新增】LLM 嵌入消融：
      - --ablate_ligand_llm / --ablate_target_llm / --llm_ablation_mode
"""

import os
import time
import numpy as np
from typing import Tuple, Dict, List

import torch
import torch.nn as nn
import dgl
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score
from dgl.dataloading import NeighborSampler, as_edge_prediction_sampler, DataLoader
from dgl.dataloading.negative_sampler import GlobalUniform
from torch.optim.swa_utils import AveragedModel   # 用作 EMA

# =============== Package imports ===============
try:
    from model import HGTModel as Model
    from data import (
        set_seed, process_data, build_graph, compute_loss, remove_unseen_nodes
    )
    try:
        from data import augment_similarity_graph  # type: ignore
    except Exception:
        augment_similarity_graph = None
except ImportError:
    # Fallback for direct execution during local development.
    from model import HGTModel as Model  # type: ignore
    from data import (  # type: ignore
        set_seed, process_data, build_graph, compute_loss, remove_unseen_nodes
    )
    try:
        from data import augment_similarity_graph  # type: ignore
    except Exception:
        augment_similarity_graph = None

# =============== 反向关系映射（用于 exclude='reverse_types'） ===============
def _reverse_map_all(g: dgl.DGLHeteroGraph) -> Dict[str, str]:
    rev = {}
    if ('ingredient', 'it', 'target') in g.canonical_etypes:
        rev['it'] = 'ti'
    if ('target', 'ti', 'ingredient') in g.canonical_etypes:
        rev['ti'] = 'it'
    if ('ingredient', 'is', 'ingredient') in g.canonical_etypes:
        rev['is'] = 'is'
    if ('target', 'ts', 'target') in g.canonical_etypes:
        rev['ts'] = 'ts'
    return rev

# =============== Edge DataLoader ===============
def _build_edge_loader(
    g: dgl.DGLHeteroGraph,
    etype_name: str,
    eids: torch.Tensor,
    fanout_per_layer,
    batch_size: int,
    device: torch.device,
    neg_k: int = 5,
    shuffle: bool = True,
    exclude_edges: bool = True,
):
    if isinstance(fanout_per_layer, int):
        fanout_per_layer = [fanout_per_layer]

    sampler = as_edge_prediction_sampler(
        NeighborSampler(fanout_per_layer),
        negative_sampler=GlobalUniform(neg_k),
        **({
            "exclude": "reverse_types",
            "reverse_etypes": _reverse_map_all(g)
        } if exclude_edges else {})
    )

    loader = DataLoader(
        g,
        {etype_name: eids},
        sampler,
        device=device,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
    )
    return loader

# =============== 评估用图构造（CVS2/3：仅保留训练监督边） ===============
@torch.no_grad()
def _build_eval_graph_keep_train_sup_only(
    hetero_graph: dgl.DGLHeteroGraph,
    sup_rel_name: str,
    train_eids_on_full: torch.Tensor
) -> dgl.DGLHeteroGraph:
    all_sup_eids = hetero_graph.edges(etype=sup_rel_name, form='eid')
    keep_mask = torch.zeros(all_sup_eids.shape[0], dtype=torch.bool, device=all_sup_eids.device)
    keep_mask[train_eids_on_full] = True
    remove_eids = all_sup_eids[~keep_mask]
    if remove_eids.numel() == 0:
        return hetero_graph
    return dgl.remove_edges(hetero_graph, eids=remove_eids, etype=sup_rel_name)

# =============== CVS1 严格：删正向 + 对应反向 ===============
@torch.no_grad()
def _build_eval_graph_cv1_strict(
    hetero_graph: dgl.DGLHeteroGraph,
    sup_rel_name: str,
    val_eids_on_full: torch.Tensor
) -> dgl.DGLHeteroGraph:
    src_v, dst_v = hetero_graph.find_edges(val_eids_on_full, etype=sup_rel_name)
    g_eval = dgl.remove_edges(hetero_graph, eids=val_eids_on_full, etype=sup_rel_name)
    rev_map = _reverse_map_all(hetero_graph)
    rev_etype = rev_map.get(sup_rel_name, None)
    if rev_etype is not None:
        has_rev = hetero_graph.has_edges_between(dst_v, src_v, etype=rev_etype)
        if has_rev.any():
            eids_rev = hetero_graph.edge_ids(dst_v[has_rev], src_v[has_rev], etype=rev_etype)
            g_eval = dgl.remove_edges(g_eval, eids=eids_rev, etype=rev_etype)
    return g_eval

# =============== CVS1 微泄露：只删正向，保留反向（默认） ===============
@torch.no_grad()
def _build_eval_graph_cv1_keep_reverse(
    hetero_graph: dgl.DGLHeteroGraph,
    sup_rel_name: str,
    val_eids_on_full: torch.Tensor
) -> dgl.DGLHeteroGraph:
    return dgl.remove_edges(hetero_graph, eids=val_eids_on_full, etype=sup_rel_name)

# =============== 图增强兜底实现（与 utlis.augment_similarity_graph 等价） ===============
@torch.no_grad()
def _augment_similarity_graph_fallback(g: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
    def _write_sim_deg_orig(etype_name: str):
        cand = [ce for ce in g.canonical_etypes if ce[1] == etype_name and ce[0] == ce[2]]
        if not cand:
            return
        ntype = cand[0][0]
        _, dst = g.edges(etype=etype_name)
        num = g.num_nodes(ntype)
        deg = torch.bincount(dst.to(torch.long), minlength=num).to(g.device)
        g.nodes[ntype].data['sim_deg_orig'] = deg

    def _add_missing_self_loops(etype_name: str):
        cand = [ce for ce in g.canonical_etypes if ce[1] == etype_name and ce[0] == ce[2]]
        if not cand:
            return g
        ntype = cand[0][0]
        n = g.num_nodes(ntype)
        nodes = torch.arange(n, device=g.device)
        has = g.has_edges_between(nodes, nodes, etype=etype_name)
        add_nodes = nodes[~has]
        if add_nodes.numel() > 0:
            g2 = dgl.add_edges(g, add_nodes, add_nodes, etype=etype_name)
            return g2
        return g

    _write_sim_deg_orig('is'); _write_sim_deg_orig('ts')
    g = _add_missing_self_loops('is')
    g = _add_missing_self_loops('ts')
    return g

# =============== 评估 ===============
@torch.no_grad()
def evaluate(model, loader, sup_rel, args) -> Tuple[float, float]:
    model.eval()
    pos_all, neg_all = [], []
    with torch.no_grad():
        for _, pos_g, neg_g, blocks in loader:
            x_dict = blocks[0].srcdata['features']
            pos_score, neg_score = model(args, pos_g, neg_g, blocks, x_dict)
            pos_all.append(pos_score[sup_rel].reshape(-1).cpu())
            neg_all.append(neg_score[sup_rel].reshape(-1).cpu())

    if not pos_all or not neg_all:
        return 0.0, 0.0
    pos = torch.cat(pos_all).numpy()
    neg = torch.cat(neg_all).numpy()
    y_true = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
    y_pred = np.concatenate([pos, neg])
    auroc = roc_auc_score(y_true, y_pred)
    auprc = average_precision_score(y_true, y_pred)
    return auroc, auprc

# =============== 简单 K 折（CVS1=随机边；CVS2/3=按源/目标分组） ===============
def _make_folds(g: dgl.DGLHeteroGraph, sup_rel: Tuple[str,str,str], k: int, cv_mode: str):
    sup_rel_name = sup_rel[1]
    src_all, dst_all = g.edges(etype=sup_rel_name)
    eids_all = g.edges(etype=sup_rel_name, form='eid')
    idx = np.arange(eids_all.shape[0])

    cv_mode = cv_mode.upper()
    if cv_mode == "CVS1":
        kf = KFold(n_splits=k, shuffle=True, random_state=411)
        folds = []
        for tr, va in kf.split(idx):
            folds.append((eids_all[torch.as_tensor(tr, device=eids_all.device)],
                          eids_all[torch.as_tensor(va, device=eids_all.device)]))
        return folds

    if cv_mode == "CVS2":  # 新配体：按源节点分组
        uniq_src = torch.unique(src_all).cpu().numpy()
        rng = np.random.RandomState(411)
        rng.shuffle(uniq_src)
        buckets = [[] for _ in range(min(k, len(uniq_src)))]
        for i, u in enumerate(uniq_src):
            buckets[i % len(buckets)].append(u)
        folds = []
        src_np, e_np = src_all.cpu().numpy(), eids_all.cpu().numpy()
        for b in buckets:
            va_mask = np.isin(src_np, b)
            tr_mask = ~va_mask
            folds.append((torch.as_tensor(e_np[tr_mask], device=eids_all.device),
                          torch.as_tensor(e_np[va_mask], device=eids_all.device)))
        return folds

    if cv_mode == "CVS3":  # 新靶点：按目标节点分组
        uniq_dst = torch.unique(dst_all).cpu().numpy()
        rng = np.random.RandomState(411)
        rng.shuffle(uniq_dst)
        buckets = [[] for _ in range(min(k, len(uniq_dst)))]
        for i, v in enumerate(uniq_dst):
            buckets[i % len(buckets)].append(v)
        folds = []
        dst_np, e_np = dst_all.cpu().numpy(), eids_all.cpu().numpy()
        for b in buckets:
            va_mask = np.isin(dst_np, b)
            tr_mask = ~va_mask
            folds.append((torch.as_tensor(e_np[tr_mask], device=eids_all.device),
                          torch.as_tensor(e_np[va_mask], device=eids_all.device)))
        return folds

    raise ValueError(f"Unknown cv_mode: {cv_mode}")

# =============== 训练主流程 ===============
def train(args, hetero_graph: dgl.DGLHeteroGraph, rel_list, device):
    set_seed(getattr(args, "seed", 410))

    sup_rel = rel_list[0]                 # e.g. ('ingredient','it','target')
    sup_rel_name = sup_rel[1]

    # 兼容旧脚本：把 input_gate_type=etype 映射到 'se'
    if getattr(args, "input_gate_type", "none") == "etype":
        args.input_gate_type = "se"

    # 兼容两个严格冷启动开关
    if getattr(args, "strict_unseen", False):
        args.strict_cold_start = True

    folds = _make_folds(hetero_graph, sup_rel, args.k_fold, args.cv_mode)

    results = []
    for fold, (train_eids, val_eids) in enumerate(folds, start=1):
        t0 = time.time()
        fanout = [args.fanout] * args.num_layers

        # ==== 构造训练子图（严格冷启动：仅 CVS2/3 删除“未见节点”） ====
        g_train = hetero_graph
        if getattr(args, "strict_cold_start", False):
            src_all, dst_all = hetero_graph.edges(etype=sup_rel_name)
            if args.cv_mode.upper() == "CVS2":
                val_src_nodes = torch.unique(src_all[val_eids]).cpu().numpy()
                g_train, _, _ = remove_unseen_nodes('ingredient', hetero_graph, val_src_nodes)
            elif args.cv_mode.upper() == "CVS3":
                val_dst_nodes = torch.unique(dst_all[val_eids]).cpu().numpy()
                g_train, _, _ = remove_unseen_nodes('target', hetero_graph, val_dst_nodes)
            # CVS1 不删节点

        # 采样阶段的边排除策略（CVS1 若显式 --no_exclude_cv1 才允许关闭；其余一律开启）
        exclude_edges = not (args.cv_mode.upper() == "CVS1" and getattr(args, "no_exclude_cv1", False))

        # ==== DataLoader ====
        train_eids_sub = g_train.edges(etype=sup_rel_name, form='eid')
        train_loader = _build_edge_loader(
            g_train, sup_rel_name, train_eids_sub, fanout, args.batch_size,
            device, neg_k=args.neg_k, shuffle=True, exclude_edges=exclude_edges
        )
        val_loader = _build_edge_loader(
            hetero_graph, sup_rel_name, val_eids, fanout, args.batch_size,
            device, neg_k=1, shuffle=False, exclude_edges=exclude_edges
        )

        # ==== 模型（训练在 g_train） ====
        model = Model(args, g_train, rel_list).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

        # ==== EMA ====
        ema = AveragedModel(model) if getattr(args, "use_ema", False) else None
        ema_decay = float(getattr(args, "ema_decay", 0.999))

        # ==== 学习率调度 ====
        if getattr(args, "use_cosine", False):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt, T_0=max(1, args.cosine_T0), T_mult=max(1, args.cosine_Tmult))
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_period, gamma=args.lr_decay)

        if getattr(args, "print_model", False):
            print(
                f"[Model] input_gate={getattr(args,'input_gate_type','none')} "
                f"(reduce={getattr(args,'input_gate_reduce',4)}) | "
                f"residual_gate={getattr(args,'residual_gate',False)} | "
                f"score_gate={getattr(args,'score_gate','none')} | film_cond={getattr(args,'film_condition','dst')} | "
                f"semantic_gate={getattr(args,'semantic_gate','none')} | head_gate={getattr(args,'head_gate',False)} | "
                f"strict_cold_start={getattr(args,'strict_cold_start',False)} | exclude_edges={exclude_edges} "
                f"(cv_mode={args.cv_mode}, cv1_eval={getattr(args,'cv1_eval','keep_reverse')})"
            )

        best_metric = -float('inf')
        best_ckpt_path = None

        for epoch in tqdm(range(args.num_epochs), desc=f"HGT-Fold{fold}"):
            model.train()
            losses = []
            for _, pos_g, neg_g, blocks in train_loader:
                x_dict = blocks[0].srcdata['features']
                pos_score, neg_score = model(args, pos_g, neg_g, blocks, x_dict)
                loss = compute_loss(
                    pos_score, neg_score, sup_rel,
                    tau=getattr(args, "tau", 0.07),
                    top_m=getattr(args, "top_m", 0),
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                opt.step()

                # EMA 更新
                if ema is not None:
                    for p_ema, p in zip(ema.parameters(), model.parameters()):
                        p_ema.data.mul_(ema_decay).add_(p.data, alpha=(1.0 - ema_decay))

                losses.append(loss.item())

            scheduler.step()
            if (epoch + 1) % max(1, args.log_every) == 0:
                print(f"[Fold {fold}] epoch={epoch+1} loss={np.mean(losses):.4f}")

            # ==== 周期性验证（用 EMA 或当前参数） ====
            if ((epoch + 1) % max(1, args.val_every) == 0) or ((epoch + 1) == args.num_epochs):
                # 构造评估图
                if args.cv_mode.upper() == "CVS1":
                    mode = getattr(args, "cv1_eval", "keep_reverse")
                    if mode == "strict":
                        g_eval = _build_eval_graph_cv1_strict(hetero_graph, sup_rel_name, val_eids)
                    elif mode == "keep_reverse":
                        g_eval = _build_eval_graph_cv1_keep_reverse(hetero_graph, sup_rel_name, val_eids)
                    elif mode == "legacy":
                        g_eval = hetero_graph
                    else:
                        raise ValueError(f"Unknown cv1_eval: {mode}")
                else:
                    g_eval = _build_eval_graph_keep_train_sup_only(hetero_graph, sup_rel_name, train_eids)

                model_eval = Model(args, g_eval, rel_list).to(device)
                state = (ema.module.state_dict() if ema is not None else model.state_dict())
                model_eval.load_state_dict(state, strict=True)

                auroc_val, auprc_val = evaluate(model_eval, val_loader, sup_rel, args)
                monitor = auprc_val if args.monitor_metric == "auprc" else auroc_val

                if best_ckpt_path is None:
                    os.makedirs(args.checkpoint_dir, exist_ok=True)
                    best_ckpt_path = os.path.join(args.checkpoint_dir, f"best_fold{fold}_{args.monitor_metric}.pt")

                if monitor > best_metric:
                    best_metric = float(monitor)
                    torch.save(
                        {
                            "state_dict": state,  # 保存 EMA 或当前参数
                            "args": vars(args),
                            "fold": fold,
                            "epoch": int(epoch + 1),
                            "monitor_metric": args.monitor_metric,
                            "best_value": float(best_metric),
                            "sup_rel": sup_rel,
                            "use_ema": bool(ema is not None),
                        },
                        best_ckpt_path,
                    )
                    print(f"[Fold {fold}] ✅ New best {args.monitor_metric}={monitor:.4f} @ epoch {epoch+1} → {best_ckpt_path}")
                else:
                    print(f"[Fold {fold}] val {args.monitor_metric}={monitor:.4f} (best={best_metric:.4f})")

        # ==== 折内最终评估（EMA/当前） ====
        train_time = time.time() - t0
        if args.cv_mode.upper() == "CVS1":
            mode = getattr(args, "cv1_eval", "keep_reverse")
            if mode == "strict":
                g_eval = _build_eval_graph_cv1_strict(hetero_graph, sup_rel_name, val_eids)
            elif mode == "keep_reverse":
                g_eval = _build_eval_graph_cv1_keep_reverse(hetero_graph, sup_rel_name, val_eids)
            elif mode == "legacy":
                g_eval = hetero_graph
            else:
                raise ValueError(f"Unknown cv1_eval: {mode}")
        else:
            g_eval = _build_eval_graph_keep_train_sup_only(hetero_graph, sup_rel_name, train_eids)

        model_eval = Model(args, g_eval, rel_list).to(device)
        final_state = (ema.module.state_dict() if ema is not None else model.state_dict())
        model_eval.load_state_dict(final_state, strict=True)
        model_eval.eval()

        pos_all, neg_all = [], []
        with torch.no_grad():
            for _, pos_g, neg_g, blocks in val_loader:
                x_dict = blocks[0].srcdata['features']
                pos_score, neg_score = model_eval(args, pos_g, neg_g, blocks, x_dict)
                pos_all.append(pos_score[sup_rel].reshape(-1).cpu())
                neg_all.append(neg_score[sup_rel].reshape(-1).cpu())

        pos = torch.cat(pos_all).numpy() if pos_all else np.array([])
        neg = torch.cat(neg_all).numpy() if neg_all else np.array([])
        y_true = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])
        y_pred = np.concatenate([pos, neg])
        auroc = roc_auc_score(y_true, y_pred)
        auprc = average_precision_score(y_true, y_pred)
        print(f"[Fold {fold}] time={train_time:.1f}s | AUROC={auroc:.4f} | AUPRC={auprc:.4f}")

        results.append((auroc, auprc))

    arr = np.array(results)
    print(f"[CV] AUROC mean={arr[:,0].mean():.4f} std={arr[:,0].std():.4f} | "
          f"AUPRC mean={arr[:,1].mean():.4f} std={arr[:,1].std():.4f}")

# =============== CLI ===============
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # 设备 / 随机
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=410)

    # K 折 / 采样
    parser.add_argument("--k_fold", type=int, default=10)
    parser.add_argument("--cv_mode", choices=["CVS1","CVS2","CVS3","cv1","cv2","cv3"], default="CVS1")
    parser.add_argument("--strict_cold_start", action="store_true")
    parser.add_argument("--strict_unseen", action="store_true")   # 等价于 strict_cold_start
    parser.add_argument("--no_exclude_cv1", action="store_true")
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--fanout", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--neg_k", type=int, default=16)

    # 数据
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ligand_embed", type=str, required=True)
    parser.add_argument("--ligand_id_key", type=str, default="node_id")
    parser.add_argument("--target_embed", type=str, required=True)
    parser.add_argument("--target_id_key", type=str, default="node_id")

    # 兼容旧脚本参数
    parser.add_argument("--graph_struct", type=int, default=3)
    parser.add_argument("--method", type=int, default=5)

    # 模型结构
    parser.add_argument("--in_dim", type=int, default=512)
    parser.add_argument("--h_dim", type=int, default=512)
    parser.add_argument("--out_dim", type=int, default=512)
    parser.add_argument("--hgt_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)

    # 门控/语义
    parser.add_argument("--input_gate_type", choices=["none","se","glu","etype"], default="se",
                        help="'etype' 作为旧版别名，将自动映射到 'se'")
    parser.add_argument("--input_gate_reduce", type=int, default=4)
    parser.add_argument("--residual_gate", action="store_true")
    parser.add_argument("--score_gate", choices=["none","gmu","film"], default="gmu")
    parser.add_argument("--film_condition", choices=["src","dst","both"], default="src")
    parser.add_argument("--semantic_gate", choices=["none","etype"], default="etype")
    parser.add_argument("--sem_gate_bias", type=float, default=0.8)
    parser.add_argument("--head_gate", action="store_true")

    # ====== LLM 嵌入消融（新增） ======
    parser.add_argument(
        "--ablate_ligand_llm",
        action="store_true",
        help="是否对配体 embedding 做 LLM 消融（在 process_data 中对整条向量操作）",
    )
    parser.add_argument(
        "--ablate_target_llm",
        action="store_true",
        help="是否对靶标 embedding 做 LLM 消融",
    )
    parser.add_argument(
        "--llm_ablation_mode",
        choices=["zero", "random", "shuffle"],
        default="zero",
        help="LLM 消融方式：zero=置零；random=随机噪声；shuffle=打乱节点-向量对应",
    )

    # 训练
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=None, help="兼容别名，若提供则覆盖 --wd")
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--lr_period", type=int, default=30)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument("--use_cosine", action="store_true")
    parser.add_argument("--cosine_T0", type=int, default=10)
    parser.add_argument("--cosine_Tmult", type=int, default=2)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--val_every", type=int, default=3)

    # InfoNCE
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--top_m", type=int, default=0)

    # 输入投影 MLP 配置（已在 model_gated.py 内读取）
    parser.add_argument("--proj_hidden_mult", type=int, default=4)
    parser.add_argument("--proj_dropout", type=float, default=0.2)

    # EMA（新增）
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)

    # 其它
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--monitor_metric", choices=["auprc","auroc"], default="auprc")
    parser.add_argument("--print_model", action="store_true")

    # CVS1 评估口径（默认“只删正向、保留反向”）
    parser.add_argument("--cv1_eval", choices=["strict","keep_reverse","legacy"], default="keep_reverse",
                        help="CVS1 评估口径：strict=删正向+反向；keep_reverse=只删正向(默认)；legacy=不删。")

    # ====== 图增强开关（保持你的方案 B） ======
    parser.add_argument("--augment_sim_loops", action="store_true",
                        help="为 is/ts 同型边补自环并写 sim_deg_orig（默认关闭）")
    parser.add_argument("--augment_cv1", action="store_true",
                        help="CVS1 下也执行同型增强（默认不在 CVS1 执行）")

    args = parser.parse_args()
    if args.weight_decay is not None:
        args.wd = args.weight_decay

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 构图 + 特征（此处会根据 ablate_xxx_llm / llm_ablation_mode 在 utlis.process_data 里做消融）
    edges, is_edges, ts_edges, initial_features = process_data(args)
    hetero_graph, rel_list = build_graph(args, edges, is_edges, ts_edges, initial_features, device)

    # === 轻量“图增强”：同型相似边补自环（可控开关，默认 CVS1 不增强） ===
    do_aug = False
    if getattr(args, "augment_sim_loops", False):
        if args.cv_mode.upper() != "CVS1" or getattr(args, "augment_cv1", False):
            do_aug = True

    if do_aug:
        if augment_similarity_graph is not None:
            try:
                hetero_graph = augment_similarity_graph(hetero_graph)
            except Exception as _e:
                print(f"[augment] fallback due to: {type(_e).__name__}: {_e}")
                hetero_graph = _augment_similarity_graph_fallback(hetero_graph)
        else:
            hetero_graph = _augment_similarity_graph_fallback(hetero_graph)

    train(args, hetero_graph, rel_list, device)
