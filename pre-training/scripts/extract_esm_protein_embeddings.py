# -*- coding: utf-8 -*-
"""
embeding_protein.py
一条龙：抽取 ESM 向量 -> 聚合重复 ID -> （可选）降维到 320 -> L2 归一化 ->
对齐“全量 target 列表” -> 存 npz（键：composite_embedding）
并内置：
- AMP 半精度（--no_amp 可关闭，使用 torch.amp.autocast）
- 动态分批：按 token 预算（--batch_tokens）+ 注意力 L^2 预算（--attn_budget）防 OOM
- 超长序列分块（>1000 aa 自动滑窗编码，长度加权平均）
- 断点续跑：--resume_pt 指向已部分抽取的 .pt；每 --save_every 条增量保存到 --pt_out

输出：
- .npz：{ composite_embedding: (N,D), node_id: [N] }   —— 训练直接用
- .pt ：{ ids: [...], embeddings: torch.Tensor[M,D] } —— 原始抽取结果（M 条）
- .csv（可选）：对齐诊断（是否命中、聚合数量）
"""

import os
import re
import csv
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

# -----------------------
# Utilities
# -----------------------
AA = set("ACDEFGHIKLMNPQRSTVWY")

def clean_seq(seq: str) -> str:
    s = re.sub(r"[\s\*\-\.]", "", str(seq).upper())
    return "".join(ch if ch in AA else "X" for ch in s)

def read_excel_or_csv(path, id_col, seq_col):
    import pandas as pd
    if str(path).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    if id_col not in df or seq_col not in df:
        raise KeyError(f"列不存在：{id_col} / {seq_col}")
    ids = [str(x) for x in df[id_col].astype(str).tolist()]
    seqs = [clean_seq(x) for x in df[seq_col].astype(str).tolist()]
    return ids, seqs

def read_fasta(path):
    ids, seqs = [], []
    with open(path, "r", encoding="utf-8") as f:
        cur_id, cur_seq = None, []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id is not None:
                    ids.append(cur_id)
                    seqs.append(clean_seq("".join(cur_seq)))
                cur_id = line[1:].split()[0]
                cur_seq = []
            else:
                cur_seq.append(line)
        if cur_id is not None:
            ids.append(cur_id)
            seqs.append(clean_seq("".join(cur_seq)))
    return ids, seqs

def load_pt(pt_path):
    obj = torch.load(pt_path, map_location="cpu")
    ids = [str(x) for x in obj["ids"]]
    emb = obj["embeddings"].float()
    return ids, emb

def save_pt(pt_path, ids, emb):
    pt_path = Path(pt_path)
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"ids": ids, "embeddings": emb.cpu()}, pt_path)

def aggregate_duplicates(ids, emb, how="mean"):
    """对重复 ID 聚合，不改名"""
    bag = defaultdict(list)
    for i, k in enumerate(ids):
        bag[k].append(i)
    uniq_ids, uniq_emb, dup_count = [], [], {}
    for k, idxs in bag.items():
        E = emb[idxs]
        if how == "mean":
            v = E.mean(0)
        elif how == "first":
            v = E[0]
        else:
            raise ValueError("how must be mean|first")
        uniq_ids.append(k)
        uniq_emb.append(v)
        dup_count[k] = len(idxs)
    uniq_emb = torch.stack(uniq_emb, 0)
    return uniq_ids, uniq_emb, dup_count

def pca_reduce(np_array, out_dim=320):
    if out_dim is None or out_dim <= 0 or np_array.shape[1] == out_dim:
        return np_array.astype(np.float32)
    # 优先 sklearn
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=out_dim, svd_solver="auto", random_state=0)
        Z = pca.fit_transform(np_array)
        return Z.astype(np.float32)
    except Exception:
        # 退化到 SVD
        X = np_array - np_array.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        W = Vt[:out_dim].T
        Z = X @ W
        return Z.astype(np.float32)

def align_to_master(ids, emb_np, master_ids):
    id2row = {k: i for i, k in enumerate(ids)}
    D = emb_np.shape[1]
    aligned = np.zeros((len(master_ids), D), dtype=np.float32)
    hit = 0
    hit_mask, src_row = [], []
    for i, mid in enumerate(master_ids):
        j = id2row.get(str(mid))
        if j is not None:
            aligned[i] = emb_np[j]
            hit += 1
            hit_mask.append(1)
            src_row.append(j)
        else:
            hit_mask.append(0)
            src_row.append(-1)
    cov = hit / max(1, len(master_ids))
    return aligned, cov, np.array(hit_mask, dtype=np.int32), np.array(src_row, dtype=np.int32)

# -----------------------
# Long-seq chunk encoder
# -----------------------
def encode_long_seq_avg(model, alphabet, seq, device, amp=True, window=1000, stride=900, layer=33):
    """
    把超长序列切成片段（重叠），分别跑表示，最后按片段长度加权平均。
    window: 每片有效AA长度（不含CLS/EOS），建议 <= 1000
    stride: 每次前进步长（900 表示 100 重叠）
    """
    batch_converter = alphabet.get_batch_converter()
    L = len(seq)
    if L <= window:
        _, _, tokens = batch_converter([("", seq)])
        tokens = tokens.to(device)
        with torch.no_grad():
            if amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    rep = model(tokens, repr_layers=[layer], return_contacts=False)["representations"][layer]
            else:
                rep = model(tokens, repr_layers=[layer], return_contacts=False)["representations"][layer]
        return rep[0, 1:1+L].mean(0).float().cpu()

    reps, weights = [], []
    start = 0
    while start < L:
        sub = seq[start:start+window]
        if not sub: break
        _, _, tokens = batch_converter([("", sub)])
        tokens = tokens.to(device)
        with torch.no_grad():
            if amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    rep = model(tokens, repr_layers=[layer], return_contacts=False)["representations"][layer]
            else:
                rep = model(tokens, repr_layers=[layer], return_contacts=False)["representations"][layer]
        reps.append(rep[0, 1:1+len(sub)].mean(0).float().cpu())
        weights.append(len(sub))
        if start + window >= L: break
        start += stride
    reps = torch.stack(reps, 0)
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.sum()
    return (w.unsqueeze(1) * reps).sum(0)

# -----------------------
# ESM embedding with AMP + dynamic batching + resume
# -----------------------
def esm_embed_stream(
    ids, seqs, *,
    esm_name="esm2_t33_650M_UR50D",
    batch_tokens=4000,
    attn_budget=1_000_000,
    device="cuda:0",
    amp=True,
    save_every=200,
    pt_out=None,
    resume_ids=None,
    resume_emb=None,
):
    """
    动态分批（token + L^2 预算），AMP 半精度，支持断点续跑。
    - ids/seqs: 全量输入
    - resume_ids/resume_emb: 若提供，则跳过已完成部分并在其后追加
    - 每处理 save_every 条，若 pt_out 非空，则增量落盘一次
    返回：all_ids, all_emb (torch.Tensor)
    """
    try:
        import esm
    except ModuleNotFoundError:
        raise RuntimeError("未安装 fair-esm，请先 pip install fair-esm")

    model, alphabet = getattr(esm.pretrained, esm_name)()
    model.eval().to(device)
    if amp:
        model.half()
    batch_converter = alphabet.get_batch_converter()

    # 恢复进度
    start_idx = 0
    out_ids = []
    out_chunks = []

    if resume_ids is not None and resume_emb is not None:
        done = len(resume_ids)
        out_ids.extend(resume_ids)
        out_chunks.append(resume_emb.cpu())
        start_idx = done
        print(f"[Resume] detected {done} items, resume from index {start_idx}")

    entries = [(i, ids[i], seqs[i]) for i in range(len(ids))]
    ptr = start_idx
    processed_since_save = 0

    while ptr < len(entries):
        cur, L2sum, toksum = [], 0, 0
        # 装批：优先满足 L^2 预算，其次 token 预算；至少放 1 条
        while ptr < len(entries):
            _, _id, _seq = entries[ptr]
            L = len(_seq)
            need_single = (L * L > attn_budget)
            if need_single and len(cur) > 0:
                break
            if ((L2sum + L * L) <= attn_budget and (toksum + L + 2) <= max(1, batch_tokens)) or len(cur) == 0:
                cur.append((_id, _seq))
                L2sum += L * L
                toksum += L + 2
                ptr += 1
                if need_single:
                    break
            else:
                break

        seqs_cur = [s for (_, s) in cur]
        batch = [("", s) for s in seqs_cur]
        _, _, tokens = batch_converter(batch)
        tokens = tokens.to(device)

        with torch.no_grad():
            try:
                if amp:
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        rep = model(tokens, repr_layers=[33], return_contacts=False)["representations"][33]
                else:
                    rep = model(tokens, repr_layers=[33], return_contacts=False)["representations"][33]
                reps_mean = []
                for bi, s in enumerate(seqs_cur):
                    L = len(s)
                    if L > 1000:
                        reps_mean.append(encode_long_seq_avg(model, alphabet, s, device, amp=amp, layer=33))
                    else:
                        reps_mean.append(rep[bi, 1:1+L].mean(0).float().cpu())
                chunk = torch.stack(reps_mean, 0)
                out_chunks.append(chunk)
                out_ids.extend([_id for (_id, _) in cur])
                processed_since_save += len(cur)

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                # 批里>1条：把最后一条留到下一批
                if len(seqs_cur) > 1:
                    ptr -= 1
                    cur.pop()
                    print("[ESM] OOM -> fallback: pop last sample, retry smaller batch")
                    continue
                # 批里==1条：对该超长序列走分块编码（不要改 tokens dtype）
                print("[ESM] OOM -> fallback: encode single long sequence by chunks")
                vec = encode_long_seq_avg(model, alphabet, seqs_cur[0], device, amp=False, layer=33)
                chunk = vec.unsqueeze(0)
                out_chunks.append(chunk)
                out_ids.extend([cur[0][0]])
                processed_since_save += 1

        print(f"[ESM] processed {len(out_ids)}/{len(ids)} (L2sum={L2sum}, tokens~{toksum})")
        torch.cuda.empty_cache()

        # 增量保存
        if pt_out and processed_since_save >= save_every:
            save_pt(pt_out, out_ids, torch.cat(out_chunks, 0))
            print(f"[ESM] checkpoint saved -> {pt_out}  (count={len(out_ids)})")
            processed_since_save = 0

    all_emb = torch.cat(out_chunks, 0) if out_chunks else torch.empty(0, 0)
    assert len(out_ids) == all_emb.shape[0], "ids/emb 数量不一致"
    # 结束时再保存一次
    if pt_out:
        save_pt(pt_out, out_ids, all_emb)
        print(f"[ESM] final saved -> {pt_out}")
    return out_ids, all_emb

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["excel", "csv", "fasta", "pt"], required=True)
    ap.add_argument("--excel", help="Excel/CSV 路径（source=excel/csv）")
    ap.add_argument("--id_col", default="Node ID")
    ap.add_argument("--seq_col", default="Sequence")
    ap.add_argument("--fasta", help="FASTA 路径（source=fasta）")
    ap.add_argument("--pt_in", help="已有 .pt（source=pt）")
    ap.add_argument("--pt_out", help="抽取过程/结果保存 .pt（断点续跑会写入）")

    # ESM 模型与显存预算
    ap.add_argument("--esm", default="esm2_t33_650M_UR50D")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch_tokens", type=int, default=1200, help="每批 token 线性预算（调小可以防 OOM）")
    ap.add_argument("--attn_budget", type=int, default=800000, help="注意力 L^2 预算（调小会更保守）")
    ap.add_argument("--no_amp", action="store_true", help="关闭 AMP 半精度")
    ap.add_argument("--save_every", type=int, default=200, help="每处理多少条保存一次 .pt（断点续跑）")
    ap.add_argument("--resume_pt", default=None, help="从该 .pt 断点续跑（ids/embeddings）")

    # 对齐到“全量 target 列表”
    ap.add_argument("--master", required=True, help="全量 target 列表（xlsx/csv/txt）")
    ap.add_argument("--master_id_col", default="Node ID")

    # 降维
    ap.add_argument("--reduce_dim", type=int, default=320, help="降到该维度；<=0 不降维")

    # 输出
    ap.add_argument("--npz_out", required=True, help="输出 npz 路径（含 composite_embedding/node_id）")
    ap.add_argument("--report_csv", default=None, help="诊断 CSV（每个 ID 是否命中、聚合数量）")

    args = ap.parse_args()
    amp = not args.no_amp
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1) 读取上游
    if args.source in ("excel", "csv"):
        if not args.excel:
            ap.error("--excel 必填")
        ids, seqs = read_excel_or_csv(args.excel, args.id_col, args.seq_col)
        # 断点加载（可选）
        resume_ids = resume_emb = None
        if args.resume_pt and Path(args.resume_pt).exists():
            resume_ids, resume_emb = load_pt(args.resume_pt)
        all_ids, all_emb = esm_embed_stream(
            ids, seqs,
            esm_name=args.esm,
            batch_tokens=args.batch_tokens,
            attn_budget=args.attn_budget,
            device=args.device,
            amp=amp,
            save_every=args.save_every,
            pt_out=args.pt_out,
            resume_ids=resume_ids,
            resume_emb=resume_emb,
        )
    elif args.source == "fasta":
        if not args.fasta:
            ap.error("--fasta 必填")
        ids, seqs = read_fasta(args.fasta)
        resume_ids = resume_emb = None
        if args.resume_pt and Path(args.resume_pt).exists():
            resume_ids, resume_emb = load_pt(args.resume_pt)
        all_ids, all_emb = esm_embed_stream(
            ids, seqs,
            esm_name=args.esm,
            batch_tokens=args.batch_tokens,
            attn_budget=args.attn_budget,
            device=args.device,
            amp=amp,
            save_every=args.save_every,
            pt_out=args.pt_out,
            resume_ids=resume_ids,
            resume_emb=resume_emb,
        )
    elif args.source == "pt":
        if not args.pt_in:
            ap.error("--pt_in 必填（source=pt）")
        all_ids, all_emb = load_pt(args.pt_in)
    else:
        ap.error("未知 source")

    print(f"[Load] N={len(all_ids)}  D={all_emb.shape[1]}")

    # 2) 重复 ID 聚合
    uniq_ids, uniq_emb, dup_count = aggregate_duplicates(all_ids, all_emb, how="mean")
    print(f"[Dedup] unique={len(uniq_ids)} | duplicates_aggregated={sum(c-1 for c in dup_count.values())}")

    # 3) （可选）降维
    emb_np = uniq_emb.cpu().numpy().astype(np.float32)
    if args.reduce_dim and args.reduce_dim > 0 and emb_np.shape[1] != args.reduce_dim:
        print(f"[Reduce] {emb_np.shape[1]} -> {args.reduce_dim} (PCA/SVD)")
        emb_np = pca_reduce(emb_np, out_dim=args.reduce_dim)

    # 3.5) L2 归一化（每个蛋白向量除以自身范数，防止范数漂移影响打分）
    row_norm = np.linalg.norm(emb_np, axis=1, keepdims=True)
    row_norm[row_norm == 0] = 1.0
    emb_np = emb_np / row_norm

    # 4) 读“全量 target 列表”
    master_path = str(args.master).lower()
    if master_path.endswith((".xlsx", ".xls")):
        import pandas as pd
        dfm = pd.read_excel(args.master)
        master_ids = [str(x) for x in dfm[args.master_id_col].astype(str).tolist()]
    elif master_path.endswith(".csv"):
        import pandas as pd
        dfm = pd.read_csv(args.master)
        master_ids = [str(x) for x in dfm[args.master_id_col].astype(str).tolist()]
    else:
        master_ids = [line.strip() for line in open(args.master, "r", encoding="utf-8") if line.strip()]
    print(f"[Master] targets={len(master_ids)}")

    # 5) 对齐
    aligned, cov, hit_mask, src_row = align_to_master(uniq_ids, emb_np, master_ids)
    print(f"[Align] coverage={cov:.3f} | aligned shape={aligned.shape} | zeros={(hit_mask==0).sum()}")

    # 6) 保存 npz
    outp = Path(args.npz_out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(outp, composite_embedding=aligned.astype(np.float32), node_id=np.array(master_ids))
    print(f"[OK] wrote npz -> {outp}")

    # 7) 诊断报表（可选）
    if args.report_csv:
        rcsv = Path(args.report_csv)
        rcsv.parent.mkdir(parents=True, exist_ok=True)
        with open(rcsv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["row", "node_id", "hit", "source_row", "dup_aggregated_count"])
            for i, mid in enumerate(master_ids):
                cnt = dup_count.get(mid, 1)
                w.writerow([i, mid, int(hit_mask[i]), int(src_row[i]), cnt])
        print(f"[OK] wrote report -> {rcsv}")

if __name__ == "__main__":
    # 防止碎片化造成 OOM（PyTorch 建议）
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()

