# -*- coding: utf-8 -*-
"""
Stage-A: Relation-similarity Graph Contrastive Pretraining with EMA teacher
- Drug (ingredient) & Target (protein) embeddings -> projection heads -> aligned space
- GraphCL_drug + GraphCL_prot (neighbors as multi-positives), EMA teacher
- Optional: cross-modal alignment (drug-target edges) and multi-view (SMILES) hooks reserved
"""

import os, argparse, math, time
from typing import Tuple, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------
# Utils: robust loading
# --------------------------
def load_torch_tensor(path: str) -> torch.Tensor:
    """Robust loader for .pt: accept Tensor / {'embeddings': Tensor} / state_dict-like."""
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj.float().contiguous()
    if isinstance(obj, dict):
        # common keys
        for k in ["embeddings", "embedding", "tensor", "feat", "features"]:
            if k in obj and isinstance(obj[k], torch.Tensor):
                return obj[k].float().contiguous()
        # state_dict like?
        for k, v in obj.items():
            if isinstance(v, torch.Tensor) and v.ndim >= 2:
                return v.float().contiguous()
    raise ValueError(f"Unrecognized .pt format at {path}")

def load_npz_array(path: str, key: Optional[str]) -> torch.Tensor:
    with np.load(path, allow_pickle=True) as z:
        if key is None:
            # pick the largest 2D array
            cand = [(k, v) for k, v in z.items() if isinstance(v, np.ndarray) and v.ndim >= 2]
            if not cand: raise ValueError("No 2D array found in npz.")
            k = max(cand, key=lambda kv: kv[1].size)[0]
            arr = z[k]
        else:
            if key not in z: raise KeyError(f"Key {key} not in {list(z.keys())}")
            arr = z[key]
    arr = np.asarray(arr)
    if arr.ndim == 1: arr = arr[:, None]
    return torch.tensor(arr, dtype=torch.float32).contiguous()

def pad_or_trunc(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Unify feature dim (pad 0 or truncate)."""
    if x.size(1) == dim:
        return x.contiguous()
    if x.size(1) > dim:
        return x[:, :dim].contiguous()
    pad = torch.zeros(x.size(0), dim - x.size(1), dtype=x.dtype)
    return torch.cat([x, pad], dim=1).contiguous()

# --------------------------
# Similarity -> neighbors
# --------------------------
def read_similarity(path: str, n: Optional[int]=None) -> np.ndarray:
    """
    Read similarity either as:
      - square matrix (NxN)
      - edge list with 3 columns: i j sim
    Return a dense NxN float32 matrix.
    """
    arr = np.loadtxt(path, dtype=float)
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        M = arr.astype(np.float32)
    elif arr.ndim == 2 and arr.shape[1] >= 3:
        if n is None:
            n = int(arr[:, :2].max()) + 1
        M = np.zeros((n, n), dtype=np.float32)
        i = arr[:, 0].astype(int); j = arr[:, 1].astype(int); s = arr[:, 2].astype(np.float32)
        M[i, j] = np.maximum(M[i, j], s)
        M[j, i] = np.maximum(M[j, i], s)
    else:
        raise ValueError(f"Bad similarity file {path}")
    # clip & zero diag
    np.fill_diagonal(M, 0.0)
    M = np.clip(M, 0.0, None)
    return M

def build_neighbors_from_sim(M: np.ndarray, topk: int = 20, thresh: float = 0.0) -> List[np.ndarray]:
    """
    For each row, pick neighbors by:
      1) threshold (>= thresh), then
      2) top-k (exclude self)
    Return list of index arrays (pos neighbors).
    """
    N = M.shape[0]
    neigh = []
    for i in range(N):
        sims = M[i]
        cand = np.where(sims >= thresh)[0]
        cand = cand[cand != i]
        if cand.size > topk:
            idx = np.argpartition(-sims[cand], kth=topk-1)[:topk]
            cand = cand[idx]
        neigh.append(np.asarray(cand, dtype=np.int64))
    return neigh

# --------------------------
# Losses
# --------------------------
def mp_infonce_student_vs_teacher(
    z_s: torch.Tensor,            # [B, D] student
    z_t_all: torch.Tensor,        # [N, D] teacher (ALL nodes)
    pos_index: List[np.ndarray],  # list of arrays(len=B), each array = positive ids in ALL set
    batch_ids: np.ndarray,        # [B] the global ids of batch anchors
    tau: float = 0.07,
) -> torch.Tensor:
    """
    Multi-Positive InfoNCE: for each anchor i (student),
    positives are teacher embeddings of neighbor ids (from relation graph).
    Negatives are all others in teacher table.
    """
    device = z_s.device
    # cosine sim
    z_s = F.normalize(z_s, dim=-1)
    z_t_all = F.normalize(z_t_all, dim=-1)
    # logits: [B, N]
    logits = (z_s @ z_t_all.t()) / tau
    # numerical stability: logsumexp
    logsumexp_all = torch.logsumexp(logits, dim=1)                    # [B]
    # mask positives
    B, N = logits.shape
    pos_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    for bi, gid in enumerate(batch_ids):
        # <--- MODIFIED: Check if gid is within bounds of pos_index
        if gid >= len(pos_index):
            # This can happen if edges.txt has IDs larger than similarity matrix
            continue 
        pos = pos_index[gid]  # np array of neighbor ids
        if pos.size > 0:
            # <--- MODIFIED: Ensure positive indices are within bounds of the teacher table
            pos = pos[pos < N] 
            if pos.size > 0:
                pos_mask[bi, torch.as_tensor(pos, dtype=torch.long, device=device)] = True
                
    pos_logits = torch.where(pos_mask, logits, torch.full_like(logits, -1e9))
    logsumexp_pos = torch.logsumexp(pos_logits, dim=1)                 # [B]
    # loss per sample (if no pos -> skip)
    has_pos = pos_mask.any(dim=1)
    if not torch.any(has_pos):
        return torch.tensor(0.0, device=device, requires_grad=True)
    loss = -(logsumexp_pos[has_pos] - logsumexp_all[has_pos]).mean()
    return loss


# --------------------------
# EMA teacher
# --------------------------
def ema_update(student: nn.Module, teacher: nn.Module, m: float):
    with torch.no_grad():
        for ps, pt in zip(student.parameters(), teacher.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=(1.0 - m))


# --------------------------
# Model components
# --------------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


# --------------------------
# Main
# --------------------------
def build_args():
    p = argparse.ArgumentParser("Stage-A Relation Graph Contrastive (EMA) for ESM & Molecule embeddings")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--ligand_embed", type=str, required=True)
    p.add_argument("--target_embed", type=str, required=True)
    p.add_argument("--target_npz_key", type=str, default=None)

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--proj_dim", type=int, default=512)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--ema_m", type=float, default=0.999)

    p.add_argument("--rel_topk", type=int, default=20)
    p.add_argument("--rel_thresh", type=float, default=0.0)

    # <--- MODIFIED: Removed --use_xmod and --xmod_tau
    # <--- MODIFIED: Added --xmod_weight to balance losses
    p.add_argument("--xmod_weight", type=float, default=1.0, 
                     help="Weight for the cross-modal alignment loss (default: 1.0)")

    # (optional) multi-view for molecules (e.g., SMILES embedding path); leave for later
    p.add_argument("--smiles_embed", type=str, default=None)

    p.add_argument("--save_dir", type=str, default="./stageA_ckpts")
    return p

def main():
    args = build_args().parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # 1) load base embeddings
    Z_lig = load_torch_tensor(args.ligand_embed)      # [N_d, D_l]
    Z_tgt = load_npz_array(args.target_embed, args.target_npz_key)  # [N_p, D_p]
    N_d, d_l = Z_lig.shape
    N_p, d_p = Z_tgt.shape
    print(f"Loaded {N_d} ligands (D={d_l}) and {N_p} targets (D={d_p})") # <--- MODIFIED: Added print

    # 2) read similarity -> neighbors
    path_d_sim = os.path.join(args.data_dir, "ingredient_similarity.txt")
    path_p_sim = os.path.join(args.data_dir, "target_similarity.txt")
    M_d = read_similarity(path_d_sim, n=N_d)   # [N_d, N_d]
    M_p = read_similarity(path_p_sim, n=N_p)   # [N_p, N_p]
    neigh_d = build_neighbors_from_sim(M_d, topk=args.rel_topk, thresh=args.rel_thresh)
    neigh_p = build_neighbors_from_sim(M_p, topk=args.rel_topk, thresh=args.rel_thresh)

    # 3) projection heads (student & teacher)
    f_d = ProjectionHead(d_l, args.proj_dim).to(device)
    f_p = ProjectionHead(d_p, args.proj_dim).to(device)
    g_d = ProjectionHead(d_l, args.proj_dim).to(device)  # teacher drug
    g_p = ProjectionHead(d_p, args.proj_dim).to(device)  # teacher prot
    # init teacher = student
    g_d.load_state_dict(f_d.state_dict())
    g_p.load_state_dict(f_p.state_dict())
    for m in [g_d, g_p]:
        for p_ in m.parameters():
            p_.requires_grad_(False)

    opt = torch.optim.AdamW(list(f_d.parameters()) + list(f_p.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)

    # cache base embeddings on device (optional)
    Z_lig = Z_lig.to(device)
    Z_tgt = Z_tgt.to(device)

    # pre-build index arrays
    drug_ids = np.arange(N_d, dtype=np.int64)
    prot_ids = np.arange(N_p, dtype=np.int64)

    # <--- MODIFIED: Load cross-modal edges (required for alignment)
    edges_pos = None
    path_edges = os.path.join(args.data_dir, "edges.txt")
    if not os.path.exists(path_edges):
        print(f"[Warning] edges.txt not found in {args.data_dir}. Cross-modal alignment loss will be zero.")
        # Initialize empty neighbor lists
        neigh_d_p = [np.array([], dtype=np.int64) for _ in range(N_d)]
        neigh_p_d = [np.array([], dtype=np.int64) for _ in range(N_p)]
    else:
        E = []
        with open(path_edges, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip(): continue
                sp = line.strip().split()
                if len(sp) < 2: continue
                u, v = int(float(sp[0])), int(float(sp[1]))
                E.append((u, v))
        
        if not E:
            print("[Warning] edges.txt is empty. Cross-modal alignment loss will be zero.")
            neigh_d_p = [np.array([], dtype=np.int64) for _ in range(N_d)]
            neigh_p_d = [np.array([], dtype=np.int64) for _ in range(N_p)]
        else:
            E = np.array(E, dtype=np.int64)
            # auto-detect 1-based
            if E[:,0].max() >= N_d or E[:,1].max() >= N_p:
                print("[XMOD] Auto-detected 1-based indexing, converting to 0-based.")
                E[:,0] -= 1; E[:,1] -= 1
            
            # Ensure all indices are valid
            E = E[(E[:,0] < N_d) & (E[:,1] < N_p) & (E[:,0] >= 0) & (E[:,1] >= 0)]
            edges_pos = E
            print(f"[XMOD] loaded {len(E)} positive pairs from edges.txt")

            # Build cross-modal neighbor lists
            neigh_d_p = [[] for _ in range(N_d)] # drug_id -> list[prot_ids]
            neigh_p_d = [[] for _ in range(N_p)] # prot_id -> list[drug_ids]
            for u, v in edges_pos:
                neigh_d_p[u].append(v)
                neigh_p_d[v].append(u)
            # Convert to numpy arrays like neigh_d/neigh_p
            neigh_d_p = [np.array(x, dtype=np.int64) for x in neigh_d_p]
            neigh_p_d = [np.array(x, dtype=np.int64) for x in neigh_p_d]


    # <--- MODIFIED: Removed the inefficient xmod_loss function
    
    # ---- training loop ----
    print(f"Starting training for {args.epochs} epochs... (Device: {device})")
    for epoch in range(1, args.epochs + 1):
        f_d.train(); f_p.train()
        t0 = time.time()

        # 4) build teacher tables for ALL nodes（无梯度）
        with torch.no_grad():
            g_d.eval(); g_p.eval()
            Z_d_teacher = g_d(Z_lig).detach()  # [N_d, D]
            Z_p_teacher = g_p(Z_tgt).detach()  # [N_p, D]

        # 5) mini-batch over drugs & proteins separately
        #    为了高效，这里把两类批次迭代绑在一起（各自随机打乱）
        np.random.shuffle(drug_ids); np.random.shuffle(prot_ids)
        num_batches = max(math.ceil(N_d / args.batch_size), math.ceil(N_p / args.batch_size))
        total_loss, nsteps = 0.0, 0
        # <--- MODIFIED: Added trackers for loss components
        total_loss_gcl, total_loss_xmod = 0.0, 0.0


        for b in range(num_batches):
            b_drug = drug_ids[b*args.batch_size : (b+1)*args.batch_size]
            b_prot = prot_ids[b*args.batch_size : (b+1)*args.batch_size]

            # Skip empty batches
            if b_drug.size == 0 and b_prot.size == 0:
                continue

            # --- Intra-modal GraphCL loss ---
            loss_d = torch.tensor(0.0, device=device)
            loss_p = torch.tensor(0.0, device=device)
            if b_drug.size > 0:
                bd = torch.as_tensor(b_drug, device=device, dtype=torch.long)
                z_d_s = f_d(Z_lig[bd])    # [Bd, D]
                loss_d = mp_infonce_student_vs_teacher(
                    z_s=z_d_s, z_t_all=Z_d_teacher,
                    pos_index=neigh_d, batch_ids=b_drug, tau=args.tau
                )
            if b_prot.size > 0:
                bp = torch.as_tensor(b_prot, device=device, dtype=torch.long)
                z_p_s = f_p(Z_tgt[bp])    # [Bp, D]
                loss_p = mp_infonce_student_vs_teacher(
                    z_s=z_p_s, z_t_all=Z_p_teacher,
                    pos_index=neigh_p, batch_ids=b_prot, tau=args.tau
                )
            
            loss_gcl = loss_d + loss_p

            # <--- MODIFIED: NEW Cross-Modal Alignment Loss ---
            # Re-use the student embeddings (z_d_s, z_p_s) from above
            loss_d_x = torch.tensor(0.0, device=device)
            loss_p_x = torch.tensor(0.0, device=device)
            if b_drug.size > 0:
                loss_d_x = mp_infonce_student_vs_teacher(
                    z_s=z_d_s, z_t_all=Z_p_teacher,  # drug student vs. PROT teacher
                    pos_index=neigh_d_p,            # drug -> prot positives
                    batch_ids=b_drug, tau=args.tau  # Use same tau
                )
            if b_prot.size > 0:
                loss_p_x = mp_infonce_student_vs_teacher(
                    z_s=z_p_s, z_t_all=Z_d_teacher,  # prot student vs. DRUG teacher
                    pos_index=neigh_p_d,            # prot -> drug positives
                    batch_ids=b_prot, tau=args.tau
                )
            
            loss_xmod = loss_d_x + loss_p_x
            
            # <--- MODIFIED: Combined loss with weighting
            loss = loss_gcl + args.xmod_weight * loss_xmod

            # <--- MODIFIED: Removed the old inefficient --use_xmod block

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(f_d.parameters()) + list(f_p.parameters()), max_norm=5.0)
            opt.step()

            # EMA update
            ema_update(f_d, g_d, m=args.ema_m)
            ema_update(f_p, g_p, m=args.ema_m)

            total_loss += float(loss.detach().cpu())
            total_loss_gcl += float(loss_gcl.detach().cpu()) # <--- MODIFIED
            total_loss_xmod += float(loss_xmod.detach().cpu()) # <--- MODIFIED
            nsteps += 1

        dt = time.time() - t0
        avg_loss = total_loss / max(1, nsteps)
        avg_gcl = total_loss_gcl / max(1, nsteps)
        avg_xmod = total_loss_xmod / max(1, nsteps)
        # <--- MODIFIED: Improved print statement
        print(f"[Epoch {epoch:03d}] loss={avg_loss:.4f} (GCL={avg_gcl:.4f}, XMOD={avg_xmod:.4f})  time={dt:.1f}s")


        # 保存 checkpoint（每若干轮）
        if epoch % 10 == 0 or epoch == args.epochs:
            torch.save(f_d.state_dict(), os.path.join(args.save_dir, "proj_ligand.pt"))
            torch.save(f_p.state_dict(), os.path.join(args.save_dir, "proj_target.pt"))
            # also save full heads for inference convenience
            torch.save({"proj": f_d.state_dict(), "in_dim": d_l, "proj_dim": args.proj_dim},
                       os.path.join(args.save_dir, "head_ligand.pt"))
            torch.save({"proj": f_p.state_dict(), "in_dim": d_p, "proj_dim": args.proj_dim},
                       os.path.join(args.save_dir, "head_target.pt"))

    # 导出全库对齐后的向量（学生头）
    with torch.no_grad():
        f_d.eval(); f_p.eval()
        Z_lig_aligned = f_d(Z_lig).cpu()
        Z_tgt_aligned = f_p(Z_tgt).cpu()
    torch.save(Z_lig_aligned, os.path.join(args.save_dir, "aligned_ligand.pt"))
    torch.save(Z_tgt_aligned, os.path.join(args.save_dir, "aligned_target.pt"))
    print("[Done] saved projection heads and aligned embeddings.")

if __name__ == "__main__":
    main()
