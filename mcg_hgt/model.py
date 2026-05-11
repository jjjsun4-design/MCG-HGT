# -*- coding: utf-8 -*-
"""
model_gated.py — HGT + 输入门(SE/GLU) + 门控残差 + 打分头(GMU/FiLM) +
                 【关系级语义门】+【注意力头门】 + 【类型化输入MLP统一维度】
-------------------------------------------------------------------------------
在原始功能完全保留的基础上，新增/修正：
- 为每个节点类型添加一套两层 MLP（GELU + Dropout + LayerNorm），将原始特征
  统一到 in_dim；同构化时不再携带 features，改为在 forward 里拼接统一后的特征。
- HGTConv 的 heads 改为第三个位置参数（不要再用 num_heads=... 避免冲突）。
- 关系语义门的偏置设在最后一个 Linear，而不是 Sigmoid。
"""
from __future__ import annotations
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import function as fn
from dgl.nn.pytorch.conv import HGTConv


# -------------------------- small utils --------------------------

def _getattr(obj, name, default):
    return getattr(obj, name, default) if obj is not None else default


def _ensure_divisible(dim: int, heads: int) -> None:
    if dim % max(1, heads) != 0:
        raise ValueError(f"dim={dim} 必须能被 num_heads={heads} 整除 (HGTConv: out = head_size * heads)")


def _etype_key(etype: Tuple[str, str, str]) -> str:
    return f"{etype[0]}__{etype[1]}__{etype[2]}"


# -------------------------- 输入门控 --------------------------

class _SEGate(nn.Module):
    def __init__(self, dim: int, reduce: int = 4, bias_init: float = 1.0):
        super().__init__()
        hidden = max(1, dim // max(1, reduce))
        self.gate = nn.Sequential(
            nn.Linear(dim, hidden, bias=True), nn.ReLU(inplace=True),
            nn.Linear(hidden, dim, bias=True), nn.Sigmoid()
        )
        with torch.no_grad():
            nn.init.xavier_uniform_(self.gate[0].weight); nn.init.zeros_(self.gate[0].bias)
            nn.init.zeros_(self.gate[2].weight); nn.init.constant_(self.gate[2].bias, bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x) * x


class _GLUGate(nn.Module):
    def __init__(self, dim: int, bias_init: float = 1.0):
        super().__init__()
        self.Wa = nn.Linear(dim, dim, bias=True)
        self.Wb = nn.Linear(dim, dim, bias=True)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.Wa.weight); nn.init.zeros_(self.Wa.bias)
            nn.init.xavier_uniform_(self.Wb.weight); nn.init.constant_(self.Wb.bias, bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Wa(x) * torch.sigmoid(self.Wb(x))


# -------------------------- GMU / FiLM 打分 --------------------------

class _GMUScore(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj_u = nn.Linear(dim, dim, bias=False)
        self.proj_v = nn.Linear(dim, dim, bias=False)
        h = max(32, dim // 4)
        self.gate = nn.Sequential(nn.Linear(2 * dim, h), nn.ReLU(inplace=True), nn.Linear(h, 1), nn.Sigmoid())

    def forward(self, hu: torch.Tensor, hv: torch.Tensor) -> torch.Tensor:
        z = self.gate(torch.cat([hu, hv], dim=-1)).squeeze(-1)
        s = (self.proj_u(hu) * self.proj_v(hv)).sum(-1)
        return z * s


class _FiLMScore(nn.Module):
    def __init__(self, dim: int, condition: str = 'dst'):
        super().__init__()
        assert condition in ('dst', 'src')
        self.condition = condition
        h = max(32, dim // 4)
        self.cond_mlp = nn.Sequential(nn.Linear(dim, h), nn.ReLU(inplace=True), nn.Linear(h, 2 * dim))
        self.proj_u = nn.Linear(dim, dim, bias=False)
        self.proj_v = nn.Linear(dim, dim, bias=False)

    def forward(self, hu: torch.Tensor, hv: torch.Tensor) -> torch.Tensor:
        if self.condition == 'dst':
            gamma, beta = self.cond_mlp(hv).chunk(2, dim=-1)
            mod_u = gamma * self.proj_u(hu) + beta
            v = self.proj_v(hv)
            return (mod_u * v).sum(-1)
        else:
            gamma, beta = self.cond_mlp(hu).chunk(2, dim=-1)
            mod_v = gamma * self.proj_v(hv) + beta
            u = self.proj_u(hu)
            return (u * mod_v).sum(-1)


# -------------------------- 编码器（含注意力头门 + 类型化输入MLP） --------------------------

class _FullGraphHGTEncoder(nn.Module):
    def __init__(
        self,
        g_hetero: dgl.DGLHeteroGraph,
        in_dim: int, hid_dim: int, out_dim: int,
        num_layers: int = 2, heads: int = 4, dropout: float = 0.2,
        input_gate_type: str = 'none', input_gate_reduce: int = 4,
        residual_gate: bool = False, gate_bias: float = 1.0,
        head_gate: bool = False,
    ):
        super().__init__()
        self.g_hetero = g_hetero

        # === 同构化（不携带 feature） ===
        g_homo = dgl.to_homogeneous(g_hetero)
        try:
            g_homo = g_homo.to(g_hetero.device)
        except Exception:
            pass
        self.g_homo: dgl.DGLGraph = g_homo
        self.ntype = self.g_homo.ndata[dgl.NTYPE].clone()
        self.nid = self.g_homo.ndata[dgl.NID].clone()
        self.etype = self.g_homo.edata[dgl.ETYPE].clone()

        self.ntypes_order: List[str] = list(g_hetero.ntypes)
        self.ntype_name_to_id = {n: i for i, n in enumerate(self.ntypes_order)}
        self.num_nodes_by_type = {n: g_hetero.num_nodes(n) for n in self.ntypes_order}

        self.dropout = nn.Dropout(dropout)
        self.heads = heads
        self.head_gate = head_gate

        # === 每种节点类型一个输入投影 MLP（raw_dim -> 2*in_dim -> in_dim） ===
        self.in_dim = in_dim
        self.input_proj = nn.ModuleDict()
        hidden_mult = 2
        proj_dropout = 0.1
        hidden_dim = max(in_dim, hidden_mult * in_dim)
        for nt in self.ntypes_order:
            if 'features' not in g_hetero.nodes[nt].data:
                raise KeyError(f"节点类型 {nt} 缺少 'features'")
            d_in = int(g_hetero.nodes[nt].data['features'].shape[1])
            if d_in == in_dim:
                self.input_proj[nt] = nn.Identity()
            else:
                self.input_proj[nt] = nn.Sequential(
                    nn.Linear(d_in, hidden_dim, bias=True),
                    nn.GELU(),
                    nn.Dropout(proj_dropout),
                    nn.Linear(hidden_dim, in_dim, bias=True),
                    nn.LayerNorm(in_dim),
                )

        # 输入门控（可选）
        self.input_gate_type = input_gate_type
        self.input_gates = nn.ModuleDict()
        if input_gate_type in ('se', 'glu'):
            for nt in self.ntypes_order:
                self.input_gates[nt] = (_SEGate(in_dim, input_gate_reduce, gate_bias)
                                        if input_gate_type == 'se' else _GLUGate(in_dim, gate_bias))

        # HGT 堆栈 + 规范化 + 残差门 + 头门
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residual_gate = residual_gate
        if residual_gate:
            self.res_proj = nn.ModuleList()
            self.res_gate = nn.ModuleList()
        if head_gate:
            self.head_alphas = nn.ParameterList()

        _ensure_divisible(hid_dim, heads)
        _ensure_divisible(out_dim, heads)
        head_hid = hid_dim // heads
        head_out = out_dim // heads

        if num_layers == 1:
            _ensure_divisible(in_dim, heads)
            self.layers.append(HGTConv(in_dim, head_out, heads,
                                       num_ntypes=len(g_hetero.ntypes),
                                       num_etypes=len(g_hetero.canonical_etypes),
                                       dropout=dropout, use_norm=True))
            self.norms.append(nn.LayerNorm(out_dim))
            if residual_gate:
                self.res_proj.append(nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity())
                self.res_gate.append(nn.Linear(out_dim, out_dim, bias=True))
            if head_gate:
                self.head_alphas.append(nn.Parameter(torch.zeros(heads)))
        else:
            # 输入层
            _ensure_divisible(in_dim, heads)
            self.layers.append(HGTConv(in_dim, head_hid, heads,
                                       num_ntypes=len(g_hetero.ntypes),
                                       num_etypes=len(g_hetero.canonical_etypes),
                                       dropout=dropout, use_norm=True))
            self.norms.append(nn.LayerNorm(hid_dim))
            if residual_gate:
                self.res_proj.append(nn.Linear(in_dim, hid_dim, bias=False) if in_dim != hid_dim else nn.Identity())
                self.res_gate.append(nn.Linear(hid_dim, hid_dim, bias=True))
            if head_gate:
                self.head_alphas.append(nn.Parameter(torch.zeros(heads)))
            # 中间层
            for _ in range(num_layers - 2):
                self.layers.append(HGTConv(hid_dim, head_hid, heads,
                                           num_ntypes=len(g_hetero.ntypes),
                                           num_etypes=len(g_hetero.canonical_etypes),
                                           dropout=dropout, use_norm=True))
                self.norms.append(nn.LayerNorm(hid_dim))
                if residual_gate:
                    self.res_proj.append(nn.Identity())
                    self.res_gate.append(nn.Linear(hid_dim, hid_dim, bias=True))
                if head_gate:
                    self.head_alphas.append(nn.Parameter(torch.zeros(heads)))
            # 输出层
            self.layers.append(HGTConv(hid_dim, head_out, heads,
                                       num_ntypes=len(g_hetero.ntypes),
                                       num_etypes=len(g_hetero.canonical_etypes),
                                       dropout=dropout, use_norm=True))
            self.norms.append(nn.LayerNorm(out_dim))
            if residual_gate:
                self.res_proj.append(nn.Linear(hid_dim, out_dim, bias=False) if hid_dim != out_dim else nn.Identity())
                self.res_gate.append(nn.Linear(out_dim, out_dim, bias=True))
            if head_gate:
                self.head_alphas.append(nn.Parameter(torch.zeros(heads)))

        if residual_gate:
            with torch.no_grad():
                for g in self.res_gate:
                    nn.init.zeros_(g.weight); nn.init.constant_(g.bias, gate_bias)

    # ---------- helpers ----------

    def _apply_input_gate(self, h: torch.Tensor) -> torch.Tensor:
        if self.input_gate_type not in ('se', 'glu'):
            return h
        out = h.clone()
        for nt in self.ntypes_order:
            mask = (self.ntype == self.ntype_name_to_id[nt])
            if mask.any():
                out[mask] = self.input_gates[nt](h[mask])
        return out

    def _apply_head_gate(self, y: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if not self.head_gate:
            return y
        H = self.heads
        if y.dim() != 2 or y.size(-1) % H != 0:
            return y
        dh = y.size(-1) // H
        y3 = y.view(y.size(0), H, dh)
        alpha = torch.sigmoid(self.head_alphas[layer_idx])  # [H]
        y3 = y3 * alpha.view(1, H, 1)
        return y3.reshape(y.size(0), H * dh)

    def _build_homogeneous_features(self) -> torch.Tensor:
        """将各类型 raw 特征用各自 MLP 投到 in_dim，并按同构顺序拼接为 h。"""
        device = self.g_homo.device
        N = self.g_homo.num_nodes()
        h = torch.zeros(N, self.in_dim, device=device)
        for nt in self.ntypes_order:
            mask = (self.ntype == self.ntype_name_to_id[nt])
            if not mask.any():
                continue
            raw = self.g_hetero.nodes[nt].data['features'].detach().to(device)   # [N_nt, d_raw_nt]
            proj = self.input_proj[nt](raw)                                      # [N_nt, in_dim]
            idx_local = self.nid[mask].long()                                     # homo 位置 -> 该类型局部 id
            h[mask] = proj[idx_local]
        return h

    # ---------- forward ----------

    def forward(self) -> Dict[str, torch.Tensor]:
        # 1) 统一维度的 h
        h = self._build_homogeneous_features()

        # 2) 输入门控
        h = self._apply_input_gate(h)

        # 3) HGT 堆栈 + 头门 +（可选）门控残差
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            y = layer(self.g_homo, h, self.ntype, self.etype, presorted=True)
            y = self._apply_head_gate(y, i)
            if self.residual_gate:
                x = self.res_proj[i](h)
                g = torch.sigmoid(self.res_gate[i](y))
                y = g * y + (1.0 - g) * x
            h = self.dropout(F.gelu(norm(y)))

        # 4) 还原到各类型完整矩阵
        z_dict: Dict[str, torch.Tensor] = {}
        for nt_name in self.ntypes_order:
            t_id = self.ntype_name_to_id[nt_name]
            mask = (self.ntype == t_id)
            h_t = h[mask]
            nid_t = self.nid[mask].long()
            full = torch.zeros(self.num_nodes_by_type[nt_name], h_t.size(-1), device=h_t.device, dtype=h_t.dtype)
            full[nid_t] = h_t
            z_dict[nt_name] = full
        return z_dict


# -------------------------- 打分头（含关系级语义门） --------------------------

class ScorePredictor(nn.Module):
    def __init__(self, out_dim: int, g_hetero: dgl.DGLHeteroGraph,
                 score_gate: str = 'none', film_condition: str = 'dst',
                 semantic_gate: str = 'none', sem_hidden: int = 64, sem_gate_bias: float = 0.0):
        super().__init__()
        self.score_gate = score_gate
        self.semantic_gate = semantic_gate  # 'none' or 'etype'
        self.etypes = list(g_hetero.canonical_etypes)

        if score_gate == 'gmu':
            self.scorer = _GMUScore(out_dim)
        elif score_gate == 'film':
            self.scorer = _FiLMScore(out_dim, condition=film_condition)
        else:
            self.scorer = None

        if self.semantic_gate == 'etype':
            self.rel_gate = nn.ModuleDict()
            for et in self.etypes:
                key = _etype_key(et)
                mlp = nn.Sequential(
                    nn.Linear(2 * out_dim, sem_hidden), nn.ReLU(inplace=True),
                    nn.Linear(sem_hidden, 1), nn.Sigmoid()
                )
                # 设定「最后一个 Linear」的偏置，不是 Sigmoid
                with torch.no_grad():
                    nn.init.zeros_(mlp[2].weight)
                    nn.init.constant_(mlp[2].bias, sem_gate_bias)
                self.rel_gate[key] = mlp
        else:
            self.rel_gate = None

    def _base_score(self, hu: torch.Tensor, hv: torch.Tensor) -> torch.Tensor:
        if self.scorer is None:
            return (hu * hv).sum(-1)  # u·v
        return self.scorer(hu, hv)

    def forward(self, edge_subgraph, h_dict: Dict[str, torch.Tensor]) -> Dict[Tuple[str, str, str], torch.Tensor]:
        with edge_subgraph.local_scope():
            # 将 full-graph 表征切到子图顺序
            for ntype, h_full in h_dict.items():
                n_sub = edge_subgraph.num_nodes(ntype)
                if n_sub == 0:
                    continue
                if dgl.NID in edge_subgraph.nodes[ntype].data:
                    idx = edge_subgraph.nodes[ntype].data[dgl.NID].long().to(h_full.device)
                    h_sub = h_full[idx]
                else:
                    if h_full.shape[0] != n_sub:
                        raise RuntimeError(
                            f"Missing dgl.NID for node type {ntype}, and size mismatch: full={h_full.shape[0]} vs sub={n_sub}")
                    h_sub = h_full
                edge_subgraph.nodes[ntype].data['h'] = h_sub

            scores = {}

            def _edge_udf_factory(etype_key: str):
                def _edge_udf(edges):
                    hu, hv = edges.src['h'], edges.dst['h']
                    s = self._base_score(hu, hv)
                    if self.semantic_gate == 'etype':
                        g = self.rel_gate[etype_key](torch.cat([hu, hv], dim=-1)).squeeze(-1)
                        s = g * s
                    return {'score': s}
                return _edge_udf

            for etype in edge_subgraph.canonical_etypes:
                if edge_subgraph.num_edges(etype) == 0:
                    _dev = next(iter(h_dict.values())).device if len(h_dict) > 0 else None
                    scores[etype] = torch.empty(0, device=_dev)
                    continue
                ek = _etype_key(etype)
                edge_subgraph.apply_edges(_edge_udf_factory(ek), etype=etype)
                scores[etype] = edge_subgraph.edges[etype].data['score']
            return scores


# -------------------------- 顶层模型 --------------------------

class HGTModel(nn.Module):
    def __init__(self, args, g_hetero: dgl.DGLHeteroGraph, rel_list):
        super().__init__()
        self.rel_list = rel_list
        in_dim = _getattr(args, 'in_dim', None)
        hid_dim = _getattr(args, 'h_dim', _getattr(args, 'hid_dim', None))
        out_dim = _getattr(args, 'out_dim', None)
        num_layers = _getattr(args, 'num_layers', 2)
        heads = _getattr(args, 'hgt_heads', _getattr(args, 'heads', 4))
        dropout = _getattr(args, 'dropout', 0.2)

        input_gate_type = _getattr(args, 'input_gate_type', 'none')
        input_gate_reduce = _getattr(args, 'input_gate_reduce', 4)
        residual_gate = _getattr(args, 'residual_gate', False)
        gate_bias = _getattr(args, 'gate_bias', 1.0)
        score_gate = _getattr(args, 'score_gate', 'none')
        film_condition = _getattr(args, 'film_condition', 'dst')
        semantic_gate = _getattr(args, 'semantic_gate', 'none')
        sem_hidden = _getattr(args, 'sem_hidden', 64)
        sem_gate_bias = _getattr(args, 'sem_gate_bias', 0.0)
        head_gate = _getattr(args, 'head_gate', False)

        self.encoder = _FullGraphHGTEncoder(
            g_hetero, in_dim, hid_dim, out_dim,
            num_layers=num_layers, heads=heads, dropout=dropout,
            input_gate_type=input_gate_type, input_gate_reduce=input_gate_reduce,
            residual_gate=residual_gate, gate_bias=gate_bias,
            head_gate=head_gate,
        )
        self.pred = ScorePredictor(out_dim=out_dim, g_hetero=g_hetero,
                                   score_gate=score_gate, film_condition=film_condition,
                                   semantic_gate=semantic_gate, sem_hidden=sem_hidden,
                                   sem_gate_bias=sem_gate_bias)

    def forward(self, args, positive_graph, negative_graph, blocks_unused=None, x_dict_unused=None):
        """与原版一致：忽略 blocks/x_dict，直接在全图同构视图上编码一次。"""
        z_dict = self.encoder()
        pos_score = self.pred(positive_graph, z_dict)
        neg_score = self.pred(negative_graph, z_dict)
        return pos_score, neg_score

