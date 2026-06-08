"""
SAMN-KAN-EPI: Hybrid Selective Ancestral Memory Network + KAN for EPI.

Replaces ONLY the Transformer attention in KANBlock with SAMN's local windowed
attention, survival gates, and ancestral memory bin cross-attention.
KEEPS the KAN (B-spline) FFN, SelfAttentionPooling, and KANLinear heads.

Architecture per SAMNKANBlock:
    tokens → LN → LocalSelfAttention → residual
           → SurvivalGate → top-K → GRU update bin
           → LN → BinCrossAttention → residual
           → LN → KAN([d_model, kan_hidden, d_model]) → residual  ← PRESERVED

Full model:
    DNA Sequence (64 × L) → Seq CNN → BiLSTM → [128 tokens × d_model]
    Epigenetic (9 × 5000)  → Epi CNN → BiLSTM → [500 tokens × d_model]
                          ↓ concat ↓
                    [628 tokens × d_model]
                          ↓
         SAMNKANTransformer (SAMN attention + KAN FFN)
                          ↓
         SelfAttentionPooling (with Frobenius penalty)
                          ↓
    [enh_feat, prom_feat, attn_mean, attn_max] → 4 × d_model
                          ↓
    Classification: Linear → KANLinear → KANLinear → (B, 1)
    Distance:       KANLinear → KANLinear → (B, 1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Import KAN layers from the existing codebase
from src.model_layers import KANLinear, KAN


# ═══════════════════════════════════════════════════════════════════════════════
# SAMN Core Components (self-contained, from novel model)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SAMNConfig:
    """Configuration for SAMN attention components."""
    model_dim: int = 180
    bin_dim: int = 96
    num_layers: int = 3
    num_heads: int = 6
    local_window: int = 64
    bin_slots: int = 16
    survivors_per_layer: int = 8
    dropout: float = 0.1
    gate_temperature: float = 1.0
    novelty_weight: float = 0.35
    prediction_error_weight: float = 0.20
    bin_decay: float = 0.92
    bin_update: str = "gru"
    use_straight_through_gate: bool = True
    max_length: int = 640

    def __post_init__(self) -> None:
        if self.bin_update not in {"fifo", "gru", "attention"}:
            raise ValueError("bin_update must be one of: 'fifo', 'gru', 'attention'")
        if self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")


class LocalSelfAttention(nn.Module):
    """Multi-head self-attention with a local window mask."""

    def __init__(self, model_dim: int, num_heads: int, local_window: int, dropout: float):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.local_window = local_window
        self.qkv = nn.Linear(model_dim, model_dim * 3, bias=False)
        self.out = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = self._local_mask(seq_len, x.device)
        scores = scores.masked_fill(~mask, -1e4)
        weights = F.softmax(scores, dim=-1)
        weights = weights.masked_fill(~mask, 0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = self.dropout(weights)
        y = torch.matmul(weights, v)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        y = self.out(y)
        return y

    def _local_mask(self, seq_len: int, device: torch.device) -> Tensor:
        radius = max(1, self.local_window // 2)
        positions = torch.arange(seq_len, device=device)
        distance = (positions[:, None] - positions[None, :]).abs()
        return (distance <= radius).view(1, 1, seq_len, seq_len)


class BinCrossAttention(nn.Module):
    """Tokens read from the bounded ancestral bin."""

    def __init__(self, model_dim: int, bin_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = model_dim
        self.bin_dim = bin_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.q = nn.Linear(model_dim, model_dim, bias=False)
        self.k = nn.Linear(bin_dim, model_dim, bias=False)
        self.v = nn.Linear(bin_dim, model_dim, bias=False)
        self.out = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: Tensor, bin_state: Tensor) -> Tensor:
        batch, seq_len, _ = tokens.shape
        slots = bin_state.shape[1]
        q = self.q(tokens).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(bin_state).view(batch, slots, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(bin_state).view(batch, slots, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        weights = self.dropout(F.softmax(scores, dim=-1))
        y = torch.matmul(weights, v)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.model_dim)
        y = self.out(y)
        return y


class SurvivalGate(nn.Module):
    """Scores tokens for survival into the focus bin."""

    def __init__(self, config: SAMNConfig):
        super().__init__()
        self.config = config
        self.compress = nn.Linear(config.model_dim, config.bin_dim)
        self.predict_from_bin = nn.Linear(config.bin_dim, config.model_dim)
        self.score_mlp = nn.Sequential(
            nn.Linear(config.model_dim + config.bin_dim, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, 1),
        )

    def forward(
        self, tokens: Tensor, bin_state: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        batch, seq_len, _ = tokens.shape
        compressed = self.compress(tokens)
        bin_pool = bin_state.mean(dim=1)
        expanded_pool = bin_pool[:, None, :].expand(batch, seq_len, -1)
        learned_score = self.score_mlp(
            torch.cat([tokens, expanded_pool], dim=-1)
        ).squeeze(-1)
        novelty = self._novelty(compressed, bin_state)
        prediction_error = self._prediction_error(tokens, bin_pool)
        score = (
            learned_score
            + self.config.novelty_weight * novelty
            + self.config.prediction_error_weight * prediction_error
        )
        gate, gate_probability = self._topk_gate(score)
        debug = {
            "score": score.detach(),
            "novelty": novelty.detach(),
            "prediction_error": prediction_error.detach(),
            "gate": gate.detach(),
            "gate_probability": gate_probability.detach(),
        }
        return compressed, gate, gate_probability, score, debug

    def _novelty(self, candidates: Tensor, bin_state: Tensor) -> Tensor:
        candidate_norm = F.normalize(candidates, dim=-1)
        bin_norm = F.normalize(bin_state, dim=-1)
        similarity = torch.matmul(candidate_norm, bin_norm.transpose(1, 2))
        max_similarity = similarity.max(dim=-1).values
        return (1.0 - max_similarity).clamp(min=0.0, max=2.0)

    def _prediction_error(self, tokens: Tensor, bin_pool: Tensor) -> Tensor:
        prediction = self.predict_from_bin(bin_pool)[:, None, :]
        error = (tokens - prediction).pow(2).mean(dim=-1)
        return error / (error.detach().mean(dim=-1, keepdim=True) + 1e-6)

    def _topk_gate(self, score: Tensor) -> Tuple[Tensor, Tensor]:
        k = min(self.config.survivors_per_layer, score.shape[-1])
        topk = torch.topk(score, k=k, dim=-1).indices
        hard = torch.zeros_like(score).scatter(1, topk, 1.0)
        soft = torch.sigmoid(score / max(self.config.gate_temperature, 1e-4))
        if not self.config.use_straight_through_gate:
            gate = hard
        else:
            gate = hard.detach() - soft.detach() + soft
        return gate, soft


# ═══════════════════════════════════════════════════════════════════════════════
# DropPath (stochastic depth) — matches original model.py
# ═══════════════════════════════════════════════════════════════════════════════


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# ═══════════════════════════════════════════════════════════════════════════════
# SAMNKANBlock: SAMN attention + KAN FFN (the hybrid block)
# ═══════════════════════════════════════════════════════════════════════════════


class SAMNKANBlock(nn.Module):
    """Hybrid encoder block: SAMN attention + KAN FFN.

    Replaces global self-attention in KANBlock with SAMN's:
      1. Local windowed self-attention
      2. Survival gate → select top-K → GRU update bin
      3. Bin cross-attention (read from ancestral memory)

    Preserves KAN([d_model, kan_hidden, d_model]) as the FFN (pre-norm).
    """

    def __init__(self, config: SAMNConfig, kan_hidden: int = 64,
                 drop_path_rate: float = 0.0):
        super().__init__()
        dim = config.model_dim

        # SAMN attention components
        self.norm_local = nn.LayerNorm(dim, eps=1e-5)
        self.local_attention = LocalSelfAttention(
            dim, config.num_heads, config.local_window, config.dropout)

        self.survival_gate = SurvivalGate(config)

        self.norm_bin = nn.LayerNorm(dim, eps=1e-5)
        self.bin_cross_attention = BinCrossAttention(
            dim, config.bin_dim, config.num_heads, config.dropout)

        # KAN FFN — PRESERVED from original KANBlock (pre-norm)
        self.norm_kan = nn.LayerNorm(dim, eps=1e-5)
        self.kan = KAN([dim, kan_hidden, dim])

        # DropPath (stochastic depth)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.dropout = nn.Dropout(config.dropout)

        # Bin update (GRU-based)
        self.config = config
        self.slot_gru = nn.GRUCell(config.bin_dim, config.bin_dim)
        self.bin_norm = nn.LayerNorm(config.bin_dim)

    def forward(self, x: Tensor, bin_state: Tensor
                ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """
        Args:
            x: (B, S, d_model) token embeddings
            bin_state: (B, bin_slots, bin_dim) ancestral memory state

        Returns:
            x: (B, S, d_model) updated token embeddings
            next_bin: (B, bin_slots, bin_dim) updated memory state
            aux: dict of auxiliary loss terms
        """
        b, t, d = x.shape

        # 1) Local self-attention (pre-norm, residual)
        local = self.local_attention(self.norm_local(x))
        x = x + self.drop_path(self.dropout(local))

        # 2) Survival gate → select top-K → update bin
        candidates, gate, gate_prob, score, gate_debug = \
            self.survival_gate(x, bin_state)
        selected, selected_gates, selected_indices = \
            self._select_candidates(candidates, gate, score)
        selected = selected * selected_gates.unsqueeze(-1)
        next_bin = self._gru_update(bin_state, selected)

        # 3) Bin cross-attention (pre-norm, residual)
        bin_read = self.bin_cross_attention(self.norm_bin(x), next_bin)
        x = x + self.drop_path(self.dropout(bin_read))

        # 4) KAN FFN (pre-norm, residual) — PRESERVED from original KANBlock
        #    KAN requires 2D input: reshape (B*S, d) → KAN → reshape (B, S, d)
        x = x + self.drop_path(
            self.kan(self.norm_kan(x).reshape(-1, d)).reshape(b, t, d)
        )

        # Auxiliary terms
        aux = self._auxiliary_terms(gate, gate_prob, next_bin)
        aux["selected_indices"] = selected_indices.detach()

        return x, next_bin, aux

    def _select_candidates(self, candidates: Tensor, gate: Tensor,
                           score: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        k = min(self.config.survivors_per_layer, candidates.shape[1],
                self.config.bin_slots)
        selected_indices = torch.topk(score, k=k, dim=-1).indices
        gather_index = selected_indices.unsqueeze(-1).expand(
            -1, -1, candidates.shape[-1])
        selected = torch.gather(candidates, dim=1, index=gather_index)
        selected_gate = torch.gather(gate, dim=1, index=selected_indices)
        return selected, selected_gate, selected_indices

    def _gru_update(self, bin_state: Tensor, selected: Tensor) -> Tensor:
        batch, slots, dim = bin_state.shape
        writes = torch.zeros_like(bin_state)
        write_count = min(selected.shape[1], slots)
        writes[:, :write_count, :] = selected[:, :write_count, :]
        aged = bin_state * self.config.bin_decay
        updated = self.slot_gru(
            writes.reshape(batch * slots, dim),
            aged.reshape(batch * slots, dim),
        ).view(batch, slots, dim)
        write_mask = torch.zeros(batch, slots, 1, device=bin_state.device,
                                 dtype=bin_state.dtype)
        write_mask[:, :write_count, :] = 1.0
        next_bin = write_mask * updated + (1.0 - write_mask) * aged
        return self.bin_norm(next_bin)

    def _auxiliary_terms(self, gate: Tensor, gate_probability: Tensor,
                         bin_state: Tensor) -> Dict[str, Tensor]:
        denom = gate.numel()
        gate_mean = gate.sum() / max(denom, 1)
        p = gate_probability.clamp(1e-6, 1.0 - 1e-6)
        entropy = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
        gate_entropy = entropy.sum() / max(denom, 1)
        norm_bin = F.normalize(bin_state, dim=-1)
        similarity = torch.matmul(norm_bin, norm_bin.transpose(1, 2)).pow(2)
        slots = similarity.shape[-1]
        eye = torch.eye(slots, device=similarity.device, dtype=torch.bool)[None, :, :]
        slot_diversity_penalty = similarity.masked_fill(eye, 0.0).sum() / max(
            similarity.shape[0] * slots * max(slots - 1, 1), 1)
        return {
            "gate_mean": gate_mean,
            "gate_entropy": gate_entropy,
            "slot_diversity_penalty": slot_diversity_penalty,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SAMNKANTransformer: Stacks SAMNKANBlocks (replaces KANTransformer)
# ═══════════════════════════════════════════════════════════════════════════════


class SAMNKANTransformer(nn.Module):
    """Hybrid encoder: SAMN attention + KAN FFN, stacked with drop path.

    Drop-in replacement for KANTransformer from model.py.
    Additionally manages the ancestral memory bin state across layers.
    """

    def __init__(self, samn_config: SAMNConfig, kan_hidden: int = 64,
                 drop_path_rate: float = 0.0):
        super().__init__()
        depth = samn_config.num_layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.initial_bin = nn.Parameter(
            torch.randn(samn_config.bin_slots, samn_config.bin_dim) * 0.02)

        self.blocks = nn.ModuleList([
            SAMNKANBlock(samn_config, kan_hidden=kan_hidden,
                         drop_path_rate=dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(samn_config.model_dim, eps=1e-5)

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Args:
            x: (B, S, d_model)

        Returns:
            x: (B, S, d_model) — encoded tokens
            auxiliary_terms: dict of scalar tensors (averaged across layers)
        """
        batch = x.shape[0]
        bin_state = self.initial_bin[None, :, :].expand(batch, -1, -1)

        aux_by_layer: List[Dict[str, Tensor]] = []
        for block in self.blocks:
            x, bin_state, aux = block(x, bin_state)
            aux_by_layer.append(aux)

        x = self.norm(x)

        # Aggregate auxiliary terms across layers
        keys = [k for k in aux_by_layer[0].keys() if k != "selected_indices"]
        auxiliary_terms = {
            key: torch.stack([a[key] for a in aux_by_layer]).mean()
            for key in keys
        }

        return x, auxiliary_terms


# ═══════════════════════════════════════════════════════════════════════════════
# Positional Encoding (sinusoidal, identical to original model.py)
# ═══════════════════════════════════════════════════════════════════════════════


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ═══════════════════════════════════════════════════════════════════════════════
# SelfAttentionPooling (identical to original model.py)
# ═══════════════════════════════════════════════════════════════════════════════


class SelfAttentionPooling(nn.Module):
    """Structured self-attention pooling (Lin et al. 2017)."""

    def __init__(self, d_model, da=64, r=32):
        super().__init__()
        self.r = r
        self.ws1 = nn.Linear(d_model, da, bias=True)
        self.ws2 = nn.Linear(da, r, bias=True)

    def forward(self, H):
        A = torch.softmax(self.ws2(torch.tanh(self.ws1(H))), dim=1)  # (B, S, r)
        A = A.transpose(1, 2)  # (B, r, S)
        M = torch.bmm(A, H)   # (B, r, d_model)
        return M, A

    def penalization_term(self, A):
        AAT = torch.bmm(A, A.transpose(1, 2))
        I = torch.eye(self.r, device=A.device).unsqueeze(0)
        return torch.norm(AAT - I, p='fro', dim=(1, 2)).mean()


# ═══════════════════════════════════════════════════════════════════════════════
# SAMNKANKansformerEPI: Full hybrid model
# ═══════════════════════════════════════════════════════════════════════════════


class SAMNKANKansformerEPI(nn.Module):
    """POCD-KansformerEPI with SAMN attention + KAN FFN hybrid encoder.

    Preserves from original Kansformer:
      - Dual-branch CNN+BiLSTM feature extraction (identical)
      - KAN([d_model, kan_hidden, d_model]) FFN in each encoder block
      - SelfAttentionPooling with Frobenius penalty
      - KANLinear classification and distance regression heads
      - [enh_feat, prom_feat, attn_mean, attn_max] pooling = 4 × d_model

    Replaces from original Kansformer:
      - Global Multi-Head Self-Attention → SAMN local attention
      - (nothing) → adds SurvivalGate + BinCrossAttention + ancestral memory

    Forward returns: (cls_out, reg_out, A, samn_aux)
      - A: attention matrix from SelfAttentionPooling (for Frobenius penalty)
      - samn_aux: dict of SAMN auxiliary loss terms
    """

    def __init__(self, config: dict):
        super().__init__()
        d_model = config["model"]["hidden_dim"]            # 180
        n_seq_tokens = config["model"].get("n_tokens", 128)
        drop = config["model"].get("dropout", 0.1)
        depth = config["model"].get("num_layers", 3)
        num_heads = config["model"].get("num_heads", 6)
        kan_hidden = config["model"].get("kan_hidden", 64)

        self.n_seq_tokens = n_seq_tokens
        self.epi_pool_factor = 10  # MaxPool1d kernel → 5000/10 = 500
        self.n_epi_tokens = config["data"]["epigenetic_bins"] // self.epi_pool_factor
        self.d_model = d_model

        # ─── Sequence Branch (POCD-ND encoded: 64 channels × L) ───
        self.seq_cnn = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(n_seq_tokens),
        )

        # Seq BiLSTM (2 layers, bidirectional)
        self.seq_bilstm = nn.LSTM(
            d_model, d_model // 2, batch_first=True,
            bidirectional=True, num_layers=2, dropout=drop,
        )
        self.seq_drop = nn.Dropout(drop)

        # ─── Epigenetic Branch (n_epi channels × 5000 bins) ───
        n_epi = config["data"]["n_epigenetic_features"]  # 9
        self.epi_cnn = nn.Sequential(
            nn.Conv1d(n_epi, d_model, kernel_size=11, padding=5),
            nn.BatchNorm1d(d_model),
            nn.LeakyReLU(),
            nn.MaxPool1d(self.epi_pool_factor),  # 5000 → 500 tokens
        )

        # Epi BiLSTM
        self.epi_bilstm = nn.LSTM(
            d_model, d_model // 2, batch_first=True,
            bidirectional=True, num_layers=2, dropout=drop,
        )
        self.epi_drop = nn.Dropout(drop)

        # ─── Fusion + Positional Encoding ───
        total_tokens = n_seq_tokens + self.n_epi_tokens  # 128 + 500 = 628
        self.proj_drop = nn.Dropout(drop)
        self.pos_enc = PositionalEncoding(d_model, max_len=total_tokens + 2)

        # ─── SAMN-KAN Hybrid Encoder ───
        samn_cfg_dict = config.get("model", {}).get("samn", {})
        samn_config = SAMNConfig(
            model_dim=d_model,
            num_layers=samn_cfg_dict.get("num_layers", depth),
            num_heads=num_heads,
            local_window=samn_cfg_dict.get("local_window", 64),
            bin_slots=samn_cfg_dict.get("bin_slots", 16),
            bin_dim=samn_cfg_dict.get("bin_dim", 96),
            survivors_per_layer=samn_cfg_dict.get("survivors_per_layer", 8),
            dropout=drop,
            gate_temperature=samn_cfg_dict.get("gate_temperature", 1.0),
            novelty_weight=samn_cfg_dict.get("novelty_weight", 0.35),
            prediction_error_weight=samn_cfg_dict.get(
                "prediction_error_weight", 0.20),
            bin_decay=samn_cfg_dict.get("bin_decay", 0.92),
            bin_update=samn_cfg_dict.get("bin_update", "gru"),
            max_length=total_tokens + 2,
        )
        self.transformer = SAMNKANTransformer(
            samn_config, kan_hidden=kan_hidden, drop_path_rate=0.0)

        # ─── Self-Attention Pooling (PRESERVED from original) ───
        sa_da = config["model"].get("sa_da", 64)
        sa_r = config["model"].get("sa_r", 32)
        self.att_pool = SelfAttentionPooling(d_model, da=sa_da, r=sa_r)

        # Pool: [enh_feat, prom_feat, attn_mean, attn_max] = 4 × d_model
        pool_dim = d_model * 4  # 720, matching original exactly

        # ─── Classification Head (PRESERVED: Linear → KANLinear → KANLinear) ───
        self.head_drop = nn.Dropout(config["model"].get("head_dropout", 0.2))
        self.fc_linear = nn.Linear(pool_dim, 128)
        self.fc_kan1 = KANLinear(128, 64)
        self.fc_kan2 = KANLinear(64, 1)

        # ─── Distance Regression Head (PRESERVED: KANLinear → KANLinear) ───
        self.dist_kan1 = KANLinear(pool_dim, d_model)
        self.dist_kan2 = KANLinear(d_model, 1)

    def forward(self, seq: Tensor, epi: Tensor,
                enh_idx: Tensor, prom_idx: Tensor):
        """
        Args:
            seq: (B, 64, L) POCD-ND encoded DNA sequence
            epi: (B, n_feats, 5000) epigenetic features
            enh_idx: (B,) or (B, 1) enhancer bin index in 5000-space
            prom_idx: (B,) or (B, 1) promoter bin index in 5000-space

        Returns:
            cls_out: (B, 1) classification logits
            reg_out: (B, 1) distance regression
            A: (B, r, S) attention matrix from SelfAttentionPooling
            samn_aux: dict of SAMN auxiliary losses
        """
        B = seq.size(0)

        # ─── Sequence branch ───
        x_seq = self.seq_cnn(seq)                # (B, d_model, n_seq_tokens)
        x_seq = x_seq.permute(0, 2, 1)           # (B, n_seq_tokens, d_model)
        x_seq, _ = self.seq_bilstm(x_seq)        # (B, n_seq_tokens, d_model)
        x_seq = self.seq_drop(x_seq)

        # ─── Epigenetic branch ───
        x_epi = self.epi_cnn(epi)                # (B, d_model, n_epi_tokens)
        x_epi = x_epi.permute(0, 2, 1)           # (B, n_epi_tokens, d_model)
        x_epi, _ = self.epi_bilstm(x_epi)        # (B, n_epi_tokens, d_model)
        x_epi = self.epi_drop(x_epi)

        # ─── Concat + positional encoding ───
        z = torch.cat([x_seq, x_epi], dim=1)     # (B, n_seq + n_epi, d_model)
        z = self.proj_drop(z)
        z = self.pos_enc(z)

        # ─── SAMN-KAN Hybrid Encoder ───
        z, samn_aux = self.transformer(z)         # (B, n_seq + n_epi, d_model)

        # ─── Self-Attention Pooling (PRESERVED) ───
        M, A = self.att_pool(z)                   # M: (B, r, d_model), A: (B, r, S)

        # ─── Enhancer / Promoter index-based feature extraction ───
        enh_token = self.n_seq_tokens + torch.div(
            enh_idx.long().view(B), self.epi_pool_factor, rounding_mode="trunc")
        prom_token = self.n_seq_tokens + torch.div(
            prom_idx.long().view(B), self.epi_pool_factor, rounding_mode="trunc")

        max_idx = self.n_seq_tokens + self.n_epi_tokens - 1
        enh_token = enh_token.clamp(self.n_seq_tokens, max_idx)
        prom_token = prom_token.clamp(self.n_seq_tokens, max_idx)

        batch_idx = torch.arange(B, device=z.device)
        enh_feat = z[batch_idx, enh_token, :]     # (B, d_model)
        prom_feat = z[batch_idx, prom_token, :]   # (B, d_model)

        # Attention-weighted mean and max
        attn_mean = M.mean(dim=1)                 # (B, d_model)
        attn_max = M.max(dim=1)[0]                # (B, d_model)

        z_pool = torch.cat(
            [enh_feat, prom_feat, attn_mean, attn_max], dim=1
        )  # (B, 4*d_model)

        # ─── Classification head (PRESERVED: Linear → KANLinear → KANLinear) ───
        feats = self.head_drop(z_pool)
        feats = self.fc_linear(feats)             # (B, 128)
        feats = self.fc_kan1(feats)               # (B, 64)
        cls_out = self.fc_kan2(feats)             # (B, 1)

        # ─── Distance head (PRESERVED: KANLinear → KANLinear) ───
        dist_feats = self.head_drop(z_pool)
        dist_feats = self.dist_kan1(dist_feats)   # (B, d_model)
        reg_out = self.dist_kan2(dist_feats)      # (B, 1)

        return cls_out, reg_out, A, samn_aux

    def attention_penalty(self, A: Tensor) -> Tensor:
        """Frobenius penalty on SelfAttentionPooling (identical to original)."""
        return self.att_pool.penalization_term(A)

    def samn_auxiliary_loss(self, aux_terms: Dict[str, Tensor],
                           entropy_weight: float = 0.01,
                           diversity_weight: float = 0.05) -> Tensor:
        """Compute SAMN-specific regularization losses.

        - gate_entropy: encourages exploration (higher = more diverse gating)
        - slot_diversity_penalty: penalizes redundant memory slots
        """
        loss = (
            -entropy_weight * aux_terms["gate_entropy"]
            + diversity_weight * aux_terms["slot_diversity_penalty"]
        )
        return loss
