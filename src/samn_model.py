"""
SAMN-EPI: Selective Ancestral Memory Network for Enhancer-Promoter Interaction.

Replaces the KANTransformer encoder in POCD-KansformerEPI with SAMN
(Selective Ancestral Memory Network). The dual-branch CNN+BiLSTM feature
extraction is preserved exactly; only the core encoder changes.

Architecture:
    DNA Sequence (64 × L) → Seq CNN → BiLSTM → [128 tokens × d_model]
    Epigenetic (9 × 5000)  → Epi CNN → BiLSTM → [500 tokens × d_model]
                          ↓ concat ↓
                    [628 tokens × d_model]
                          ↓
          SAMN (local attention + survival gates + ancestral memory)
                          ↓
    Pooling: [enh_feat, prom_feat, mean_pool, max_pool, bin_pool]
                          ↓
    Classification Head → (B, 1)
    Distance Head       → (B, 1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════════════════
# SAMN Core Components (self-contained, copied from novel model)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FocusBinConfig:
    """Configuration for the Selective Ancestral Memory Network."""
    vocab_size: int = 6
    pad_token_id: int = 5
    max_length: int = 640
    input_feature_dim: Optional[int] = 180
    model_dim: int = 180
    bin_dim: int = 96
    num_layers: int = 4
    num_heads: int = 6
    local_window: int = 64
    bin_slots: int = 16
    survivors_per_layer: int = 8
    ffn_multiplier: int = 4
    num_classes: int = 1
    dropout: float = 0.1
    gate_temperature: float = 1.0
    novelty_weight: float = 0.35
    prediction_error_weight: float = 0.20
    bin_decay: float = 0.92
    bin_update: str = "gru"
    use_straight_through_gate: bool = True
    return_dict: bool = False

    def __post_init__(self) -> None:
        if self.bin_update not in {"fifo", "gru", "attention"}:
            raise ValueError("bin_update must be one of: 'fifo', 'gru', 'attention'")
        if self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")


@dataclass
class FocusBinOutput:
    logits: Optional[Tensor]
    last_hidden_state: Tensor
    bin_state: Tensor
    trace: Optional[List[Dict[str, Tensor]]] = None
    auxiliary_terms: Optional[Dict[str, Tensor]] = None


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

    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        batch, seq_len, dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = self._local_mask(seq_len, x.device)
        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].bool()
            mask = mask & key_mask
        scores = scores.masked_fill(~mask, -1e4)
        weights = F.softmax(scores, dim=-1)
        weights = weights.masked_fill(~mask, 0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = self.dropout(weights)
        y = torch.matmul(weights, v)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        y = self.out(y)
        if attention_mask is not None:
            y = y * attention_mask[:, :, None].to(y.dtype)
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

    def forward(self, tokens: Tensor, bin_state: Tensor,
                attention_mask: Optional[Tensor] = None) -> Tensor:
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
        if attention_mask is not None:
            y = y * attention_mask[:, :, None].to(y.dtype)
        return y


class SurvivalGate(nn.Module):
    """Scores tokens for survival into the focus bin."""

    def __init__(self, config: FocusBinConfig):
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
        attention_mask: Optional[Tensor] = None,
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
        if attention_mask is not None:
            score = score.masked_fill(~attention_mask.bool(), -1e4)
        gate, gate_probability = self._topk_gate(score, attention_mask)
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

    def _topk_gate(self, score: Tensor,
                   attention_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        k = min(self.config.survivors_per_layer, score.shape[-1])
        topk = torch.topk(score, k=k, dim=-1).indices
        hard = torch.zeros_like(score).scatter(1, topk, 1.0)
        soft = torch.sigmoid(score / max(self.config.gate_temperature, 1e-4))
        if not self.config.use_straight_through_gate:
            gate = hard
        else:
            gate = hard.detach() - soft.detach() + soft
        if attention_mask is not None:
            mask = attention_mask.to(gate.dtype)
            gate = gate * mask
            soft = soft * mask
        return gate, soft


class FocusBinBlock(nn.Module):
    """One selective ancestral memory block."""

    def __init__(self, config: FocusBinConfig):
        super().__init__()
        self.config = config
        self.local_attention = LocalSelfAttention(
            config.model_dim, config.num_heads, config.local_window, config.dropout
        )
        self.survival_gate = SurvivalGate(config)
        self.bin_cross_attention = BinCrossAttention(
            config.model_dim, config.bin_dim, config.num_heads, config.dropout
        )
        hidden = config.model_dim * config.ffn_multiplier
        self.ffn = nn.Sequential(
            nn.Linear(config.model_dim, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.model_dim),
        )
        self.norm_local = nn.LayerNorm(config.model_dim)
        self.norm_bin = nn.LayerNorm(config.model_dim)
        self.norm_ffn = nn.LayerNorm(config.model_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.slot_gru = nn.GRUCell(config.bin_dim, config.bin_dim)
        self.write_gate = nn.Linear(config.bin_dim * 2, config.bin_dim)
        self.write_value = nn.Linear(config.bin_dim, config.bin_dim)
        self.bin_norm = nn.LayerNorm(config.bin_dim)

    def forward(
        self, tokens: Tensor, bin_state: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor], Dict[str, Tensor]]:
        local = self.local_attention(tokens, attention_mask)
        tokens = self.norm_local(tokens + self.dropout(local))
        if attention_mask is not None:
            tokens = tokens * attention_mask[:, :, None].to(tokens.dtype)

        candidates, gate, gate_probability, score, gate_debug = \
            self.survival_gate(tokens, bin_state, attention_mask)
        selected_candidates, selected_gates, selected_indices = \
            self._select_candidates(candidates, gate, score)
        selected_candidates = selected_candidates * selected_gates.unsqueeze(-1)
        next_bin = self._update_bin(bin_state, selected_candidates)

        bin_read = self.bin_cross_attention(tokens, next_bin, attention_mask)
        tokens = self.norm_bin(tokens + self.dropout(bin_read))
        tokens = self.norm_ffn(tokens + self.dropout(self.ffn(tokens)))
        if attention_mask is not None:
            tokens = tokens * attention_mask[:, :, None].to(tokens.dtype)

        debug = {
            **gate_debug,
            "selected_indices": selected_indices.detach(),
            "bin_state": next_bin.detach(),
        }
        aux = self._auxiliary_terms(gate, gate_probability, next_bin, attention_mask)
        return tokens, next_bin, debug, aux

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

    def _update_bin(self, bin_state: Tensor, selected: Tensor) -> Tensor:
        if self.config.bin_update == "fifo":
            return self._fifo_update(bin_state, selected)
        if self.config.bin_update == "gru":
            return self._gru_update(bin_state, selected)
        return self._attention_update(bin_state, selected)

    def _fifo_update(self, bin_state: Tensor, selected: Tensor) -> Tensor:
        aged = bin_state * self.config.bin_decay
        next_bin = torch.cat([selected, aged], dim=1)
        return next_bin[:, :self.config.bin_slots, :]

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

    def _attention_update(self, bin_state: Tensor, selected: Tensor) -> Tensor:
        if selected.shape[1] == 0:
            return bin_state * self.config.bin_decay
        aged = bin_state * self.config.bin_decay
        scores = torch.matmul(aged, selected.transpose(1, 2)) / (
            self.config.bin_dim ** 0.5)
        weights = F.softmax(scores, dim=-1)
        write = torch.matmul(weights, selected)
        candidate = torch.tanh(self.write_value(write))
        gate = torch.sigmoid(
            self.write_gate(torch.cat([aged, candidate], dim=-1)))
        return self.bin_norm((1.0 - gate) * aged + gate * candidate)

    def _auxiliary_terms(
        self, gate: Tensor, gate_probability: Tensor,
        bin_state: Tensor, attention_mask: Optional[Tensor],
    ) -> Dict[str, Tensor]:
        if attention_mask is None:
            valid = torch.ones_like(gate)
        else:
            valid = attention_mask.to(gate.dtype)
        denom = valid.sum().clamp_min(1.0)
        gate_mean = (gate * valid).sum() / denom
        p = gate_probability.clamp(1e-6, 1.0 - 1e-6)
        entropy = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
        gate_entropy = (entropy * valid).sum() / denom
        norm_bin = F.normalize(bin_state, dim=-1)
        similarity = torch.matmul(norm_bin, norm_bin.transpose(1, 2)).pow(2)
        slots = similarity.shape[-1]
        eye = torch.eye(slots, device=similarity.device, dtype=torch.bool)[
            None, :, :]
        slot_diversity_penalty = similarity.masked_fill(eye, 0.0).sum() / max(
            similarity.shape[0] * slots * max(slots - 1, 1), 1)
        return {
            "gate_mean": gate_mean,
            "gate_entropy": gate_entropy,
            "slot_diversity_penalty": slot_diversity_penalty,
        }


class SAMNEncoder(nn.Module):
    """SAMN encoder that processes continuous embeddings (no token embedding).

    This is a streamlined version of SelectiveAncestralMemoryNetwork that
    accepts pre-projected continuous features (inputs_embeds) and returns
    last_hidden_state + bin_state for downstream pooling.
    """

    def __init__(self, config: FocusBinConfig):
        super().__init__()
        self.config = config
        self.input_projection = (
            nn.Sequential(
                nn.Linear(config.input_feature_dim, config.model_dim),
                nn.LayerNorm(config.model_dim),
            )
            if config.input_feature_dim is not None
              and config.input_feature_dim != config.model_dim
            else nn.LayerNorm(config.model_dim)
        )
        self.position = nn.Embedding(config.max_length, config.model_dim)
        self.initial_bin = nn.Parameter(
            torch.randn(config.bin_slots, config.bin_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [FocusBinBlock(config) for _ in range(config.num_layers)])

    def forward(
        self,
        inputs_embeds: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """
        Args:
            inputs_embeds: (B, S, input_feature_dim) continuous features
            attention_mask: (B, S) binary mask

        Returns:
            hidden_states: (B, S, model_dim)
            bin_state: (B, bin_slots, bin_dim)
            auxiliary_terms: dict of scalar tensors
        """
        batch, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device

        x = self.input_projection(inputs_embeds)

        if attention_mask is None:
            attention_mask = torch.ones(batch, seq_len, dtype=torch.bool,
                                        device=device)
        else:
            attention_mask = attention_mask.to(dtype=torch.bool, device=device)

        positions = torch.arange(seq_len, device=device)
        x = x + self.position(positions)[None, :, :]
        x = x * attention_mask[:, :, None].to(x.dtype)

        bin_state = self.initial_bin[None, :, :].expand(batch, -1, -1)

        aux_by_layer: List[Dict[str, Tensor]] = []
        for block in self.blocks:
            x, bin_state, _, aux = block(x, bin_state, attention_mask)
            aux_by_layer.append(aux)

        # Aggregate auxiliary terms across layers
        keys = aux_by_layer[0].keys()
        auxiliary_terms = {
            key: torch.stack([a[key] for a in aux_by_layer]).mean()
            for key in keys
        }

        return x, bin_state, auxiliary_terms


# ═══════════════════════════════════════════════════════════════════════════════
# Positional Encoding (sinusoidal, same as original KansformerEPI)
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
# SAMN-EPI Model (replaces Kansformer)
# ═══════════════════════════════════════════════════════════════════════════════


class SAMNKansformerEPI(nn.Module):
    """POCD-KansformerEPI with SAMN replacing KANTransformer.

    The dual-branch CNN+BiLSTM feature extraction pipeline is preserved
    identically to the original. Only the core encoder is replaced.

    Forward signature and output format match the original Kansformer class
    so the training/evaluation loop can remain consistent.
    """

    def __init__(self, config: dict):
        super().__init__()
        d_model = config["model"]["hidden_dim"]                # 180
        n_seq_tokens = config["model"].get("n_tokens", 128)
        drop = config["model"].get("dropout", 0.1)

        self.n_seq_tokens = n_seq_tokens
        self.epi_pool_factor = 10  # MaxPool1d kernel → 5000/10 = 500
        self.n_epi_tokens = (
            config["data"]["epigenetic_bins"] // self.epi_pool_factor
        )
        self.d_model = d_model

        # ─── Sequence Branch (POCD-ND encoded: 64 channels x L) ───
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

        # ─── Fusion + Pre-SAMN Positional Encoding ───
        total_tokens = n_seq_tokens + self.n_epi_tokens  # 128 + 500 = 628
        self.proj_drop = nn.Dropout(drop)
        self.pos_enc = PositionalEncoding(d_model, max_len=total_tokens + 2)

        # ─── SAMN Encoder (replaces KANTransformer) ───
        samn_cfg = config.get("model", {}).get("samn", {})
        self.samn_config = FocusBinConfig(
            model_dim=d_model,
            input_feature_dim=d_model,  # Already projected by CNN+BiLSTM
            num_layers=samn_cfg.get("num_layers", 4),
            num_heads=config["model"].get("num_heads", 6),
            local_window=samn_cfg.get("local_window", 64),
            bin_slots=samn_cfg.get("bin_slots", 16),
            bin_dim=samn_cfg.get("bin_dim", 96),
            survivors_per_layer=samn_cfg.get("survivors_per_layer", 8),
            ffn_multiplier=samn_cfg.get("ffn_multiplier", 4),
            max_length=total_tokens + 2,
            dropout=drop,
            gate_temperature=samn_cfg.get("gate_temperature", 1.0),
            novelty_weight=samn_cfg.get("novelty_weight", 0.35),
            prediction_error_weight=samn_cfg.get(
                "prediction_error_weight", 0.20),
            bin_decay=samn_cfg.get("bin_decay", 0.92),
            bin_update=samn_cfg.get("bin_update", "gru"),
            num_classes=1,
        )
        self.samn_encoder = SAMNEncoder(self.samn_config)

        # ─── Pooling & Classification Head ───
        # [enh_feat, prom_feat, mean_pool, max_pool, bin_pool]
        # = d_model + d_model + d_model + d_model + bin_dim = 4*d_model + bin_dim
        pool_dim = d_model * 4 + self.samn_config.bin_dim  # 720 + 96 = 816

        self.head_drop = nn.Dropout(
            config["model"].get("head_dropout", 0.2))
        self.fc1 = nn.Linear(pool_dim, 256)
        self.fc_act = nn.GELU()
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

        # ─── Distance Regression Head ───
        self.dist_fc1 = nn.Linear(pool_dim, d_model)
        self.dist_act = nn.GELU()
        self.dist_fc2 = nn.Linear(d_model, 1)

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
            aux_terms: dict of SAMN auxiliary losses
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
        z = torch.cat([x_seq, x_epi], dim=1)     # (B, 628, d_model)
        z = self.proj_drop(z)
        z = self.pos_enc(z)

        # ─── SAMN Encoder (replaces KANTransformer) ───
        z, bin_state, aux_terms = self.samn_encoder(z)
        # z: (B, 628, d_model), bin_state: (B, bin_slots, bin_dim)

        # ─── Enhancer/Promoter index-based feature extraction ───
        enh_token = self.n_seq_tokens + torch.div(
            enh_idx.long().view(B), self.epi_pool_factor,
            rounding_mode="trunc",
        )
        prom_token = self.n_seq_tokens + torch.div(
            prom_idx.long().view(B), self.epi_pool_factor,
            rounding_mode="trunc",
        )
        max_idx = self.n_seq_tokens + self.n_epi_tokens - 1
        enh_token = enh_token.clamp(self.n_seq_tokens, max_idx)
        prom_token = prom_token.clamp(self.n_seq_tokens, max_idx)

        batch_idx = torch.arange(B, device=z.device)
        enh_feat = z[batch_idx, enh_token, :]     # (B, d_model)
        prom_feat = z[batch_idx, prom_token, :]   # (B, d_model)

        # ─── Pooling ───
        mean_pool = z.mean(dim=1)                 # (B, d_model)
        max_pool = z.max(dim=1)[0]                # (B, d_model)
        bin_pool = bin_state.mean(dim=1)           # (B, bin_dim)

        z_pool = torch.cat(
            [enh_feat, prom_feat, mean_pool, max_pool, bin_pool], dim=1
        )  # (B, 4*d_model + bin_dim)

        # ─── Classification head ───
        feats = self.head_drop(z_pool)
        feats = self.fc_act(self.fc1(feats))       # (B, 256)
        feats = self.fc_act(self.fc2(feats))       # (B, 64)
        cls_out = self.fc3(feats)                  # (B, 1)

        # ─── Distance head ───
        dist_feats = self.head_drop(z_pool)
        dist_feats = self.dist_act(self.dist_fc1(dist_feats))  # (B, d_model)
        reg_out = self.dist_fc2(dist_feats)        # (B, 1)

        return cls_out, reg_out, aux_terms

    def samn_auxiliary_loss(self, aux_terms: Dict[str, Tensor],
                           entropy_weight: float = 0.01,
                           diversity_weight: float = 0.05) -> Tensor:
        """Compute SAMN-specific regularization losses.

        - gate_entropy: encourages exploration (higher = more diverse gating)
        - slot_diversity_penalty: penalizes redundant memory slots
        """
        # We MAXIMIZE gate entropy (negate) and MINIMIZE slot diversity
        loss = (
            -entropy_weight * aux_terms["gate_entropy"]
            + diversity_weight * aux_terms["slot_diversity_penalty"]
        )
        return loss
