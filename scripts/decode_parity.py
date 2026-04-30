#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import struct
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer

MODEL = Path(os.environ.get("QWEN36_PARITY_MODEL", str(Path.home() / "models/Qwen3.6-27B-Text-NVFP4-MTP")))
DUMP = Path(os.environ.get("QWEN36_PARITY_DUMP", "/tmp/qwen36_decode_dump"))
PROMPT = os.environ.get("QWEN36_PARITY_PROMPT", "hello")
DECODE_TOKEN = int(os.environ.get("QWEN36_PARITY_DECODE_TOKEN", "11"))
DEVICE = torch.device(os.environ.get("QWEN36_PARITY_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
COS_FLOOR = float(os.environ.get("QWEN36_DECODE_COS_FLOOR", "0.998"))

HIDDEN = 5120
Q_HEADS = 24
KV_HEADS = 4
HEAD_DIM = 256
ROPE_DIMS = 64
ROPE_THETA = 10_000_000.0
Q_DIM = Q_HEADS * HEAD_DIM
KV_DIM = KV_HEADS * HEAD_DIM
QK_HEADS = 16
V_HEADS = 48
KEY_DIM = 128
VALUE_DIM = 128
LINEAR_QKV_DIM = QK_HEADS * KEY_DIM * 2 + V_HEADS * VALUE_DIM
FULL_LAYERS = set(range(3, 64, 4))


def bf16_round(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def load_bf16(path: Path) -> torch.Tensor:
    raw = path.read_bytes()
    u16 = np.frombuffer(raw, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return torch.from_numpy(u32.view(np.float32).copy())


def decode_e4m3_value(code: int) -> float:
    sign = -1.0 if (code & 0x80) else 1.0
    exponent = (code >> 3) & 0x0F
    mantissa = code & 0x07
    if exponent == 0:
        if mantissa == 0:
            return 0.0
        return sign * math.ldexp(float(mantissa) / 8.0, -6)
    if exponent == 0x0F and mantissa == 0x07:
        return sign * 448.0
    return sign * math.ldexp(1.0 + float(mantissa) / 8.0, exponent - 7)


def encode_e4m3_positive_value(value: float) -> int:
    if not value > 0.0:
        return 0
    if value >= 448.0:
        return 0x7E

    min_normal = float.fromhex("0x1p-6")
    subnormal_step = float.fromhex("0x1p-9")
    normal_boundary = (7.0 * subnormal_step + min_normal) * 0.5
    if value < min_normal:
        if value >= normal_boundary:
            return 0x08
        mantissa = int(math.floor(value / subnormal_step + 0.49999994))
        if mantissa <= 0:
            return 0
        return mantissa

    bits = struct.unpack("<I", struct.pack("<f", float(value)))[0]
    exponent_field = int((bits >> 23) & 0xFF) - 120
    mantissa = ((bits & 0x007FFFFF) + 0x0007FFFF) >> 20
    if mantissa >= 8:
        mantissa = 0
        exponent_field += 1
    if exponent_field >= 15:
        exponent_field = 15
        if mantissa > 6:
            mantissa = 6
    return int((exponent_field << 3) | mantissa)


def nvfp4_activation_dequant_rows(x: torch.Tensor, input_scale: float = 1.0) -> torch.Tensor:
    device = x.device
    input_scale = float(input_scale) if input_scale > 0.0 else 1.0
    values_lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=device,
    )
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    rows = x.float().reshape(-1, x.shape[-1])
    out = torch.empty_like(rows)
    for row_idx in range(rows.shape[0]):
        row = rows[row_idx]
        for start in range(0, rows.shape[1], 16):
            group = row[start : start + 16]
            amax = group.abs().max()
            scale_value = max(float(amax / (6.0 * input_scale)), 1.0e-8) if amax > 0 else 1.0
            scale_code = encode_e4m3_positive_value(scale_value)
            decoded_scale = max(decode_e4m3_value(scale_code) * input_scale, 1.0e-8)
            decoded_scale_device = torch.tensor(decoded_scale, dtype=torch.float32, device=device)
            scaled = group / decoded_scale_device
            magnitude = torch.clamp(scaled.abs(), max=6.0)
            code = torch.bucketize(magnitude.cpu(), boundaries, right=False)
            dequant = values_lut[code].to(device) * decoded_scale_device
            out[row_idx, start : start + 16] = torch.where(scaled < 0.0, -dequant, dequant)
    return out.reshape_as(x.float())


def dequant_nvfp4_from(f, prefix: str) -> torch.Tensor:
    e2m1_lut = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=DEVICE,
    )
    packed = f.get_tensor(f"{prefix}.weight").to(DEVICE)
    scale = f.get_tensor(f"{prefix}.weight_scale").to(DEVICE)
    tscale = f.get_tensor(f"{prefix}.weight_scale_2").float().item()
    out_features = packed.shape[0]
    packed_in = packed.shape[1]
    in_features = packed_in * 2
    low = (packed & 0x0F).long()
    high = ((packed >> 4) & 0x0F).long()
    w = torch.empty(out_features, in_features, dtype=torch.float32, device=DEVICE)
    w[:, 0::2] = e2m1_lut[low]
    w[:, 1::2] = e2m1_lut[high]
    scale_f = scale.float()
    scale_expanded = scale_f.repeat_interleave(16, dim=1)
    return w * scale_expanded * tscale


def linear_nvfp4(
    f, prefix: str, x: torch.Tensor, quant_source: torch.Tensor | None = None
) -> torch.Tensor:
    w = dequant_nvfp4_from(f, prefix)
    input_scale = float(f.get_tensor(f"{prefix}.input_scale").float().item())
    qx = nvfp4_activation_dequant_rows(
        x if quant_source is None else quant_source,
        input_scale=input_scale,
    )
    y = qx.float() @ w.t()
    del w, qx
    return bf16_round(y)


def rmsnorm_bf16(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    rms = torch.rsqrt((x.float() * x.float()).mean(dim=-1, keepdim=True) + 1e-6)
    return bf16_round(x.float() * rms * (1.0 + weight.to(DEVICE).float()))


def rmsnorm_direct_bf16(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    rms = torch.rsqrt((x.float() * x.float()).mean(dim=-1, keepdim=True) + 1e-6)
    return bf16_round(x.float() * rms * weight.to(DEVICE).float())


def rmsnorm_with_residual_bf16(
    hidden: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    value = hidden.float() if residual is None else hidden.float() + residual.float()
    normed = rmsnorm_bf16(value, weight)
    return bf16_round(value), normed


def rmsnorm_with_residual_fused_quant(
    hidden: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value = hidden.float() if residual is None else hidden.float() + residual.float()
    weight_f = weight.to(DEVICE).float()
    rms = torch.rsqrt((value * value).mean(dim=-1, keepdim=True) + 1e-6)
    normed_f32 = value * rms * (1.0 + weight_f)
    return bf16_round(value), bf16_round(normed_f32), normed_f32


def apply_partial_rope_ref(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    out = x.float().clone()
    half = ROPE_DIMS // 2
    pair = torch.arange(half, dtype=torch.float32, device=DEVICE)
    inv_freq = ROPE_THETA ** (-(2.0 * pair) / ROPE_DIMS)
    for tok_idx, pos in enumerate(positions.to(DEVICE).float()):
        angle = pos * inv_freq
        cosv = torch.cos(angle)
        sinv = torch.sin(angle)
        x0 = out[tok_idx, :, :half].clone()
        x1 = out[tok_idx, :, half:ROPE_DIMS].clone()
        out[tok_idx, :, :half] = x0 * cosv - x1 * sinv
        out[tok_idx, :, half:ROPE_DIMS] = x1 * cosv + x0 * sinv
    return out


def causal_attention_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = q.float()
    k = k.float()
    v = v.float()
    tokens = q.shape[0]
    out = torch.empty(tokens, Q_HEADS, HEAD_DIM, dtype=torch.float32, device=DEVICE)
    q_per_kv = Q_HEADS // KV_HEADS
    scale = HEAD_DIM ** -0.5
    for token_idx in range(tokens):
        for qh in range(Q_HEADS):
            kvh = qh // q_per_kv
            scores = (k[: token_idx + 1, kvh] * q[token_idx, qh]).sum(dim=-1) * scale
            weights = torch.softmax(scores, dim=0)
            out[token_idx, qh] = (weights[:, None] * v[: token_idx + 1, kvh]).sum(dim=0)
    return out


def decode_attention_ref(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> torch.Tensor:
    q = q.float()
    k_cache = k_cache.float()
    v_cache = v_cache.float()
    out = torch.empty(Q_HEADS, HEAD_DIM, dtype=torch.float32, device=DEVICE)
    q_per_kv = Q_HEADS // KV_HEADS
    scale = HEAD_DIM ** -0.5
    for qh in range(Q_HEADS):
        kvh = qh // q_per_kv
        scores = (k_cache[:, kvh] * q[qh]).sum(dim=-1) * scale
        weights = torch.softmax(scores, dim=0)
        out[qh] = (weights[:, None] * v_cache[:, kvh]).sum(dim=0)
    return out


def conv1d_prefill_ref(f, prefix: str, qkv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = qkv.shape[0]
    conv_w = f.get_tensor(f"{prefix}.conv1d.weight").float()[:, 0, :].to(DEVICE)
    kernel = conv_w.shape[1]
    history = torch.zeros(qkv.shape[1], kernel - 1, dtype=torch.float32, device=DEVICE)
    out = torch.empty_like(qkv)
    for tok_idx in range(tokens):
        total = qkv[tok_idx].float() * conv_w[:, kernel - 1]
        for k in range(kernel - 1):
            lag = kernel - 1 - k
            if tok_idx >= lag:
                hist_value = qkv[tok_idx - lag].float()
            else:
                hist_value = history[:, k + tok_idx]
            total += hist_value * conv_w[:, k]
        out[tok_idx] = bf16_round(torch.nn.functional.silu(total))
    for k in range(kernel - 1):
        lag = kernel - 2 - k
        if tokens > lag:
            history[:, k] = qkv[tokens - 1 - lag].float()
        else:
            history[:, k] = history[:, k + tokens]
    return out, bf16_round(history)


def conv1d_decode_ref(
    f, prefix: str, qkv: torch.Tensor, history: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    conv_w = f.get_tensor(f"{prefix}.conv1d.weight").float()[:, 0, :].to(DEVICE)
    kernel = conv_w.shape[1]
    total = qkv[0].float() * conv_w[:, kernel - 1]
    for k in range(kernel - 1):
        total += history[:, k].float() * conv_w[:, k]
    new_history = history.clone()
    for k in range(kernel - 2):
        new_history[:, k] = new_history[:, k + 1]
    new_history[:, kernel - 2] = qkv[0].float()
    return bf16_round(torch.nn.functional.silu(total)).reshape(1, -1), bf16_round(new_history)


def deltanet_ref(
    conv: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = conv.shape[0]
    q_all = conv[:, : KEY_DIM * QK_HEADS].reshape(tokens, QK_HEADS, KEY_DIM)
    k_all = conv[:, KEY_DIM * QK_HEADS : 2 * KEY_DIM * QK_HEADS].reshape(tokens, QK_HEADS, KEY_DIM)
    v_all = conv[:, 2 * KEY_DIM * QK_HEADS :].reshape(tokens, V_HEADS, VALUE_DIM)
    if state is None:
        state = torch.zeros(V_HEADS, VALUE_DIM, KEY_DIM, dtype=torch.float32, device=DEVICE)
    else:
        state = state.float().clone()
    out = torch.empty(tokens, V_HEADS, VALUE_DIM, dtype=torch.float32, device=DEVICE)
    q_repeat = V_HEADS // QK_HEADS
    for tok_idx in range(tokens):
        for vh in range(V_HEADS):
            qh = vh // q_repeat
            q = q_all[tok_idx, qh].float()
            k = k_all[tok_idx, qh].float()
            v = v_all[tok_idx, vh].float()
            q_norm = torch.rsqrt((q * q).sum() + 1.0e-6) * (KEY_DIM ** -0.5)
            k_norm = torch.rsqrt((k * k).sum() + 1.0e-6)
            qn = q * q_norm
            kn = k * k_norm
            decayed = state[vh] * torch.exp(gate[tok_idx, vh])
            kv_mem = decayed @ kn
            s_q = decayed @ qn
            k_q = (kn * qn).sum()
            delta = (v - kv_mem) * beta[tok_idx, vh]
            out[tok_idx, vh] = bf16_round(s_q + delta * k_q)
            state[vh] = bf16_round(decayed + delta[:, None] * kn[None, :])
    return out.reshape(tokens, V_HEADS * VALUE_DIM), state


def linear_attention_prefill(f, layer_idx: int, normed: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prefix = f"model.language_model.layers.{layer_idx}.linear_attn"
    qkv = linear_nvfp4(f, f"{prefix}.in_proj_qkv", normed)
    b = linear_nvfp4(f, f"{prefix}.in_proj_b", normed)
    a = linear_nvfp4(f, f"{prefix}.in_proj_a", normed)
    conv, conv_history = conv1d_prefill_ref(f, prefix, qkv)
    a_log = f.get_tensor(f"{prefix}.A_log").float().to(DEVICE)
    dt_bias = f.get_tensor(f"{prefix}.dt_bias").float().to(DEVICE)
    gate = -torch.exp(a_log) * torch.nn.functional.softplus(a.float() + dt_bias)
    beta = torch.sigmoid(b.float())
    dn, dn_state = deltanet_ref(conv, gate, beta)
    z = linear_nvfp4(f, f"{prefix}.in_proj_z", normed)
    norm_w = f.get_tensor(f"{prefix}.norm.weight").float().to(DEVICE)
    dn_normed = rmsnorm_direct_bf16(dn.reshape(normed.shape[0] * V_HEADS, VALUE_DIM), norm_w)
    dn_normed = dn_normed.reshape(normed.shape[0], V_HEADS * VALUE_DIM)
    swiglu = bf16_round(torch.nn.functional.silu(z.float()) * dn_normed.float())
    out = linear_nvfp4(f, f"{prefix}.out_proj", swiglu)
    return out, {"conv_history": conv_history, "dn_state": dn_state}


def linear_attention_decode(
    f,
    layer_idx: int,
    normed: torch.Tensor,
    state: dict[str, torch.Tensor],
    quant_source: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prefix = f"model.language_model.layers.{layer_idx}.linear_attn"
    qkv = linear_nvfp4(f, f"{prefix}.in_proj_qkv", normed, quant_source)
    b = linear_nvfp4(f, f"{prefix}.in_proj_b", normed, quant_source)
    a = linear_nvfp4(f, f"{prefix}.in_proj_a", normed, quant_source)
    conv, conv_history = conv1d_decode_ref(f, prefix, qkv, state["conv_history"])
    a_log = f.get_tensor(f"{prefix}.A_log").float().to(DEVICE)
    dt_bias = f.get_tensor(f"{prefix}.dt_bias").float().to(DEVICE)
    gate = -torch.exp(a_log) * torch.nn.functional.softplus(a.float() + dt_bias)
    beta = torch.sigmoid(b.float())
    dn, dn_state = deltanet_ref(conv, gate, beta, state["dn_state"])
    z = linear_nvfp4(f, f"{prefix}.in_proj_z", normed, quant_source)
    norm_w = f.get_tensor(f"{prefix}.norm.weight").float().to(DEVICE)
    dn_normed = rmsnorm_direct_bf16(dn.reshape(V_HEADS, VALUE_DIM), norm_w).reshape(1, V_HEADS * VALUE_DIM)
    swiglu = bf16_round(torch.nn.functional.silu(z.float()) * dn_normed.float())
    out = linear_nvfp4(f, f"{prefix}.out_proj", swiglu)
    return out, {"conv_history": conv_history, "dn_state": dn_state}


def full_attention_prefill(f, layer_idx: int, normed: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prefix = f"model.language_model.layers.{layer_idx}.self_attn"
    tokens = normed.shape[0]
    q_raw = linear_nvfp4(f, f"{prefix}.q_proj", normed)
    k_raw = linear_nvfp4(f, f"{prefix}.k_proj", normed)
    v_raw = linear_nvfp4(f, f"{prefix}.v_proj", normed).reshape(tokens, KV_HEADS, HEAD_DIM)
    q_gate = q_raw.reshape(tokens, Q_HEADS, HEAD_DIM * 2)
    q = q_gate[..., :HEAD_DIM]
    gate = q_gate[..., HEAD_DIM:]
    q_norm_w = f.get_tensor(f"{prefix}.q_norm.weight").float().to(DEVICE)
    k_norm_w = f.get_tensor(f"{prefix}.k_norm.weight").float().to(DEVICE)
    q = rmsnorm_bf16(q.reshape(tokens * Q_HEADS, HEAD_DIM), q_norm_w).reshape(tokens, Q_HEADS, HEAD_DIM)
    k = rmsnorm_bf16(k_raw.reshape(tokens * KV_HEADS, HEAD_DIM), k_norm_w).reshape(tokens, KV_HEADS, HEAD_DIM)
    q = bf16_round(apply_partial_rope_ref(q, positions))
    k = bf16_round(apply_partial_rope_ref(k, positions))
    attn = bf16_round(causal_attention_ref(q, k, v_raw))
    gated = bf16_round(torch.sigmoid(gate.float()) * attn.float()).reshape(tokens, Q_DIM)
    out = linear_nvfp4(f, f"{prefix}.o_proj", gated)
    return out, {"k_cache": k, "v_cache": v_raw}


def full_attention_decode(
    f,
    layer_idx: int,
    normed: torch.Tensor,
    position: int,
    state: dict[str, torch.Tensor],
    quant_source: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prefix = f"model.language_model.layers.{layer_idx}.self_attn"
    q_raw = linear_nvfp4(f, f"{prefix}.q_proj", normed, quant_source)
    k_raw = linear_nvfp4(f, f"{prefix}.k_proj", normed, quant_source)
    v_raw = linear_nvfp4(f, f"{prefix}.v_proj", normed, quant_source).reshape(
        1, KV_HEADS, HEAD_DIM
    )
    q_gate = q_raw.reshape(1, Q_HEADS, HEAD_DIM * 2)
    q = q_gate[..., :HEAD_DIM]
    gate = q_gate[..., HEAD_DIM:]
    q_norm_w = f.get_tensor(f"{prefix}.q_norm.weight").float().to(DEVICE)
    k_norm_w = f.get_tensor(f"{prefix}.k_norm.weight").float().to(DEVICE)
    q = rmsnorm_bf16(q.reshape(Q_HEADS, HEAD_DIM), q_norm_w).reshape(1, Q_HEADS, HEAD_DIM)
    k = rmsnorm_bf16(k_raw.reshape(KV_HEADS, HEAD_DIM), k_norm_w).reshape(1, KV_HEADS, HEAD_DIM)
    pos = torch.tensor([position], dtype=torch.float32, device=DEVICE)
    q = bf16_round(apply_partial_rope_ref(q, pos))
    k = bf16_round(apply_partial_rope_ref(k, pos))
    k_cache = torch.cat([state["k_cache"], k], dim=0)
    v_cache = torch.cat([state["v_cache"], v_raw], dim=0)
    attn = bf16_round(decode_attention_ref(q[0], k_cache, v_cache)).reshape(1, Q_HEADS, HEAD_DIM)
    gated = bf16_round(torch.sigmoid(gate.float()) * attn.float()).reshape(1, Q_DIM)
    out = linear_nvfp4(f, f"{prefix}.o_proj", gated)
    return out, {"k_cache": k_cache, "v_cache": v_cache}


def mlp_ref(
    f, layer_idx: int, normed: torch.Tensor, quant_source: torch.Tensor | None = None
) -> torch.Tensor:
    _, _, _, out = mlp_parts_ref(f, layer_idx, normed, quant_source)
    return out


def mlp_parts_ref(
    f, layer_idx: int, normed: torch.Tensor, quant_source: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prefix = f"model.language_model.layers.{layer_idx}.mlp"
    gate = linear_nvfp4(f, f"{prefix}.gate_proj", normed, quant_source)
    up = linear_nvfp4(f, f"{prefix}.up_proj", normed, quant_source)
    swiglu = bf16_round(torch.nn.functional.silu(gate.float()) * up.float())
    out = linear_nvfp4(f, f"{prefix}.down_proj", swiglu)
    return gate, up, swiglu, out


def compare_dump(label: str, dump_name: str, ref: torch.Tensor, shape: tuple[int, ...]) -> float:
    path = DUMP / f"{dump_name}.bf16"
    if not path.exists():
        print(f"[{label}] missing dump {path}")
        return float("nan")
    ours = load_bf16(path).reshape(*shape).to(DEVICE)
    ours_f = ours.float().flatten()
    ref_f = ref.float().flatten()
    diff = (ours_f - ref_f).abs()
    cos = torch.nn.functional.cosine_similarity(ours_f, ref_f, dim=0).item()
    print(
        f"[{label}] mean_abs ours={ours_f.abs().mean().item():.6f} "
        f"ref={ref_f.abs().mean().item():.6f} max_diff={diff.max().item():.6f} "
        f"mean_diff={diff.mean().item():.6f} cos={cos:.6f}"
    )
    if cos < COS_FLOOR:
        print(f"  !! below floor {COS_FLOOR:.6f}")
    return cos


def build_prefill_state(f, ids: list[int]) -> dict[int, dict[str, torch.Tensor]]:
    embed = f.get_tensor("model.language_model.embed_tokens.weight").to(DEVICE).float()
    hidden = bf16_round(embed[torch.tensor(ids, dtype=torch.long, device=DEVICE)])
    residual: torch.Tensor | None = None
    positions = torch.arange(len(ids), dtype=torch.float32, device=DEVICE)
    states: dict[int, dict[str, torch.Tensor]] = {}
    for layer_idx in range(64):
        common = f"model.language_model.layers.{layer_idx}"
        residual, normed = rmsnorm_with_residual_bf16(
            hidden, f.get_tensor(f"{common}.input_layernorm.weight"), residual
        )
        if layer_idx in FULL_LAYERS:
            attn_out, state = full_attention_prefill(f, layer_idx, normed, positions)
        else:
            attn_out, state = linear_attention_prefill(f, layer_idx, normed)
        states[layer_idx] = state
        residual, post_normed = rmsnorm_with_residual_bf16(
            attn_out, f.get_tensor(f"{common}.post_attention_layernorm.weight"), residual
        )
        hidden = mlp_ref(f, layer_idx, post_normed)
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    return states


def build_prefill_state_from_dumps(f, tokens: int) -> dict[int, dict[str, torch.Tensor]]:
    positions = torch.arange(tokens, dtype=torch.float32, device=DEVICE)
    states: dict[int, dict[str, torch.Tensor]] = {}
    for layer_idx in range(64):
        normed = load_bf16(DUMP / f"layer{layer_idx:02}_input_normed.bf16").reshape(
            tokens, HIDDEN
        ).to(DEVICE)
        if layer_idx in FULL_LAYERS:
            _, state = full_attention_prefill(f, layer_idx, normed, positions)
        else:
            _, state = linear_attention_prefill(f, layer_idx, normed)
        states[layer_idx] = state
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    return states


def run_local_decode_checks(
    f, ids: list[int], decode_prefix: str, states: dict[int, dict[str, torch.Tensor]]
) -> None:
    worst = (1.0, "none")
    for layer_idx in range(64):
        common = f"model.language_model.layers.{layer_idx}"
        if layer_idx == 0:
            hidden = load_bf16(DUMP / f"{decode_prefix}_post_embed.bf16").reshape(1, HIDDEN).to(DEVICE)
            residual_in = None
        else:
            hidden = load_bf16(DUMP / f"{decode_prefix}_layer{layer_idx - 1:02}_mlp_out.bf16").reshape(
                1, HIDDEN
            ).to(DEVICE)
            residual_in = load_bf16(
                DUMP / f"{decode_prefix}_layer{layer_idx - 1:02}_residual_after_attn.bf16"
            ).reshape(1, HIDDEN).to(DEVICE)

        residual, normed, normed_quant = rmsnorm_with_residual_fused_quant(
            hidden, f.get_tensor(f"{common}.input_layernorm.weight"), residual_in
        )
        cos = compare_dump(
            f"local.decode.layer{layer_idx:02}.input_normed",
            f"{decode_prefix}_layer{layer_idx:02}_input_normed",
            normed,
            (1, HIDDEN),
        )
        if cos == cos and cos < worst[0]:
            worst = (cos, f"local.decode.layer{layer_idx:02}.input_normed")

        if layer_idx in FULL_LAYERS:
            attn_out, states[layer_idx] = full_attention_decode(
                f, layer_idx, normed, len(ids), states[layer_idx], normed_quant
            )
        else:
            attn_out, states[layer_idx] = linear_attention_decode(
                f, layer_idx, normed, states[layer_idx], normed_quant
            )
        cos = compare_dump(
            f"local.decode.layer{layer_idx:02}.attn_out",
            f"{decode_prefix}_layer{layer_idx:02}_attn_out",
            attn_out,
            (1, HIDDEN),
        )
        if cos == cos and cos < worst[0]:
            worst = (cos, f"local.decode.layer{layer_idx:02}.attn_out")

        engine_attn = load_bf16(
            DUMP / f"{decode_prefix}_layer{layer_idx:02}_attn_out.bf16"
        ).reshape(1, HIDDEN).to(DEVICE)
        residual, post_normed, post_normed_quant = rmsnorm_with_residual_fused_quant(
            engine_attn, f.get_tensor(f"{common}.post_attention_layernorm.weight"), residual
        )
        for suffix, ref, shape in [
            ("post_attn_normed", post_normed, (1, HIDDEN)),
            ("residual_after_attn", residual, (1, HIDDEN)),
        ]:
            cos = compare_dump(
                f"local.decode.layer{layer_idx:02}.{suffix}",
                f"{decode_prefix}_layer{layer_idx:02}_{suffix}",
                ref,
                shape,
            )
            if cos == cos and cos < worst[0]:
                worst = (cos, f"local.decode.layer{layer_idx:02}.{suffix}")

        engine_post_normed = load_bf16(
            DUMP / f"{decode_prefix}_layer{layer_idx:02}_post_attn_normed.bf16"
        ).reshape(1, HIDDEN).to(DEVICE)
        mlp_gate, mlp_up, mlp_swiglu, mlp_out = mlp_parts_ref(
            f, layer_idx, engine_post_normed, post_normed_quant
        )
        for suffix, ref, shape in [
            ("mlp_gate", mlp_gate, (1, 17408)),
            ("mlp_up", mlp_up, (1, 17408)),
            ("mlp_swiglu", mlp_swiglu, (1, 17408)),
            ("mlp_out", mlp_out, (1, HIDDEN)),
        ]:
            cos = compare_dump(
                f"local.decode.layer{layer_idx:02}.{suffix}",
                f"{decode_prefix}_layer{layer_idx:02}_{suffix}",
                ref,
                shape,
            )
            if cos == cos and cos < worst[0]:
                worst = (cos, f"local.decode.layer{layer_idx:02}.{suffix}")

        print(f"-- local decode layer {layer_idx:02} complete, worst={worst[1]} cos={worst[0]:.6f}")
        if worst[0] < COS_FLOOR and os.environ.get("QWEN36_DECODE_STOP_ON_FAIL", "1") != "0":
            raise SystemExit(1)
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    hidden = load_bf16(DUMP / f"{decode_prefix}_layer63_mlp_out.bf16").reshape(1, HIDDEN).to(DEVICE)
    residual = load_bf16(DUMP / f"{decode_prefix}_layer63_residual_after_attn.bf16").reshape(
        1, HIDDEN
    ).to(DEVICE)
    final_normed = rmsnorm_bf16(
        hidden.float() + residual.float(), f.get_tensor("model.language_model.norm.weight")
    )
    cos = compare_dump("local.decode.final_normed", f"{decode_prefix}_final_normed", final_normed, (1, HIDDEN))
    if cos == cos and cos < worst[0]:
        worst = (cos, "local.decode.final_normed")

    print(f"worst local decode boundary: {worst[1]} cos={worst[0]:.6f}")
    if worst[0] < COS_FLOOR:
        raise SystemExit(1)


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    ids = tokenizer(PROMPT, add_special_tokens=True).input_ids
    decode_prefix = f"decode_pos{len(ids):05}"
    print(
        f"decode parity prompt={PROMPT!r} ids={ids} "
        f"decode_token={DECODE_TOKEN} device={DEVICE}"
    )

    worst = (1.0, "none")
    with safe_open(str(MODEL / "model.safetensors"), framework="pt", device="cpu") as f:
        if os.environ.get("QWEN36_DECODE_LOCAL", "0") == "1":
            if (DUMP / "layer00_input_normed.bf16").exists():
                states = build_prefill_state_from_dumps(f, len(ids))
            else:
                states = build_prefill_state(f, ids)
            run_local_decode_checks(f, ids, decode_prefix, states)
            return

        states = build_prefill_state(f, ids)
        embed = f.get_tensor("model.language_model.embed_tokens.weight").to(DEVICE).float()
        hidden = bf16_round(embed[DECODE_TOKEN]).reshape(1, HIDDEN)
        cos = compare_dump("decode.post_embed", f"{decode_prefix}_post_embed", hidden, (1, HIDDEN))
        if cos == cos and cos < worst[0]:
            worst = (cos, "decode.post_embed")

        residual: torch.Tensor | None = None
        for layer_idx in range(64):
            common = f"model.language_model.layers.{layer_idx}"
            residual, normed, normed_quant = rmsnorm_with_residual_fused_quant(
                hidden, f.get_tensor(f"{common}.input_layernorm.weight"), residual
            )
            cos = compare_dump(
                f"decode.layer{layer_idx:02}.input_normed",
                f"{decode_prefix}_layer{layer_idx:02}_input_normed",
                normed,
                (1, HIDDEN),
            )
            if cos == cos and cos < worst[0]:
                worst = (cos, f"decode.layer{layer_idx:02}.input_normed")
            if layer_idx in FULL_LAYERS:
                attn_out, states[layer_idx] = full_attention_decode(
                    f, layer_idx, normed, len(ids), states[layer_idx], normed_quant
                )
            else:
                attn_out, states[layer_idx] = linear_attention_decode(
                    f, layer_idx, normed, states[layer_idx], normed_quant
                )
            cos = compare_dump(
                f"decode.layer{layer_idx:02}.attn_out",
                f"{decode_prefix}_layer{layer_idx:02}_attn_out",
                attn_out,
                (1, HIDDEN),
            )
            if cos == cos and cos < worst[0]:
                worst = (cos, f"decode.layer{layer_idx:02}.attn_out")

            residual, post_normed, post_normed_quant = rmsnorm_with_residual_fused_quant(
                attn_out, f.get_tensor(f"{common}.post_attention_layernorm.weight"), residual
            )
            cos = compare_dump(
                f"decode.layer{layer_idx:02}.post_attn_normed",
                f"{decode_prefix}_layer{layer_idx:02}_post_attn_normed",
                post_normed,
                (1, HIDDEN),
            )
            if cos == cos and cos < worst[0]:
                worst = (cos, f"decode.layer{layer_idx:02}.post_attn_normed")
            cos = compare_dump(
                f"decode.layer{layer_idx:02}.residual_after_attn",
                f"{decode_prefix}_layer{layer_idx:02}_residual_after_attn",
                residual,
                (1, HIDDEN),
            )
            if cos == cos and cos < worst[0]:
                worst = (cos, f"decode.layer{layer_idx:02}.residual_after_attn")

            mlp_gate, mlp_up, mlp_swiglu, hidden = mlp_parts_ref(
                f, layer_idx, post_normed, post_normed_quant
            )
            for suffix, ref, shape in [
                ("mlp_gate", mlp_gate, (1, 17408)),
                ("mlp_up", mlp_up, (1, 17408)),
                ("mlp_swiglu", mlp_swiglu, (1, 17408)),
            ]:
                cos = compare_dump(
                    f"decode.layer{layer_idx:02}.{suffix}",
                    f"{decode_prefix}_layer{layer_idx:02}_{suffix}",
                    ref,
                    shape,
                )
                if cos == cos and cos < worst[0]:
                    worst = (cos, f"decode.layer{layer_idx:02}.{suffix}")
            cos = compare_dump(
                f"decode.layer{layer_idx:02}.mlp_out",
                f"{decode_prefix}_layer{layer_idx:02}_mlp_out",
                hidden,
                (1, HIDDEN),
            )
            if cos == cos and cos < worst[0]:
                worst = (cos, f"decode.layer{layer_idx:02}.mlp_out")

            print(f"-- decode layer {layer_idx:02} complete, worst={worst[1]} cos={worst[0]:.6f}")
            if (
                worst[0] < COS_FLOOR
                and os.environ.get("QWEN36_DECODE_STOP_ON_FAIL", "1") != "0"
            ):
                raise SystemExit(1)
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        assert residual is not None
        final_value = hidden.float() + residual.float()
        final_normed = rmsnorm_bf16(
            final_value.reshape(1, HIDDEN), f.get_tensor("model.language_model.norm.weight")
        )
        cos = compare_dump(
            "decode.final_normed",
            f"{decode_prefix}_final_normed",
            final_normed,
            (1, HIDDEN),
        )
        if cos == cos and cos < worst[0]:
            worst = (cos, "decode.final_normed")

    print(f"worst decode boundary: {worst[1]} cos={worst[0]:.6f}")
    if worst[0] < COS_FLOOR:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
