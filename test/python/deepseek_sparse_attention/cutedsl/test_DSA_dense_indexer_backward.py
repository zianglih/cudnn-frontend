# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from test_utils import torch_fork_set_rng

from deepseek_sparse_attention.cutedsl.dsa_utils import dsa_init, with_dsa_dense_indexer_backward_params
from deepseek_sparse_attention.cutedsl.dsa_reference import (
    _batched_ratio_causal_mask,
    check_ref_dense_indexer_backward,
    ref_dense_indexer_score_recompute,
)


def _allocate(cfg, sm_scale: float, ratio: int, q_causal_offsets: torch.Tensor, weights_dtype=torch.bfloat16):
    b = cfg["b"]
    s_q = cfg["s_q"]
    s_k = cfg["s_kv"]
    d = cfg["head_dim"]
    h = cfg["qhead_per_kv_head"]
    device = "cuda"

    index_q = torch.randn(b, s_q, h, d, dtype=torch.bfloat16, device=device)
    weights = torch.randn(b, s_q, h, dtype=weights_dtype, device=device)
    index_k = torch.randn(b, s_k, d, dtype=torch.bfloat16, device=device)

    with torch.no_grad():
        index_score, index_lse = ref_dense_indexer_score_recompute(
            index_q,
            index_k.unsqueeze(2),
            weights,
            ratio=ratio,
            q_causal_offsets=q_causal_offsets,
        )
        valid = _batched_ratio_causal_mask(s_q, s_k, ratio, device, b, q_causal_offsets)
        if sm_scale != 1.0:
            index_score = index_score * sm_scale
            index_lse = torch.logsumexp(
                index_score.masked_fill(~valid, float("-inf")),
                dim=-1,
            )

        attn_score = torch.rand(b, s_q, s_k, dtype=torch.float32, device=device)
        attn_score = attn_score.masked_fill(~valid, 0.0).contiguous()
        attn_l1norm = attn_score.sum(dim=-1).contiguous()

    return (
        index_q,
        weights,
        index_k,
        attn_score.contiguous(),
        attn_l1norm,
        index_score.contiguous(),
        index_lse.contiguous(),
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_dsa_dense_indexer_backward_params
@pytest.mark.parametrize(
    "weights_dtype,d_weights_dtype",
    [
        pytest.param(torch.bfloat16, None, id="bf16-w-default-dw"),
        pytest.param(torch.float32, None, id="fp32-w-default-dw"),
        pytest.param(torch.bfloat16, torch.float32, id="bf16-w-fp32-dw"),
        pytest.param(torch.float32, torch.bfloat16, id="fp32-w-bf16-dw"),
    ],
)
def test_DSA_dense_indexer_backward_wrapper(
    dtype,
    acc_dtype,
    head_dim,
    qhead_per_kv_head,
    block_I,
    weights_dtype,
    d_weights_dtype,
    ratio,
    request,
):
    try:
        from cudnn import DSA
        from cuda.bindings import driver as cuda
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    cfg = dsa_init(
        request=request,
        dtype=dtype,
        acc_dtype=acc_dtype,
        head_dim=head_dim,
        qhead_per_kv_head=qhead_per_kv_head,
        block_I=block_I,
        ratio=ratio,
        min_compute_capability=90,
        s_q_default=128,
        s_kv_default=512,
    )
    sm_scale = 1.0
    b_cfg = cfg["b"]
    s_q_cfg = cfg["s_q"]
    q_causal_offsets = torch.full((b_cfg,), 8, dtype=torch.int32, device="cuda")
    loss_coeff = float(b_cfg * s_q_cfg)
    grad_loss = torch.ones((), dtype=torch.float32, device="cuda")
    grad_scale_expected = loss_coeff / (b_cfg * s_q_cfg)

    (
        index_q,
        weights,
        index_k,
        attn_score,
        attn_l1norm,
        index_score,
        index_lse,
    ) = _allocate(cfg, sm_scale=sm_scale, ratio=ratio, q_causal_offsets=q_causal_offsets, weights_dtype=weights_dtype)
    torch_stream = torch.cuda.Stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    attn_score_ref = attn_score.clone()
    attn_l1norm_ref = attn_l1norm.clone()
    torch_stream.wait_stream(torch.cuda.current_stream())
    try:
        result = DSA.dense_indexer_backward_wrapper(
            index_q,
            weights,
            index_k,
            attn_score,
            attn_l1norm,
            index_score,
            index_lse,
            sm_scale=sm_scale,
            loss_coeff=loss_coeff,
            grad_loss=grad_loss,
            block_I=block_I,
            d_weights=torch.empty_like(weights, dtype=d_weights_dtype) if d_weights_dtype is not None else None,
            ratio=ratio,
            q_causal_offsets=q_causal_offsets,
            stream=stream,
        )
    except (ValueError, NotImplementedError, RuntimeError) as e:
        if weights_dtype == torch.float32 or d_weights_dtype == torch.float32:
            raise
        pytest.skip(f"Unsupported testcase: {e}")
    torch_stream.synchronize()

    d_index_q = result["d_index_q"]
    d_weights = result["d_weights"]
    d_index_k = result["d_index_k"]

    assert d_index_q.shape == index_q.shape
    assert d_weights.shape == weights.shape
    assert d_weights.dtype == (d_weights_dtype or weights_dtype)
    assert d_index_k.shape == index_k.shape
    assert torch.isfinite(d_index_q.float()).all()
    assert torch.isfinite(d_weights.float()).all()
    assert torch.isfinite(d_index_k.float()).all()

    if not cfg["skip_ref"]:
        check_ref_dense_indexer_backward(
            index_q,
            weights,
            index_k,
            attn_score_ref,
            attn_l1norm_ref,
            d_index_q,
            d_weights,
            d_index_k,
            sm_scale=sm_scale,
            ratio=ratio,
            grad_scale=grad_scale_expected,
            q_causal_offsets=q_causal_offsets,
        )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@pytest.mark.parametrize("weights_dtype", [torch.bfloat16, torch.float32], ids=["bf16-w", "fp32-w"])
def test_DSA_dense_indexer_backward_cuda_graph(weights_dtype):
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Dense indexer backward requires SM90+")

    try:
        from cudnn import DSA
    except ImportError:
        pytest.skip("Environment not supported: cudnn[cutedsl] not installed")

    cfg = {
        "b": 1,
        "s_q": 128,
        "s_kv": 128,
        "head_dim": 128,
        "qhead_per_kv_head": 64,
    }
    (
        index_q,
        weights,
        index_k,
        attn_score,
        attn_l1norm,
        index_score,
        index_lse,
    ) = _allocate(cfg, sm_scale=1.0, ratio=1, q_causal_offsets=None, weights_dtype=weights_dtype)
    attn_score_source = attn_score.clone()
    index_score_source = index_score.clone()
    d_index_q = torch.empty_like(index_q)
    d_weights = torch.empty_like(weights)
    d_index_k = torch.empty_like(index_k, dtype=torch.float32)
    grad_loss = torch.ones(1, dtype=torch.float32, device="cuda")

    def run():
        attn_score.copy_(attn_score_source)
        index_score.copy_(index_score_source)
        DSA.dense_indexer_backward_wrapper(
            index_q,
            weights,
            index_k,
            attn_score,
            attn_l1norm,
            index_score,
            index_lse,
            loss_coeff=float(cfg["b"] * cfg["s_q"]),
            grad_loss=grad_loss,
            block_I=128,
            ratio=1,
            d_index_q=d_index_q,
            d_weights=d_weights,
            d_index_k=d_index_k,
        )

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        run()
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    for scale in (0.5, 1.5):
        grad_loss.fill_(scale)
        graph.replay()
        check_ref_dense_indexer_backward(
            index_q,
            weights,
            index_k,
            attn_score_source,
            attn_l1norm,
            d_index_q,
            d_weights,
            d_index_k,
            grad_scale=scale,
        )


@pytest.mark.L0
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize(("queries", "keys"), [(1, 1), (17, 31), (32, 128), (33, 129), (32, 513)])
@pytest.mark.parametrize("weights_dtype", [torch.bfloat16, torch.float32], ids=["bf16-w", "fp32-w"])
def test_DSA_dense_indexer_backward_staging_sync(queries: int, keys: int, head_dim: int, weights_dtype) -> None:
    capability = torch.cuda.get_device_capability()[0]
    if capability < 9:
        pytest.skip("Dense indexer backward requires SM90+")
    if capability == 9:
        from cudnn.deepseek_sparse_attention.indexer_backward.dense_indexer_backward_sm90 import dense_indexer_backward_sm90 as make_kernel
    else:
        from cudnn.deepseek_sparse_attention.indexer_backward.dense_indexer_backward_sm100 import dense_indexer_backward_sm100 as make_kernel

    query = torch.ones(queries, 64, head_dim, device="cuda", dtype=torch.bfloat16)
    key = torch.ones(keys, head_dim, device="cuda", dtype=query.dtype)
    weights = torch.full(query.shape[:2], 1 + 2**-10, device="cuda", dtype=weights_dtype)
    grad_signal = torch.full((queries, keys), 1 + 3 * 2**-10, device="cuda", dtype=torch.float32)
    grad_signal.masked_fill_(
        torch.arange(keys, device="cuda")[None, :] > torch.arange(queries, device="cuda")[:, None] + keys - queries,
        0,
    )
    cu_query = torch.tensor([0, queries], device="cuda", dtype=torch.int32)
    cu_key = torch.tensor([0, keys], device="cuda", dtype=torch.int32)
    offsets = torch.tensor([keys - queries], device="cuda", dtype=torch.int32)
    query_grad = torch.empty_like(query)
    weights_grad = torch.empty_like(weights)
    key_grad = torch.zeros_like(key, dtype=torch.float32)
    kernel = make_kernel(
        1,
        queries,
        keys,
        64,
        head_dim,
        sm_scale=1.0,
        ratio=1,
        is_varlen=True,
        has_q_causal_offsets=True,
    )
    for sign in (1.0, -1.0, 1.0, -1.0):
        query.fill_(sign)
        key_grad.zero_()
        kernel.gemm_only(
            query,
            weights,
            key,
            query_grad,
            weights_grad,
            key_grad,
            grad_signal,
            cu_query,
            cu_key,
            offsets,
        )
        for name, gradient in (("dQ", query_grad), ("dK", key_grad), ("dW", weights_grad)):
            assert torch.isfinite(gradient).all(), f"Nonfinite {name}: {queries=}, {keys=}, {head_dim=}, {sign=}"
            if sign < 0:
                assert torch.count_nonzero(gradient).item() == 0, f"Negative QK requires zero {name}: {queries=}, {keys=}, {head_dim=}"
        if sign > 0:
            # The GEMMs intentionally consume BF16 dS, after multiplying FP32 W.
            # Narrowing W first gives 1.0 here instead of 1.0078125.
            d_scores = (grad_signal.double() * weights.double()[:, :1]).to(torch.bfloat16).double()
            expected_q = d_scores.sum(dim=1)[:, None, None].expand_as(query_grad).to(query_grad.dtype)
            expected_k = (64 * d_scores.sum(dim=0))[:, None].expand_as(key_grad).to(key_grad.dtype)
            expected_w = (head_dim * grad_signal.double().sum(dim=1))[:, None].expand_as(weights_grad).to(weights_grad.dtype)
            torch.testing.assert_close(query_grad, expected_q, atol=0, rtol=0)
            torch.testing.assert_close(key_grad, expected_k, atol=0, rtol=0)
            torch.testing.assert_close(weights_grad, expected_w, atol=0, rtol=0)
