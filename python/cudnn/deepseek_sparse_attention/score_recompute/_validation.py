# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata-only admission for native FP32 indexer weights."""

import torch


def validate_fp32_indexer_inputs(q, k, weights, *, topk_indices=None, topk_length=None, out=None, denom_out=None, q_causal_offsets=None):
    """Decline layouts that the existing adapters would copy or convert."""
    if weights.dtype != torch.float32:
        return
    for name, tensor in (("Q", q), ("K", k)):
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"FP32 W requires BF16 {name}, got {tensor.dtype}")
    for name, tensor in (("Q", q), ("K", k), ("W", weights)):
        strides = tensor.stride() if isinstance(tensor, torch.Tensor) else tensor.stride
        if not strides or strides[-1] != 1:
            raise NotImplementedError(f"FP32 W requires unit last stride for {name}, got strides {strides}")
    for name, tensor in (("topk_indices", topk_indices), ("topk_length", topk_length), ("q_causal_offsets", q_causal_offsets)):
        if tensor is not None:
            if tensor.dtype != torch.int32:
                raise ValueError(f"FP32 W requires INT32 {name}, got {tensor.dtype}")
            if not tensor.is_contiguous():
                raise NotImplementedError(
                    f"FP32 W requires contiguous {name}, got strides {tensor.stride() if isinstance(tensor, torch.Tensor) else tensor.stride}"
                )
    for name, tensor in (("out", out), ("denom_out", denom_out)):
        if tensor is not None and not tensor.is_contiguous():
            raise NotImplementedError(
                f"FP32 W requires contiguous {name}, got strides {tensor.stride() if isinstance(tensor, torch.Tensor) else tensor.stride}"
            )
