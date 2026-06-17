import torch
from mindspeed_mm.fsdp.utils.device import IS_NPU_AVAILABLE

if IS_NPU_AVAILABLE:
    import torch_npu


def eager_unpermute(permuted_tokens, sorted_indices, probs):
    num_tokens, topk = (permuted_tokens.size(0), 1) if probs is None else (probs.numel(), probs.size(1))
    # permute() returns the inverse permutation: original route id -> sorted position.
    # Gather by that inverse mapping to restore route order.
    unpermuted_tokens = permuted_tokens.index_select(0, sorted_indices.to(torch.long))
    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
    if probs is not None:
        unpermuted_tokens *= probs.unsqueeze(-1)
    return unpermuted_tokens.sum(dim=1)


def fused_unpermute(permuted_tokens, sorted_indices, probs):
    if probs is not None:
        permuted_tokens = permuted_tokens.to(probs.dtype)
    return torch_npu.npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs)


def unpermute(permuted_tokens, sorted_indices, probs=None, fused=True):
    if permuted_tokens.size(0) != sorted_indices.numel():
        raise AssertionError(f'permuted tokens({permuted_tokens.size(0)}) != sorted indices({sorted_indices.size()})')
    if fused and IS_NPU_AVAILABLE:
        return fused_unpermute(permuted_tokens, sorted_indices, probs)
    else:
        return eager_unpermute(permuted_tokens, sorted_indices, probs)
