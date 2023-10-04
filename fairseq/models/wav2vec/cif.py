import torch
from typing import Optional
from torch import Tensor

def prob_check(tensor, eps=1e-10, neg_inf=-1e8, logp=False):
    assert not torch.isnan(tensor).any(), (
        "Nan in a probability tensor."
    )
    # Add the eps here to prevent errors introduced by precision
    if logp:
        assert tensor.le(0).all() and tensor.ge(neg_inf).all(), (
            "Incorrect values in a log-probability tensor"
            ", -inf <= tensor <= 0"
        )
    else:
        assert tensor.le(1.0 + eps).all() and tensor.ge(0.0 - eps).all(), (
            "Incorrect values in a probability tensor"
            ", 0.0 <= tensor <= 1.0"
        )

def cif_function(
    input: Tensor,
    alpha: Tensor,
    beta: float = 1.0,
    tail_threshold: float = 0.5,
    padding_mask: Optional[Tensor] = None,
    target_lengths: Optional[Tensor] = None,
    eps: float = 1e-4
):
    """
        Args:
        input (Tensor): (N, S, C) Input features to be integrated.
        alpha (Tensor): (N, S) Weights corresponding to each elements in the
            input. It is expected to be after sigmoid function.
        beta (float): the threshold used for determine firing.
        tail_thres (float): the threshold for determine firing for tail handling.
        padding_mask (Tensor, optional): (N, S) A binary mask representing
            padded elements in the input.
        target_lengths (Tensor, optional): (N,) Desired length of the targets
            for each sample in the minibatch.
        eps (float, optional): Epsilon to prevent underflow for divisions.
            Default: 1e-4

    Returns -> Dict[str, List[Optional[Tensor]]]: Key/values described below.
        cif_out (Tensor): (N, T, C) The output integrated from the source.
        cif_lengths (Tensor): (N,) The output length for each element in batch.
        alpha_sum (Tensor): (N,) The sum of alpha for each element in batch.
            Can be used to compute the quantity loss.
        delays (Tensor): (N, T) The expected delay (in terms of source tokens) for
            each target tokens in the batch.
        tail_weights (Tensor, optional): (N,) During inference, return the tail.
    """
    B, T, C = input.size()
    assert tuple(alpha.size()) == (B, T), "Mismatched size between input and alpha"
    prob_check(alpha) # check validity of alpha value

    dtype = alpha.dtype
    alpha = alpha.float()

    if padding_mask is not None:
        padding_mask = padding_mask.bool()
        alpha = alpha.masked_fill(alpha, 0)

    if target_lengths is not None:
        feat_lengths = target_lengths.long()
        # Scaling Strategy: normalize alpha, make sum of alpha value equals beta * T
        desired_sum = beta * target_lengths.type_as(input)
        alpha_sum = alpha.sum(1)
        alpha = alpha * (desired_sum / alpha_sum).unsqueeze(1)
        max_num_fires = feat_lengths.max()
    else: # disable Scaling Strategy in inference
        alpha_sum = alpha.sum(1)
        # calculate fire times
        feat_lengths = (alpha_sum / beta).floor().long()
        max_num_fires = feat_lengths.max()

    # integrate step
    csum = alpha.cumsum(-1)
    with torch.no_grad():
        # example:
        # right_index:   0 0 1 1 1 3
        # left_index:    0 0 0 1 1 1
        # num_of_fires:  0 0 1 0 0 2
        right_index = (csum / beta).floor().long().clip(max=max_num_fires) # (B, T)
        left_index = right_index.roll(1, dims=1)
        left_index[:, 0] = 0
        # count number of fires from each source
        num_of_fires = right_index - left_index
        # some weights could be fire mutiple times, calculate extra fires
        extra_fires  = (num_of_fires - 1).clip(min=0)
    # The extra entry in temporal dim is for extra fire
    output = input.new_zeros((B, max_num_fires + 1, C))
    # delay = input.new_zeros((B, max_num_fires + 1))
    # source_range = torch.arange((0, T + 1)).unsqueeze(0).type_as(input)
    zero = alpha.new_zeros((1,))

    # right scatter
    fire_mask = num_of_fires > 0
    # weights corresponding to fire operation could have left or right weight
    # right_weight = csum - previous_fired_weight (B, T)
    right_weight = torch.where(
        fire_mask,
        csum - right_index.type_as(alpha) * beta,
        zero
    ).type_as(input)

    output.scatter_add_(
        1,
        right_index.unsqueeze(-1).expand(-1, -1, C), # (B, T, C)
        right_weight.unsqueeze(-1) * input # (B, T, C)
    )

    # left scatter
    left_weight = (
        alpha - right_weight - extra_fires.type_as(alpha) * beta
    ).type_as(input)
    output.scatter_add_(
        1,
        left_index.unsqueeze(-1).expand(-1, -1, C),
        left_weight.unsqueeze(-1) * input
    )

    # extra scatter
    if extra_fires.ge(0).any():
        extra_steps = extra_fires.max().item()
        tgt_index = left_index
        src_feats = input * beta
        # could be more than one extra fires
        for _ in range(extra_steps):
            tgt_index = (tgt_index + 1).clip(max=max_num_fires)
            src_mask = extra_fires > 0
            output.scatter_add_(
                1,
                tgt_index.unsqueeze(-1).expand(-1, -1, C),
                src_feats * src_mask.unsqueeze(2)
            )
            extra_fires -= 1
    
    # tail handing
    if target_lengths is not None:
        # during sum of alpha equals to beta * target_lengths
        # dosent need tail handing
        output = output[:, :max_num_fires, :]
    else:
        zero = right_weight.new_zeros((1,))
        # mask non-tail 
        r_mask = right_index == feat_lengths.unsqueeze(1)
        # aggregate tail-weights in fire position
        # (only weights in fire position has right_weights)
        tail_weights = torch.where(r_mask, right_weight, zero).sum(-1)
        # aggregate tail-weights not in fire position
        l_mask = left_index == feat_lengths.unsqueeze(1)
        tail_weights += torch.where(l_mask, left_weight, zero).sum(-1)

        extend_mask = tail_weights >= tail_threshold
        # extend 1 fire step and upscale the weights
        if extend_mask.any():
            upscale = torch.ones_like(output).scatter(
                1,
                feat_lengths.view(B, 1, 1).expand(-1, -1, C),
                # (B, T, C), may have infs so need the mask
                beta / tail_weights.masked_fill(~extend_mask, beta).view(B, 1, 1).expand(-1, -1, C)
            ).detach()

            output *= upscale
            # expand sequence length for which has extra fire
            feat_lengths += extend_mask.long()
            max_num_fires = feat_lengths.max()

    output = output[:, :max_num_fires, :]
    
    # a size (B, T) mask to erase tail tokens in sequence which dosen't apply extra fire
    tail_mask = torch.arange(max_num_fires, device=output.device).unsqueeze(0) \
        >= feat_lengths.unsqueeze(1)
    output[tail_mask] = 0

    return {
        "cif_out": [output],
        "cif_length": [feat_lengths],
        "alpha_sum": [alpha_sum.to(dtype)],
        "tail_weights": [tail_weights] if target_lengths is None else []
    }