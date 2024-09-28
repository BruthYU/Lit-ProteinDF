import pickle
import os
import numpy as np
from typing import List, Dict, Any
import collections
from omegaconf import OmegaConf
import dataclasses
from preprocess.tools import so3_utils, chemical, residue_constants, protein
from evaluate.openfold.utils import rigid_utils
from scipy.spatial.transform import Rotation
from Bio import PDB
from Bio.PDB.Chain import Chain
import string
import io
import gzip
from torch.utils import data
import torch

Protein = protein.Protein

# Global map from chain characters to integers.
ALPHANUMERIC = string.ascii_letters + string.digits + ' '
CHAIN_TO_INT = {
    chain_char: i for i, chain_char in enumerate(ALPHANUMERIC)
}
INT_TO_CHAIN = {
    i: chain_char for i, chain_char in enumerate(ALPHANUMERIC)
}

CHAIN_FEATS = [
    'atom_positions', 'aatype', 'atom_mask', 'residue_index', 'b_factors'
]
UNPADDED_FEATS = [
    't', 'rot_score_scaling', 'trans_score_scaling', 't_seq', 't_struct'
]
RIGID_FEATS = [
    'rigids_0', 'rigids_t'
]
PAIR_FEATS = [
    'rel_rots'
]


move_to_np = lambda x: x.cpu().detach().numpy()
aatype_to_seq = lambda aatype: ''.join([
        residue_constants.restypes_with_x[x] for x in aatype])


def pad_feats(raw_feats, max_len, use_torch=False):
    padded_feats = {
        feat_name: pad(feat, max_len, use_torch=use_torch)
        for feat_name, feat in raw_feats.items()
        if feat_name not in UNPADDED_FEATS + RIGID_FEATS
    }
    for feat_name in PAIR_FEATS:
        if feat_name in padded_feats:
            padded_feats[feat_name] = pad(padded_feats[feat_name], max_len, pad_idx=1)
    for feat_name in UNPADDED_FEATS:
        if feat_name in raw_feats:
            padded_feats[feat_name] = raw_feats[feat_name]
    for feat_name in RIGID_FEATS:
        if feat_name in raw_feats:
            padded_feats[feat_name] = pad_rigid(raw_feats[feat_name], max_len)
    return padded_feats

def pad_rigid(rigid: torch.tensor, max_len: int):
    num_rigids = rigid.shape[0]
    pad_amt = max_len - num_rigids
    pad_rigid = rigid_utils.Rigid.identity(
        (pad_amt,), dtype=rigid.dtype, device=rigid.device, requires_grad=False)
    return torch.cat([rigid, pad_rigid.to_tensor_7()], dim=0)

def pad(x: np.ndarray, max_len: int, pad_idx=0, use_torch=False, reverse=False):
    """Right pads dimension of numpy array.

    Args:
        x: numpy like array to pad.
        max_len: desired length after padding
        pad_idx: dimension to pad.
        use_torch: use torch padding method instead of numpy.

    Returns:
        x with its pad_idx dimension padded to max_len
    """
    # Pad only the residue dimension.
    seq_len = x.shape[pad_idx]
    pad_amt = max_len - seq_len
    pad_widths = [(0, 0)] * x.ndim
    if pad_amt < 0:
        raise ValueError(f'Invalid pad amount {pad_amt}')
    if reverse:
        pad_widths[pad_idx] = (pad_amt, 0)
    else:
        pad_widths[pad_idx] = (0, pad_amt)
    if use_torch:
        return torch.pad(x, pad_widths)
    return np.pad(x, pad_widths)

# read A3M and convert letters into
# integers in the 0..20 range,
# also keep track of insertions


def concat_np_features(
        np_dicts: List[Dict[str, np.ndarray]], add_batch_dim: bool):
    """Performs a nested concatenation of feature dicts.

    Args:
        np_dicts: list of dicts with the same structure.
            Each dict must have the same keys and numpy arrays as the values.
        add_batch_dim: whether to add a batch dimension to each feature.

    Returns:
        A single dict with all the features concatenated.
    """
    combined_dict = collections.defaultdict(list)
    for chain_dict in np_dicts:
        for feat_name, feat_val in chain_dict.items():
            if add_batch_dim:
                feat_val = feat_val[None]
            combined_dict[feat_name].append(feat_val)
    # Concatenate each feature
    for feat_name, feat_vals in combined_dict.items():
        combined_dict[feat_name] = np.concatenate(feat_vals, axis=0)
    return combined_dict

def length_batching(
        np_dicts: List[Dict[str, np.ndarray]],
        max_squared_res: int,
    ):
    get_len = lambda x: x['res_mask'].shape[0]
    dicts_by_length = [(get_len(x), x) for x in np_dicts]
    length_sorted = sorted(dicts_by_length, key=lambda x: x[0], reverse=True)
    max_len = length_sorted[0][0]
    max_batch_examples = int(max_squared_res // max_len**2)
    pad_example = lambda x: pad_feats(x, max_len)
    padded_batch = [
        pad_example(x) for (_, x) in length_sorted[:max_batch_examples]]
    return torch.utils.data.default_collate(padded_batch)

def create_data_loader(
        torch_dataset: data.Dataset,
        batch_size,
        shuffle,
        sampler=None,
        num_workers=0,
        np_collate=False,
        max_squared_res=1e6,
        length_batch=False,
        drop_last=False,
        prefetch_factor=2):
    """Creates a data loader with jax compatible data structures."""
    if np_collate:
        collate_fn = lambda x: concat_np_features(x, add_batch_dim=True)
    elif length_batch:
        collate_fn = lambda x: length_batching(
            x, max_squared_res=max_squared_res)
    else:
        collate_fn = None
    persistent_workers = True if num_workers > 0 else False
    prefetch_factor = 2 if num_workers == 0 else prefetch_factor
    return data.DataLoader(
        torch_dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
        # Need fork https://github.com/facebookresearch/hydra/issues/964
        multiprocessing_context='fork' if num_workers != 0 else None,
        )



