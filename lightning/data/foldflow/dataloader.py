
import io
import pickle
import string
import os

import numpy as np
import torch
import math
from torch.utils import data
import collections
import torch.distributed as dist
from preprocess.tools import residue_constants
from evaluate.openfold.utils import rigid_utils
from typing import Dict, List, Tuple, Union, Any
from typing import Any, Optional
from Bio import PDB
from Bio.PDB.Chain import Chain
import dataclasses
from preprocess.tools.protein import Protein

# Global map from chain characters to integers.
ALPHANUMERIC = string.ascii_letters + string.digits + " "
CHAIN_TO_INT = {chain_char: i for i, chain_char in enumerate(ALPHANUMERIC)}
INT_TO_CHAIN = {i: chain_char for i, chain_char in enumerate(ALPHANUMERIC)}

CHAIN_FEATS = ["atom_positions", "aatype", "atom_mask", "residue_index", "b_factors"]
UNPADDED_FEATS = [
    "t",
    "rot_vectorfield_scaling",
    "trans_vectorfield_scaling",
    "t_seq",
    "t_struct",
]
RIGID_FEATS = ["rigids_0", "rigids_t"]
PAIR_FEATS = ["rel_rots"]

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
        raise ValueError(f"Invalid pad amount {pad_amt}")
    if reverse:
        pad_widths[pad_idx] = (pad_amt, 0)
    else:
        pad_widths[pad_idx] = (0, pad_amt)
    if use_torch:
        return torch.pad(x, pad_widths)
    return np.pad(x, pad_widths)


def pad_rigid(rigid: torch.tensor, max_len: int):
    num_rigids = rigid.shape[0]
    pad_amt = max_len - num_rigids
    pad_rigid = rigid_utils.Rigid.identity(
        (pad_amt,), dtype=rigid.dtype, device=rigid.device, requires_grad=False
    )
    return torch.cat([rigid, pad_rigid.to_tensor_7()], dim=0)

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

def concat_np_features(np_dicts: List[Dict[str, np.ndarray]], add_batch_dim: bool):
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
    def get_len(x):
        return x["res_mask"].shape[0]

    # get_len = lambda x: x['res_mask'].shape[0]
    # Filter out Nones! (Hacky solution to not sample more examples than necessary)

    np_dicts = [x for x in np_dicts if x is not None]
    dicts_by_length = [(get_len(x), x) for x in np_dicts]

    length_sorted = sorted(dicts_by_length, key=lambda x: x[0], reverse=True)
    if len(length_sorted) == 0:
        return torch.utils.data.default_collate([{"dummy_batch": np.random.rand(100)}])

    max_len = length_sorted[0][0]
    max_batch_examples = max(int(max_squared_res // max_len**2), 1)
    pad_example = lambda x: pad_feats(x, max_len)

    keep = length_sorted[:max_batch_examples]
    padded_batch = [pad_example(x) for (_, x) in keep]

    return torch.utils.data.default_collate(padded_batch)

def length_batching_multi_gpu(
    np_dicts: List[Dict[str, np.ndarray]],
    max_squared_res: int,
    num_gpus: int,
):
    def get_len(x):
        return x["res_mask"].shape[0]

    # get_len = lambda x: x['res_mask'].shape[0]
    # Filter out Nones! (Hacky solution to not sample more examples than necessary)
    # Split per GPU based on num_gpus

    np_dicts = [x for x in np_dicts if x is not None]

    dicts_by_length = [(get_len(x), x) for x in np_dicts]

    length_sorted = sorted(dicts_by_length, key=lambda x: x[0], reverse=True)
    max_len = length_sorted[0][0]
    max_batch_examples = max(int(max_squared_res // max_len**2), 1)
    pad_example = lambda x: pad_feats(x, max_len)
    padded_batch = [pad_example(x) for (_, x) in length_sorted[:max_batch_examples]]
    return torch.utils.data.default_collate(padded_batch)

def possible_tuple_length_batching_multi_gpu(
    x: Union[List[Dict[str, np.ndarray]], Tuple[List[Dict[str, np.ndarray]], str]],
    max_squared_res: int,
    num_gpus: int,
):
    if type(x[0]) == tuple:
        # Assume this is a validation dataset of the second type
        return length_batching_multi_gpu(
            [y[0] for y in x], max_squared_res, num_gpus
        ), [y[1] for y in x]
    else:
        return length_batching_multi_gpu(x, max_squared_res, num_gpus)


def possible_tuple_length_batching(
    x: Union[List[Dict[str, np.ndarray]], Tuple[List[Dict[str, np.ndarray]], str]],
    max_squared_res: int,
):
    if type(x[0]) == tuple:
        # Assume this is a validation dataset of the second type
        return length_batching([y[0] for y in x], max_squared_res), [y[1] for y in x]
    else:
        return length_batching(x, max_squared_res)




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
    prefetch_factor=2,
    num_gpus=1,
):
    """Creates a data loader with jax compatible data structures."""
    if np_collate:
        collate_fn = lambda x: concat_np_features(x, add_batch_dim=True)
    elif length_batch:
        if num_gpus > 1:
            collate_fn = lambda x: possible_tuple_length_batching_multi_gpu(
                x, max_squared_res=max_squared_res, num_gpus=num_gpus
            )
        else:
            collate_fn = lambda x: possible_tuple_length_batching(
                x, max_squared_res=max_squared_res
            )
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
        pin_memory=True,
        drop_last=drop_last,
        # Need fork https://github.com/facebookresearch/hydra/issues/964
        multiprocessing_context="fork"
        if num_workers != 0
        else None,  # TODO Try without. Doesn't seem to matter
    )


class OldDistributedTrainSampler(data.Sampler):
    r"""Sampler that restricts data loading to a subset of the dataset.

    modified from torch.utils.data.distributed import DistributedSampler

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(
        self,
        *,
        data_conf,
        dataset,
        batch_size,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )
        self._data_conf = data_conf
        self._dataset = dataset
        self._data_csv = self._dataset.csv
        self._dataset_indices = list(range(len(self._data_csv)))
        self._data_csv["index"] = self._dataset_indices
        # _repeated_size is the size of the dataset multiply by batch size
        self._repeated_size = batch_size * len(self._data_csv)
        self._batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and self._repeated_size % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (self._repeated_size - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(self._repeated_size / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self._data_csv), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self._data_csv)))  # type: ignore[arg-type]

        # indices is expanded by self._batch_size times
        indices = np.repeat(indices, self._batch_size)

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices = np.concatenate((indices, indices[:padding_size]), axis=0)
            else:
                indices = np.concatenate(
                    (
                        indices,
                        np.repeat(indices, math.ceil(padding_size / len(indices)))[
                            :padding_size
                        ],
                    ),
                    axis=0,
                )

        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]

        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def add_epoch(self) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = self.epoch + 1