import json
import lmdb
import pickle
from torch.utils import data
import tree
import torch
import numpy as np

from lightning.data.framediff import se3_diffuser
import pandas as pd
from evaluate.openfold.utils import rigid_utils
import os
import math
import random
import logging


def _process_chain_feats(chain_feats):
    rigids_1 = rigid_utils.Rigid.from_tensor_4x4(chain_feats['rigidgroups_0'])[:, 0]
    rotmats_1 = rigids_1.get_rots().get_rot_mats()
    trans_1 = rigids_1.get_trans()
    res_plddt = chain_feats['b_factors'][:, 1]
    res_mask = torch.tensor(chain_feats['bb_mask']).int()

    return {
        'res_plddt': res_plddt,
        'aatype': chain_feats['aatype'],
        'rotmats_1': rotmats_1,
        'trans_1': trans_1,
        'res_mask': res_mask,
        'chain_idx': chain_feats["chain_idx"],
        'res_idx': chain_feats["seq_idx"],
    }


def _add_plddt_mask(feats, plddt_threshold):
    feats['plddt_mask'] = torch.tensor(
        feats['res_plddt'] > plddt_threshold).int()


class LMDB_Cache:
    def __init__(self, data_conf):
        self.local_cache = None
        self.csv = None
        self.cache_dir = data_conf.cache_dir
        self.cache_to_memory()

    def cache_to_memory(self):
        print(f"Loading cache from local dataset @ {self.cache_dir}")
        self.local_cache = lmdb.open(self.cache_dir)
        result_tuples = []
        with self.local_cache.begin() as txn:
            for _, value in txn.cursor():
                result_tuples.append(pickle.loads(value))

        '''
        Lmdb index may not match filtered_protein.csv due to multiprocessing,
        So we directly recover csv from the lmdb cache. 
        '''
        lmdb_series = [x[3] for x in result_tuples]
        self.csv = pd.DataFrame(lmdb_series).reset_index(drop=True)
        self.csv.to_csv("lmdb_protein.csv", index=True)

        def _get_list(idx):
            return list(map(lambda x: x[idx], result_tuples))
        self.chain_ftrs = _get_list(0)
        self.gt_bb_rigid_vals = _get_list(1)
        self.pdb_names = _get_list(2)
        self.csv_rows = _get_list(3)

    def get_cache_csv_row(self, idx):
        # if self.csv is not None:
        #     # We are going to get the idx row out of the csv -> so we look for true index based on index cl
        #     idx = self.csv.iloc[idx]["index"]

        return (
            self.chain_ftrs[idx],
            self.gt_bb_rigid_vals[idx],
            self.pdb_names[idx],
            self.csv_rows[idx],
        )


class frameflow_Dataset(data.Dataset):
    def __init__(self,
                 lmdb_cache,
                 task,
                 data_conf = None,
                 frame_conf = None,
                 is_training= True):
        super().__init__()
        assert lmdb_cache, "No cache to build dataset."
        self.lmdb_cache = lmdb_cache
        self.csv = self.lmdb_cache.csv
        self.task = task
        self.data_conf = data_conf
        self.is_training = is_training
        self.diffuser = se3_diffuser.SE3Diffuser(frame_conf)
        self._rng = np.random.default_rng(seed=self.data_conf.seed)

    def process_chain_feats(self, chain_feats):
        return _process_chain_feats(chain_feats)

    def _sample_scaffold_mask(self, feats, rng):
        trans_1 = feats['trans_1']
        num_res = trans_1.shape[0]
        min_motif_size = int(self.data_conf.min_motif_percent * num_res)
        max_motif_size = int(self.data_conf.max_motif_percent * num_res)

        # Sample the total number of residues that will be used as the motif.
        total_motif_size = self._rng.integers(
            low=min_motif_size,
            high=max_motif_size
        )

        # Sample motifs at different locations.
        num_motifs = rng.integers(low=1, high=total_motif_size)

        # Attempt to sample
        attempt = 0
        while attempt < 100:
            # Sample lengths of each motif.
            motif_lengths = np.sort(
                rng.integers(
                    low=1,
                    high=max_motif_size,
                    size=(num_motifs,)
                )
            )

            # Truncate motifs to not go over the motif length.
            cumulative_lengths = np.cumsum(motif_lengths)
            motif_lengths = motif_lengths[cumulative_lengths < total_motif_size]
            if len(motif_lengths) == 0:
                attempt += 1
            else:
                break
        if len(motif_lengths) == 0:
            motif_lengths = [total_motif_size]

        # Sample start location of each motif.
        seed_residues = rng.integers(
            low=0,
            high=num_res-1,
            size=(len(motif_lengths),)
        )

        # Construct the motif mask.
        motif_mask = torch.zeros(num_res)
        for motif_seed, motif_len in zip(seed_residues, motif_lengths):
            motif_mask[motif_seed:min(motif_seed+motif_len, num_res)] = 1.0
        scaffold_mask = 1 - motif_mask
        return scaffold_mask * feats['res_mask']

    def setup_inpainting(self, feats, rng):
        diffuse_mask = self._sample_scaffold_mask(feats, rng)
        if 'plddt_mask' in feats:
            diffuse_mask = diffuse_mask * feats['plddt_mask']
        if torch.sum(diffuse_mask) < 1:
            # Should only happen rarely.
            diffuse_mask = torch.ones_like(diffuse_mask)
        feats['diffuse_mask'] = diffuse_mask

    def __getitem__(self, idx):
        # Process data example.
        chain_feats, gt_bb_rigid, pdb_name, csv_row = self.lmdb_cache.get_cache_csv_row(idx)
        feats = self.process_chain_feats(chain_feats)

        if self.data_conf.add_plddt_mask:
            _add_plddt_mask(feats, self.data_conf.min_plddt_threshold)
        else:
            feats['plddt_mask'] = torch.ones_like(feats['res_mask'])

        if self.task == 'hallucination':
            feats['diffuse_mask'] = torch.ones_like(feats['res_mask']).bool()
        elif self.task == 'inpainting':
            if self.data_conf.inpainting_percent < random.random():
                feats['diffuse_mask'] = torch.ones_like(feats['res_mask'])
            else:
                rng = self._rng if self.is_training else np.random.default_rng(seed=123)
                self.setup_inpainting(feats, rng)
                # Center based on motif locations
                motif_mask = 1 - feats['diffuse_mask']
                trans_1 = feats['trans_1']
                motif_1 = trans_1 * motif_mask[:, None]
                motif_com = torch.sum(motif_1, dim=0) / (torch.sum(motif_mask) + 1)
                trans_1 -= motif_com[None, :]
                feats['trans_1'] = trans_1
        else:
            raise ValueError(f'Unknown task {self.task}')
        feats['diffuse_mask'] = feats['diffuse_mask'].int()

        # Storing the csv index is helpful for debugging.
        feats['csv_idx'] = torch.ones(1, dtype=torch.long) * idx
        return feats


