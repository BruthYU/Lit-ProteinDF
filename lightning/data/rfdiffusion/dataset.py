import json
import lmdb
import pickle
from torch.utils import data
import tree
import torch
import numpy as np
import lightning.data.rfdiffusion.dataloader as du
from lightning.data.rfdiffusion.diffusion import Diffuser
import pandas as pd
import os
import math
import random
import logging
from omegaconf import OmegaConf
import torch.nn.functional as F
def _process_chain_feats(chain_feats):
    xyz = chain_feats['atom14_pos'].float()
    res_plddt = chain_feats['b_factors'][:, 1]
    res_mask = torch.tensor(chain_feats['res_mask']).int()
    return {
        'aatype': chain_feats['aatype'],
        'xyz': xyz,
        'res_mask': res_mask,
        'chain_idx': chain_feats["chain_idx"],
        'res_idx': chain_feats["seq_idx"],
    }

def _add_plddt_mask(feats, plddt_threshold):
    feats['plddt_mask'] = torch.tensor(
        feats['res_plddt'] > plddt_threshold).int()

def _read_clusters(cluster_path):
    pdb_to_cluster = {}
    with open(cluster_path, "r") as f:
        for i,line in enumerate(f):
            for chain in line.split(' '):
                pdb = chain.split('_')[0]
                pdb_to_cluster[pdb.upper()] = i
    return pdb_to_cluster

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

class rfdiffusion_Dataset(data.Dataset):
    def __init__(self,
                 lmdb_cache,
                 task,
                 data_conf= None,
                 diffuser_conf= None,
                 is_training= True):
        super().__init__()
        assert lmdb_cache, "No cache to build dataset."
        self.lmdb_cache = lmdb_cache
        self.csv = self.lmdb_cache.csv
        self.data_conf = data_conf
        self.diffuser_conf = diffuser_conf
        self.is_training = is_training
        self.diffuser = Diffuser(**self.diffuser_conf)

        self._rng = np.random.default_rng(seed=self.data_conf.seed)
        self._pdb_to_cluster = _read_clusters(self.data_conf.cluster_path)
        self._max_cluster = max(self._pdb_to_cluster.values())
        self._missing_pdbs = 0

        def cluster_lookup(pdb):
            pdb = pdb.split(".")[0].upper()
            if pdb not in self._pdb_to_cluster:
                self._pdb_to_cluster[pdb] = self._max_cluster + 1
                self._max_cluster += 1
                self._missing_pdbs += 1
            return self._pdb_to_cluster[pdb]

        self.csv['cluster'] = self.csv['pdb_name'].map(cluster_lookup)
        self._all_clusters = dict(
            enumerate(self.csv['cluster'].unique().tolist()))
        self._num_clusters = len(self._all_clusters)


    def process_chain_feats(self, chain_feats):
        return _process_chain_feats(chain_feats)

    def __getitem__(self, idx):
        chain_feats, gt_bb_rigid, pdb_name, csv_row = self.lmdb_cache.get_cache_csv_row(idx)
        feats = self.process_chain_feats(chain_feats)
        feats['input_seq_onehot'] = F.one_hot(feats['aatype'], num_classes=22)
        fa_stack, xyz_true = self.diffuser.diffuse_pose(feats['xyz'])

        return feats

if __name__ == '__main__':
    conf = OmegaConf.load('../../config/method/rfdiffusion.yaml')
    data_conf = conf.dataset
    diffuser_conf = conf.diffuser
    lmdb_cache = LMDB_Cache(data_conf)
    rf_dataset = rfdiffusion_Dataset(lmdb_cache, "hallucination", data_conf, diffuser_conf)
    feat_1 = rf_dataset[1]
    pass
