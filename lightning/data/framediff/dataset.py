import json
import lmdb
import pickle
from torch.utils import data
import tree
import torch
import numpy as np
import preprocess.tools.utils as du
from lightning.data.framediff import se3_diffuser
import pandas as pd
import os
import random
import logging



class framediff_Dataset(data.Dataset):
    def __init__(self,
                 data_conf = None,
                 frame_conf = None,
                 is_training= True):
        super().__init__()
        self.data_conf = data_conf
        self.is_training = is_training
        self.diffuser = se3_diffuser.SE3Diffuser(frame_conf)

        self.cache_dir = self.data_conf.cache_dir
        self.local_cache = None
        self.csv = None
        self.cache_to_memory()
        pass

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        # Use a fixed seed for evaluation.
        if self.is_training:
            rng = np.random.default_rng(None)
        else:
            rng = np.random.default_rng(idx)

        chain_feats, gt_bb_rigid, pdb_name, csv_row = self.get_cache_csv_row(idx)
        # Sample t and diffuse.
        if self.is_training:
            t = rng.uniform(self.data_conf.min_t, 1.0)
            diff_feats_t = self.diffuser.forward_marginal(
                rigids_0=gt_bb_rigid,
                t=t,
                diffuse_mask=None
            )
        else:
            t = 1.0
            diff_feats_t = self.diffuser.forward_marginal(
                rigids_0=gt_bb_rigid,
                t=t,
                diffuse_mask=None
            )
        chain_feats.update(diff_feats_t)
        chain_feats['t'] = t

        # Convert all features to tensors.
        final_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), chain_feats)
        final_feats = du.pad_feats(final_feats, csv_row['modeled_seq_len'])
        return final_feats

    def cache_to_memory(self):
        print(f"Loading cache from local dataset @ {self.cache_dir}")
        csv_path = os.path.join(self.cache_dir,"filtered_protein.csv")
        self.csv = pd.read_csv(csv_path)



        self.local_cache = lmdb.open(self.cache_dir)
        result_tuples = []
        with self.local_cache.begin() as txn:
            for _, value in txn.cursor():
                result_tuples.append(pickle.loads(value))


        # Split the dataset
        if self.is_training:
            result_tuples = result_tuples[:-6]
            self.csv = self.csv[:-6]
        else:
            # TODO create_split(): samples_per_eval_length
            result_tuples = result_tuples[-6:]
            self.csv = self.csv[-6:]

        assert len(result_tuples) == len(self.csv)
        def _get_list(idx):
            return list(map(lambda x: x[idx], result_tuples))
        self.chain_ftrs = _get_list(0)
        self.gt_bb_rigid_vals = _get_list(1)
        self.pdb_names = _get_list(2)
        self.csv_rows = _get_list(3)
        pass

    def get_cache_csv_row(self, idx):
        return (
            self.chain_ftrs[idx],
            self.gt_bb_rigid_vals[idx],
            self.pdb_names[idx],
            self.csv_rows[idx],
        )




if __name__ == '__main__':

    instance = framediff_Dataset()
