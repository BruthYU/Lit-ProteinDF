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
import math
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

        split_index = math.floor(len(self.csv) * self.data_conf.split_ratio)


        # Split the dataset
        if self.is_training:
            result_tuples = result_tuples[:-split_index]
            self.csv = self.csv[:-split_index]
        else:
            result_tuples = result_tuples[-split_index:]
            self.csv = self.csv[-split_index:]


            # build dictionary for pdb_name: idx
            name_idx_dict = {}
            for idx, pdb_name in enumerate(self.csv['pdb_name']):
                name_idx_dict[pdb_name] = idx
            # sample chain batch with the same length
            self.csv.drop_duplicates(subset=['pdb_name'], keep='last')
            all_lengths = np.sort(self.csv.modeled_seq_len.unique())
            length_indices = (len(all_lengths) - 1) * np.linspace(
                0.0, 1.0, self.data_conf.num_eval_lengths)
            length_indices = length_indices.astype(int)
            eval_lengths = all_lengths[length_indices]
            eval_csv = self.csv[self.csv.modeled_seq_len.isin(eval_lengths)]
            # Fix a random seed to get the same split each time.
            eval_csv = eval_csv.groupby('modeled_seq_len').sample(
                self.data_conf.samples_per_eval_length, replace=True, random_state=123)
            eval_csv = eval_csv.sort_values('modeled_seq_len', ascending=False)
            self.csv = eval_csv

            result_tuples_indices = [name_idx_dict[name] for name in self.csv['pdb_name']]
            result_tuples = [result_tuples[idx] for idx in result_tuples_indices]





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
