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



class FrameDiff_Dataset(data.Dataset):
    def __init__(self,
                 data_conf = None,
                 frame_conf = None,
                 is_training=True):
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
            diff_feats_t = self.diffuser.sample_ref(
                n_samples=gt_bb_rigid.shape[0],
                impute=gt_bb_rigid,
                diffuse_mask=None,
                as_tensor_7=True,
            )
        chain_feats.update(diff_feats_t)
        chain_feats['t'] = t

        # Convert all features to tensors.
        final_feats = tree.map_structure(
            lambda x: x if torch.is_tensor(x) else torch.tensor(x), chain_feats)
        final_feats = du.pad_feats(final_feats, csv_row['modeled_seq_len'])
        if self.is_training:
            return final_feats
        else:
            return final_feats, pdb_name

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
        # if self.is_training:
        #     result_tuples = result_tuples[:-100]
        # else:
        #     result_tuples = result_tuples[-100:]

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


class TrainSampler(data.Sampler):

    def __init__(
            self,
            *,
            data_conf,
            dataset,
            batch_size,
            sample_mode,
        ):
        self._log = logging.getLogger(__name__)
        self._data_conf = data_conf
        self._dataset = dataset
        self._data_csv = self._dataset.csv
        self._dataset_indices = list(range(len(self._data_csv)))
        self._data_csv['index'] = self._dataset_indices
        self._batch_size = batch_size
        self.epoch = 0
        self._sample_mode = sample_mode
        self.sampler_len = len(self._dataset_indices) * self._batch_size

        if self._sample_mode in ['cluster_length_batch', 'cluster_time_batch']:
            self._pdb_to_cluster = self._read_clusters()
            self._max_cluster = max(self._pdb_to_cluster.values())
            self._log.info(f'Read {self._max_cluster} clusters.')
            self._missing_pdbs = 0
            def cluster_lookup(pdb):
                pdb = pdb.upper()
                if pdb not in self._pdb_to_cluster:
                    self._pdb_to_cluster[pdb] = self._max_cluster + 1
                    self._max_cluster += 1
                    self._missing_pdbs += 1
                return self._pdb_to_cluster[pdb]
            self._data_csv['cluster'] = self._data_csv['pdb_name'].map(cluster_lookup)
            num_clusters = len(set(self._data_csv['cluster']))
            self.sampler_len = num_clusters * self._batch_size
            self._log.info(
                f'Training on {num_clusters} clusters. PDBs without clusters: {self._missing_pdbs}'
            )

    def _read_clusters(self):
        pdb_to_cluster = {}
        with open(self._data_conf.cluster_path, "r") as f:
            for i,line in enumerate(f):
                for chain in line.split(' '):
                    pdb = chain.split('_')[0]
                    pdb_to_cluster[pdb.upper()] = i
        return pdb_to_cluster

    def __iter__(self):
        if self._sample_mode == 'length_batch':
            # Each batch contains multiple proteins of the same length.
            sampled_order = self._data_csv.groupby('modeled_seq_len').sample(
                self._batch_size, replace=True, random_state=self.epoch)
            return iter(sampled_order['index'].tolist())
        elif self._sample_mode == 'time_batch':
            # Each batch contains multiple time steps of the same protein.
            random.shuffle(self._dataset_indices)
            repeated_indices = np.repeat(self._dataset_indices, self._batch_size)
            return iter(repeated_indices)
        elif self._sample_mode == 'cluster_length_batch':
            # Each batch contains multiple clusters of the same length.
            sampled_clusters = self._data_csv.groupby('cluster').sample(
                1, random_state=self.epoch)
            sampled_order = sampled_clusters.groupby('modeled_seq_len').sample(
                self._batch_size, replace=True, random_state=self.epoch)
            return iter(sampled_order['index'].tolist())
        elif self._sample_mode == 'cluster_time_batch':
            # Each batch contains multiple time steps of a protein from a cluster.
            sampled_clusters = self._data_csv.groupby('cluster').sample(
                1, random_state=self.epoch)
            dataset_indices = sampled_clusters['index'].tolist()
            repeated_indices = np.repeat(dataset_indices, self._batch_size)
            return iter(repeated_indices.tolist())
        else:
            raise ValueError(f'Invalid sample mode: {self._sample_mode}')

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.sampler_len

if __name__ == '__main__':

    instance = FrameDiff_Dataset()
